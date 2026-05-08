"""
Node: synthesize_patch

Applies LOCALISED edits to the best translation candidate.
Never rewrites the whole text — preserves good regions unchanged.

Strategy:
  1. Build a minimal prompt with ONLY the patchable regions identified by the
     style critic (not the full text again).
  2. Call the LLM patch model to produce replacement text for each region.
  3. Apply the replacements sequentially.
  4. Re-run deterministic critics on the patched output.
  5. If the patched output still has critical violations → escalate to human.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.contracts import (
    CriticSeverity,
    PatchProposal,
    PatchRegion,
)
from app.workflow.nodes.deterministic_critics import (
    _check_glossary_locks,
    _check_no_japanese_chars,
)
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────

_SYSTEM = """\
Sei un editor letterario italiano che applica correzioni minime e mirate.
Ricevi una traduzione italiana con problemi specifici identificati.
Apporta SOLO le modifiche necessarie a risolvere i problemi elencati.
Preserva invariate tutte le parti non menzionate.
Rispetta ESATTAMENTE i termini bloccati. Non riscrivere il testo intero.
Restituisci SOLO il testo italiano corretto, senza commenti."""

_USER_TEMPLATE = """\
SORGENTE (Giapponese) — solo per riferimento contestuale:
{source_text}

TRADUZIONE DA CORREGGERE:
{current_translation}

PROBLEMI DA RISOLVERE:
{issues_block}

TERMINI BLOCCATI (non modificare):
{locked_terms_block}

Restituisci il testo italiano corretto."""


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def synthesize_patch_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    chunk_id = state["chunk_id"]
    source_text = state["source_text"]
    retrieval = state.get("retrieval_packet", {})
    locked_terms: dict[str, str] = retrieval.get("locked_terms") or {}

    # Find the best candidate text
    best_candidate_id = state.get("style_critic_best_candidate_id") or ""
    candidates: list[dict] = state.get("candidates", [])
    best_candidate = _find_candidate(candidates, best_candidate_id)

    if best_candidate is None:
        logger.warning("synthesize_patch: no best candidate found for chunk %s", chunk_id)
        elapsed = time.perf_counter() - t0
        return _escalate(chunk_id, elapsed, "Nessun candidato valido trovato per il patch")

    current_translation: str = best_candidate.get("text", "")

    # Collect patchable issues from style critic results
    style_results: list[dict] = state.get("style_critic_results", [])
    patch_regions_raw = _collect_patch_regions(style_results, best_candidate_id)

    if not patch_regions_raw:
        # Nothing specific to patch — accept as-is
        proposal = _build_proposal(
            chunk_id=chunk_id,
            base_candidate_id=best_candidate_id,
            original_text=current_translation,
            patched_text=current_translation,
            regions=[],
            confidence=state.get("style_critic_confidence", 0.5),
            passes_deterministic=True,
        )
        elapsed = time.perf_counter() - t0
        return _success(proposal, elapsed, tokens=0)

    issues_block = _format_issues(patch_regions_raw)
    locked_terms_block = (
        "\n".join(f"  {ja} → {it}" for ja, it in locked_terms.items())
        or "Nessuno"
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.google_model_patch,
            temperature=0.2,
            google_api_key=settings.google_api_key,
        )
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_USER_TEMPLATE.format(
                source_text=source_text,
                current_translation=current_translation,
                issues_block=issues_block,
                locked_terms_block=locked_terms_block,
            )),
        ]
        response = await llm.ainvoke(messages)
        patched_text: str = response.content.strip()

        usage = getattr(response, "usage_metadata", None)
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

        # Re-run deterministic critics on patched output
        lock_check = _check_glossary_locks(patched_text, locked_terms)
        jp_check = _check_no_japanese_chars(patched_text)
        passes = lock_check.passed and jp_check.passed

        if not passes:
            # Patched output still fails — escalate
            elapsed = time.perf_counter() - t0
            reason = "; ".join(
                v.description
                for r in [lock_check, jp_check]
                for v in r.violations
            )
            return _escalate(
                chunk_id, elapsed,
                f"Patch fallita i controlli deterministici: {reason}",
                tokens=tokens,
            )

        # Build PatchRegion audit objects from the suggestions
        regions = [
            PatchRegion(
                original_text=pr.get("original_span_text", ""),
                replacement_text=pr.get("suggested_replacement") or "",
                rationale=pr.get("issue_description", ""),
                contributing_critics=["style_critic"],
            )
            for pr in patch_regions_raw
        ]

        proposal = _build_proposal(
            chunk_id=chunk_id,
            base_candidate_id=best_candidate_id,
            original_text=current_translation,
            patched_text=patched_text,
            regions=regions,
            confidence=min(state.get("style_critic_confidence", 0.5) + 0.15, 1.0),
            passes_deterministic=True,
        )

        elapsed = time.perf_counter() - t0
        return _success(proposal, elapsed, tokens=tokens)

    except Exception as exc:
        logger.error("synthesize_patch failed for chunk %s: %s", chunk_id, exc)
        elapsed = time.perf_counter() - t0
        return _escalate(chunk_id, elapsed, f"Errore durante la sintesi del patch: {exc}")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _find_candidate(candidates: list[dict], candidate_id: str) -> dict | None:
    for c in candidates:
        if c.get("candidate_id") == candidate_id:
            return c
    return candidates[0] if candidates else None


def _collect_patch_regions(
    style_results: list[dict],
    best_candidate_id: str,
) -> list[dict]:
    for r in style_results:
        if r.get("candidate_id") == best_candidate_id:
            return [
                pr for pr in (r.get("patch_regions") or [])
                if pr.get("severity") in (
                    CriticSeverity.CRITICAL.value,
                    CriticSeverity.WARNING.value,
                )
            ]
    return []


def _format_issues(patch_regions: list[dict]) -> str:
    lines = []
    for i, pr in enumerate(patch_regions, 1):
        lines.append(
            f"{i}. TESTO ORIGINALE: «{pr.get('original_span_text', '')}»\n"
            f"   PROBLEMA: {pr.get('issue_description', '')}\n"
            f"   SOSTITUZIONE SUGGERITA: {pr.get('suggested_replacement') or 'non specificata'}"
        )
    return "\n\n".join(lines)


def _build_proposal(
    chunk_id: str,
    base_candidate_id: str,
    original_text: str,
    patched_text: str,
    regions: list[PatchRegion],
    confidence: float,
    passes_deterministic: bool,
) -> PatchProposal:
    return PatchProposal(
        chunk_id=chunk_id,
        base_candidate_id=base_candidate_id,
        original_text=original_text,
        patched_text=patched_text,
        regions=regions,
        confidence=confidence,
        passes_deterministic=passes_deterministic,
    )


def _success(proposal: PatchProposal, elapsed: float, tokens: int) -> dict:
    return {
        "patch_proposal": proposal.model_dump(),
        "total_tokens": tokens,
        "node_timings": {"synthesize_patch": round(elapsed, 3)},
        "audit_trail": [{
            "node": "synthesize_patch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "regions_patched": len(proposal.regions),
            "passes_deterministic": proposal.passes_deterministic,
        }],
    }


def _escalate(
    chunk_id: str,
    elapsed: float,
    reason: str,
    tokens: int = 0,
) -> dict:
    """Override the conflict decision to force human escalation."""
    from app.schemas.contracts import ConflictDecision, ConflictDecisionType
    escalated_decision = ConflictDecision(
        decision=ConflictDecisionType.ESCALATE_TO_HUMAN,
        best_candidate_id="",
        conflict_type="patch_synthesis_failure",
        reasoning=reason,
        escalation_reason=reason,
    )
    return {
        "conflict_decision": escalated_decision.model_dump(),
        "total_tokens": tokens,
        "node_timings": {"synthesize_patch": round(elapsed, 3)},
        "audit_trail": [{
            "node": "synthesize_patch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "escalated": True,
            "reason": reason,
        }],
    }
