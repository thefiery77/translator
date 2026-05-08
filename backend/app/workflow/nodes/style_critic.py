"""
Node: run_style_critic

Single LLM-based style critic that evaluates all translation candidates and
picks the best one. Returns a structured assessment with:
  - per-candidate score and confidence
  - specific patchable regions
  - the best candidate ID
  - overall confidence

Only invoked if deterministic critics pass (cascading critics pattern).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.contracts import CriticSeverity, PatchRegionSuggestion, StyleCriticResult
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────

_SYSTEM = """\
Sei un editor letterario italiano esperto e uno specialista di letteratura giapponese.
Valuta le traduzioni dal giapponese all'italiano fornite per qualità stilistica.
Considera:
  - naturalezza e musicalità della prosa italiana
  - registro e tono (coerenza con le linee guida stilistiche)
  - resa delle onomatopee giapponesi
  - adattamento culturale appropriato
  - ritmo e lunghezza delle frasi
  - fedeltà al livello di formalità del sorgente (keigo)

Restituisci SOLO JSON valido. Nessun testo aggiuntivo."""

_USER_TEMPLATE = """\
SORGENTE (Giapponese):
{source_text}

LINEE GUIDA STILISTICHE:
{style_summary}

CANDIDATI:
{candidates_block}

Restituisci un JSON con questa struttura:
{{
  "evaluations": [
    {{
      "candidate_id": "<id del candidato>",
      "generator_id": "<faithful|creative|contextual>",
      "score": 0.0-1.0,
      "confidence": 0.0-1.0,
      "issues": [
        {{
          "type": "<rhythm|register|unnatural|omission|cultural|honorific>",
          "description": "...",
          "evidence": "citazione esatta dal testo del candidato",
          "severity": "<critical|warning|info>"
        }}
      ],
      "patch_regions": [
        {{
          "original_span_text": "testo esatto nel candidato",
          "issue_description": "...",
          "suggested_replacement": "...",
          "severity": "<critical|warning|info>"
        }}
      ],
      "overall_notes": "..."
    }}
  ],
  "best_candidate_id": "<candidate_id con il punteggio più alto>",
  "overall_confidence": 0.0-1.0
}}"""


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def run_style_critic_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    chunk_id = state["chunk_id"]
    source_text = state["source_text"]
    candidates: list[dict] = state.get("candidates", [])
    retrieval = state.get("retrieval_packet", {})
    style_summary = retrieval.get("style_summary") or "Nessuna linea guida stilistica disponibile."

    if not candidates:
        elapsed = time.perf_counter() - t0
        return {
            "style_critic_results": [],
            "style_critic_best_candidate_id": "",
            "style_critic_confidence": 0.0,
            "node_timings": {"run_style_critic": round(elapsed, 3)},
            "audit_trail": [{
                "node": "run_style_critic",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "no candidates to evaluate",
            }],
        }

    # Build candidates block for prompt
    candidates_block = "\n\n".join(
        f"[Candidato {i+1}]\n"
        f"candidate_id: {c.get('candidate_id')}\n"
        f"generator_id: {c.get('generator_id')}\n"
        f"---\n{c.get('text', '')}"
        for i, c in enumerate(candidates)
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.google_model_critic,
            temperature=0,
            google_api_key=settings.google_api_key,
        )
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_USER_TEMPLATE.format(
                source_text=source_text,
                style_summary=style_summary,
                candidates_block=candidates_block,
            )),
        ]
        response = await llm.ainvoke(messages)
        raw = _parse_json(response.content)

        evaluations: list[dict] = raw.get("evaluations") or []
        best_candidate_id: str = raw.get("best_candidate_id") or (
            candidates[0].get("candidate_id", "") if candidates else ""
        )
        overall_confidence: float = float(raw.get("overall_confidence") or 0.0)

        # Map evaluations to StyleCriticResult objects
        results: list[dict] = []
        for ev in evaluations:
            patch_regions = [
                PatchRegionSuggestion(
                    candidate_id=ev.get("candidate_id", ""),
                    original_span_text=pr.get("original_span_text", ""),
                    issue_description=pr.get("issue_description", ""),
                    suggested_replacement=pr.get("suggested_replacement"),
                    severity=_safe_severity(pr.get("severity", "warning")),
                ).model_dump()
                for pr in (ev.get("patch_regions") or [])
            ]
            scr = StyleCriticResult(
                candidate_id=ev.get("candidate_id", ""),
                score=float(ev.get("score") or 0.0),
                confidence=float(ev.get("confidence") or 0.0),
                issues=ev.get("issues") or [],
                patch_regions=[PatchRegionSuggestion(**pr) for pr in patch_regions],
                overall_notes=ev.get("overall_notes") or "",
            )
            results.append(scr.model_dump())

        usage = getattr(response, "usage_metadata", None)
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

        elapsed = time.perf_counter() - t0
        return {
            "style_critic_results": results,
            "style_critic_best_candidate_id": best_candidate_id,
            "style_critic_confidence": overall_confidence,
            "total_tokens": tokens,
            "node_timings": {"run_style_critic": round(elapsed, 3)},
            "audit_trail": [{
                "node": "run_style_critic",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(elapsed, 3),
                "best_candidate_id": best_candidate_id,
                "overall_confidence": overall_confidence,
                "candidates_evaluated": len(results),
            }],
        }

    except Exception as exc:
        logger.error("run_style_critic failed for chunk %s: %s", chunk_id, exc)
        # Fall back to first candidate with zero confidence → triggers conflict analyzer
        best = candidates[0].get("candidate_id", "") if candidates else ""
        elapsed = time.perf_counter() - t0
        return {
            "style_critic_results": [],
            "style_critic_best_candidate_id": best,
            "style_critic_confidence": 0.0,
            "node_timings": {"run_style_critic": round(elapsed, 3)},
            "audit_trail": [{
                "node": "run_style_critic",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }],
        }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    return json.loads(text)


def _safe_severity(value: str) -> CriticSeverity:
    try:
        return CriticSeverity(value)
    except ValueError:
        return CriticSeverity.WARNING
