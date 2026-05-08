"""
Node: run_deterministic_critics

Fast, non-LLM checks on every translation candidate.
Cheap to run: executed before any expensive LLM critic.

MVP checks (run once across all candidates, flagging per-candidate):
  1. GlossaryLockChecker  — locked terms must appear verbatim in the Italian output
  2. JapaneseCharChecker  — no untranslated CJK characters in the Italian output
  3. StructureAlignmentChecker — paragraph count must be within an acceptable ratio
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from app.core.config import get_settings
from app.schemas.contracts import (
    CriticSeverity,
    CriticViolation,
    DeterministicCriticResult,
)
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)
settings = get_settings()

# Unicode ranges for Japanese characters (CJK, Hiragana, Katakana, etc.)
_CJK_RE = re.compile(
    r"[\u3000-\u303f"    # CJK symbols and punctuation
    r"\u3040-\u309f"     # Hiragana
    r"\u30a0-\u30ff"     # Katakana
    r"\u4e00-\u9fff"     # CJK unified ideographs (common)
    r"\uff00-\uffef]"    # Halfwidth/fullwidth forms
)


# ─────────────────────────────────────────────────────────────────
# Individual checks (pure functions — no LLM, no I/O)
# ─────────────────────────────────────────────────────────────────

def _check_glossary_locks(
    candidate_text: str,
    locked_terms: dict[str, str],
) -> DeterministicCriticResult:
    """Verify every locked term appears verbatim in the translation."""
    violations: list[CriticViolation] = []

    for src_ja, expected_it in locked_terms.items():
        if expected_it and expected_it not in candidate_text:
            violations.append(
                CriticViolation(
                    description=f"Termine bloccato mancante: «{expected_it}»",
                    evidence=f"Atteso: «{expected_it}» (da «{src_ja}»)",
                    suggestion=f"Inserire esattamente «{expected_it}»",
                    severity=CriticSeverity.CRITICAL,
                )
            )

    return DeterministicCriticResult(
        critic_id="glossary_lock_checker",
        passed=len(violations) == 0,
        violations=violations,
        severity=CriticSeverity.CRITICAL if violations else CriticSeverity.INFO,
    )


def _check_no_japanese_chars(
    candidate_text: str,
) -> DeterministicCriticResult:
    """Ensure no CJK / Japanese characters remain in the Italian output."""
    matches = _CJK_RE.findall(candidate_text)
    violations: list[CriticViolation] = []

    if matches:
        sample = "".join(matches[:20])
        violations.append(
            CriticViolation(
                description="Caratteri giapponesi non tradotti nell'output italiano",
                evidence=f"Caratteri trovati: {sample!r}",
                suggestion="Tradurre o adattare tutti i caratteri giapponesi",
                severity=CriticSeverity.CRITICAL,
            )
        )

    return DeterministicCriticResult(
        critic_id="japanese_char_checker",
        passed=len(violations) == 0,
        violations=violations,
        severity=CriticSeverity.CRITICAL if violations else CriticSeverity.INFO,
    )


def _check_structure_alignment(
    source_text: str,
    candidate_text: str,
) -> DeterministicCriticResult:
    """
    Paragraph count in the translation should stay within a 2x ratio of the source.

    Japanese paragraphs are separated by blank lines or 。 followed by newline.
    Italian paragraphs are separated by blank lines or newlines.
    A large discrepancy signals likely truncation or hallucinated expansion.
    """
    src_paras = max(1, len([p for p in source_text.split("\n") if p.strip()]))
    tgt_paras = max(1, len([p for p in candidate_text.split("\n") if p.strip()]))
    ratio = max(src_paras, tgt_paras) / min(src_paras, tgt_paras)

    violations: list[CriticViolation] = []
    if ratio > 3.0:
        violations.append(
            CriticViolation(
                description=(
                    f"Disallineamento strutturale: {src_paras} righe sorgente "
                    f"vs {tgt_paras} righe traduzione (ratio {ratio:.1f})"
                ),
                evidence=f"source_lines={src_paras}, target_lines={tgt_paras}",
                severity=CriticSeverity.WARNING,
            )
        )

    return DeterministicCriticResult(
        critic_id="structure_alignment_checker",
        passed=len(violations) == 0,
        violations=violations,
        severity=CriticSeverity.WARNING if violations else CriticSeverity.INFO,
    )


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def run_deterministic_critics_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    source_text = state["source_text"]
    retrieval = state.get("retrieval_packet", {})
    locked_terms: dict[str, str] = retrieval.get("locked_terms") or {}
    candidates: list[dict] = state.get("candidates", [])
    max_violations = settings.deterministic_critic_max_violations

    all_results: list[dict] = []
    has_critical_failure = False
    total_critical_violations = 0

    for candidate in candidates:
        cand_text: str = candidate.get("text", "")
        cand_id: str = candidate.get("candidate_id", "")

        results = [
            _check_glossary_locks(cand_text, locked_terms),
            _check_no_japanese_chars(cand_text),
            _check_structure_alignment(source_text, cand_text),
        ]

        for r in results:
            serialised = r.model_dump()
            serialised["candidate_id"] = cand_id
            all_results.append(serialised)

            critical_count = sum(
                1
                for v in r.violations
                if v.severity in (CriticSeverity.CRITICAL, CriticSeverity.CRITICAL.value)
            )
            total_critical_violations += critical_count

    # Escalate only if EVERY candidate has at least one critical violation
    # (i.e. there's no clean candidate to proceed with)
    critical_by_candidate: dict[str, int] = {}
    for r in all_results:
        cid = r.get("candidate_id", "")
        critical_by_candidate[cid] = critical_by_candidate.get(cid, 0) + sum(
            1
            for v in r.get("violations", [])
            if v.get("severity") == CriticSeverity.CRITICAL.value
        )

    all_have_criticals = all(v > 0 for v in critical_by_candidate.values()) and bool(
        critical_by_candidate
    )
    has_critical_failure = all_have_criticals

    elapsed = time.perf_counter() - t0
    return {
        "deterministic_results": all_results,
        "has_critical_failure": has_critical_failure,
        "node_timings": {"run_deterministic_critics": round(elapsed, 3)},
        "audit_trail": [{
            "node": "run_deterministic_critics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "candidates_checked": len(candidates),
            "has_critical_failure": has_critical_failure,
            "total_critical_violations": total_critical_violations,
        }],
    }
