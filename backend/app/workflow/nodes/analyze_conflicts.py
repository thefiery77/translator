"""
Node: analyze_conflicts

Analyses disagreements between critic signals and produces a ConflictDecision
with one of three outcomes:

  accept_best_candidate  — style critic is reasonably confident; accept as-is
  request_local_patch    — patchable issues detected; synthesize local corrections
  escalate_to_human      — structural/critical issues or low confidence; human needed

Decision logic (rule-based, no additional LLM call — keeps cost low):

  Priority 1 — deterministic critical violations on best candidate → escalate
  Priority 2 — style critic confidence < low_threshold              → escalate
  Priority 3 — style critic has critical-severity patch regions     → local_patch
  Priority 4 — style critic score < mid_threshold                   → local_patch
  Default    — accept_best_candidate
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.schemas.contracts import ConflictDecision, ConflictDecisionType, CriticSeverity
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.40
_MID_SCORE_THRESHOLD = 0.60

CONFLICT_TYPES = {
    "critical_deterministic": "deterministic_failure",
    "low_confidence": "low_style_confidence",
    "critical_style_patch": "style_vs_fidelity",
    "low_style_score": "style_quality",
}


async def analyze_conflicts_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    chunk_id = state["chunk_id"]

    best_candidate_id: str = state.get("style_critic_best_candidate_id") or _first_candidate_id(state)
    confidence: float = state.get("style_critic_confidence", 0.0)
    det_results: list[dict] = state.get("deterministic_results", [])
    style_results: list[dict] = state.get("style_critic_results", [])

    decision, conflict_type, escalation_reason = _decide(
        best_candidate_id=best_candidate_id,
        confidence=confidence,
        det_results=det_results,
        style_results=style_results,
    )

    conflict_decision = ConflictDecision(
        decision=decision,
        best_candidate_id=best_candidate_id,
        conflict_type=conflict_type,
        reasoning=_build_reasoning(decision, confidence, conflict_type),
        escalation_reason=escalation_reason,
    )

    elapsed = time.perf_counter() - t0
    return {
        "conflict_decision": conflict_decision.model_dump(),
        "node_timings": {"analyze_conflicts": round(elapsed, 3)},
        "audit_trail": [{
            "node": "analyze_conflicts",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "decision": decision.value,
            "conflict_type": conflict_type,
            "best_candidate_id": best_candidate_id,
            "confidence": confidence,
        }],
    }


# ─────────────────────────────────────────────────────────────────
# Decision logic
# ─────────────────────────────────────────────────────────────────

def _decide(
    best_candidate_id: str,
    confidence: float,
    det_results: list[dict],
    style_results: list[dict],
) -> tuple[ConflictDecisionType, str | None, str | None]:
    """Return (decision, conflict_type, escalation_reason)."""

    # Priority 1: deterministic critical violation on best candidate
    for r in det_results:
        if r.get("candidate_id") == best_candidate_id:
            for v in r.get("violations", []):
                if v.get("severity") == CriticSeverity.CRITICAL.value:
                    return (
                        ConflictDecisionType.ESCALATE_TO_HUMAN,
                        CONFLICT_TYPES["critical_deterministic"],
                        f"Violazione critica deterministica: {v.get('description', '')}",
                    )

    # Priority 2: style critic confidence below low threshold
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        return (
            ConflictDecisionType.ESCALATE_TO_HUMAN,
            CONFLICT_TYPES["low_confidence"],
            f"Confidenza style critic troppo bassa ({confidence:.2f} < {_LOW_CONFIDENCE_THRESHOLD})",
        )

    # Priority 3: critical-severity patch regions in style results
    for r in style_results:
        if r.get("candidate_id") == best_candidate_id:
            for pr in r.get("patch_regions", []):
                if pr.get("severity") == CriticSeverity.CRITICAL.value:
                    return (
                        ConflictDecisionType.REQUEST_LOCAL_PATCH,
                        CONFLICT_TYPES["critical_style_patch"],
                        None,
                    )

    # Priority 4: best candidate style score below mid threshold
    for r in style_results:
        if r.get("candidate_id") == best_candidate_id:
            if float(r.get("score", 1.0)) < _MID_SCORE_THRESHOLD:
                return (
                    ConflictDecisionType.REQUEST_LOCAL_PATCH,
                    CONFLICT_TYPES["low_style_score"],
                    None,
                )

    return ConflictDecisionType.ACCEPT_BEST_CANDIDATE, None, None


def _build_reasoning(
    decision: ConflictDecisionType,
    confidence: float,
    conflict_type: str | None,
) -> str:
    if decision == ConflictDecisionType.ACCEPT_BEST_CANDIDATE:
        return f"Candidato accettato direttamente (confidenza {confidence:.2f})."
    if decision == ConflictDecisionType.REQUEST_LOCAL_PATCH:
        return f"Patch locale richiesta per risolvere il conflitto «{conflict_type}»."
    return f"Escalation a revisione umana per conflitto «{conflict_type}»."


def _first_candidate_id(state: ChunkWorkflowState) -> str:
    candidates = state.get("candidates") or []
    if candidates:
        return candidates[0].get("candidate_id", "")
    return ""
