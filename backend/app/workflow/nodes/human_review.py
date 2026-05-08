"""
Node: queue_for_review

Terminal node in the workflow graph.

Assembles a ReviewItem from the accumulated workflow state and persists it to
MongoDB via the review repository. The item is placed in the human review queue
with a priority score based on escalation/confidence signals.

After this node the graph reaches END. The human reviewer picks the item from
the queue via the API, takes an action, and triggers memory updates.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.repositories.mongo.review_repo import ReviewRepository
from app.schemas.contracts import (
    ConflictDecision,
    ConflictDecisionType,
    CriticViolation,
    PatchProposal,
    ReviewItem,
    ReviewStatus,
    TranslationCandidate,
)
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Module-level repository singleton (injected at startup)
# ─────────────────────────────────────────────────────────────────

_review_repo: ReviewRepository | None = None


def init_review_repo(review_repo: ReviewRepository) -> None:
    """Call once at application startup."""
    global _review_repo
    _review_repo = review_repo


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def queue_for_review_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    book_id = state["book_id"]
    chunk_id = state["chunk_id"]
    source_text = state["source_text"]

    # Determine best candidate
    best_candidate_id = state.get("style_critic_best_candidate_id") or ""
    candidates: list[dict] = state.get("candidates", [])
    best_candidate_dict = _find_candidate(candidates, best_candidate_id)

    best_candidate: TranslationCandidate
    if best_candidate_dict:
        best_candidate = TranslationCandidate(**best_candidate_dict)
    elif candidates:
        best_candidate = TranslationCandidate(**candidates[0])
    else:
        best_candidate = TranslationCandidate(
            generator_id="faithful", text="", token_count=0
        )

    all_candidates = [TranslationCandidate(**c) for c in candidates]

    # Collect violations for display
    det_violations: list[CriticViolation] = []
    for r in state.get("deterministic_results", []):
        if r.get("candidate_id") == best_candidate.candidate_id:
            det_violations.extend([CriticViolation(**v) for v in r.get("violations", [])])

    # Style issues for best candidate
    style_issues: list[dict] = []
    for r in state.get("style_critic_results", []):
        if r.get("candidate_id") == best_candidate.candidate_id:
            style_issues = r.get("issues", [])
            break

    # Patch proposal (if any)
    patch_proposal: PatchProposal | None = None
    pp = state.get("patch_proposal")
    if pp:
        patch_proposal = PatchProposal(**pp)
        # If a patch exists, use its text as the best candidate text
        best_candidate = TranslationCandidate(
            candidate_id=best_candidate.candidate_id,
            generator_id=best_candidate.generator_id,
            text=patch_proposal.patched_text,
            token_count=best_candidate.token_count,
        )

    # Conflict decision
    conflict_decision: ConflictDecision | None = None
    cd = state.get("conflict_decision")
    if cd:
        conflict_decision = ConflictDecision(**cd)

    # Escalation reason
    escalation_reason: str | None = conflict_decision.escalation_reason if conflict_decision else None
    is_escalated = (
        state.get("has_critical_failure", False)
        or (
            conflict_decision is not None
            and conflict_decision.decision == ConflictDecisionType.ESCALATE_TO_HUMAN.value
        )
    )

    # Priority: escalated items get higher priority
    confidence = state.get("style_critic_confidence", 0.5)
    priority = 10 if is_escalated else int((1.0 - confidence) * 5)

    review_item = ReviewItem(
        book_id=book_id,
        chunk_id=chunk_id,
        source_text=source_text,
        best_candidate=best_candidate,
        all_candidates=all_candidates,
        deterministic_violations=det_violations,
        style_issues=style_issues,
        patch_proposal=patch_proposal,
        conflict_decision=conflict_decision,
        escalation_reason=escalation_reason,
        status=ReviewStatus.ESCALATED if is_escalated else ReviewStatus.PENDING,
        priority=priority,
    )

    review_item_id = review_item.id

    if _review_repo is not None:
        try:
            await _review_repo.create(review_item)
        except Exception as exc:
            logger.error(
                "queue_for_review: failed to persist ReviewItem for chunk %s: %s",
                chunk_id, exc,
            )
    else:
        logger.warning(
            "queue_for_review: review repository not initialised, item %s not persisted",
            review_item_id,
        )

    elapsed = time.perf_counter() - t0
    return {
        "review_item_id": review_item_id,
        "node_timings": {"queue_for_review": round(elapsed, 3)},
        "audit_trail": [{
            "node": "queue_for_review",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "review_item_id": review_item_id,
            "status": review_item.status,
            "priority": priority,
            "is_escalated": is_escalated,
        }],
    }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _find_candidate(candidates: list[dict], candidate_id: str) -> dict | None:
    for c in candidates:
        if c.get("candidate_id") == candidate_id:
            return c
    return None
