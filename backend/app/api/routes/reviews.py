"""
API routes — Human review queue and review actions.

Endpoints:
  GET  /reviews                       — list pending review items (filterable by book)
  GET  /reviews/{review_id}           — get full review item detail
  POST /reviews/{review_id}/actions   — submit a review action
  GET  /reviews/{review_id}/actions   — list past actions for an item
  POST /glossary/{book_id}/lock       — lock a glossary term directly
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.repositories.mongo.review_repo import ReviewRepository
from app.schemas.contracts import (
    ReviewActionRecord,
    ReviewActionRequest,
    ReviewActionType,
    ReviewStatus,
)
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reviews"])


# ─────────────────────────────────────────────────────────────────
# Review queue
# ─────────────────────────────────────────────────────────────────

@router.get("/reviews")
async def list_review_queue(
    request: Request,
    book_id: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Return pending and escalated review items, sorted by priority desc."""
    repo: ReviewRepository = request.app.state.review_repo
    items = await repo.list_pending(book_id=book_id, limit=limit, offset=offset)
    return {
        "items": [
            {
                "review_id": item.id,
                "book_id": item.book_id,
                "chunk_id": item.chunk_id,
                "status": item.status,
                "priority": item.priority,
                "escalation_reason": item.escalation_reason,
                "source_preview": item.source_text[:100],
                "best_translation_preview": item.best_candidate.text[:100],
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ],
        "count": len(items),
    }


@router.get("/reviews/{review_id}")
async def get_review_item(review_id: str, request: Request):
    """Return the full review item with all critic warnings and patch proposals."""
    repo: ReviewRepository = request.app.state.review_repo
    item = await repo.get(review_id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item non trovato")
    return item.model_dump()


# ─────────────────────────────────────────────────────────────────
# Review actions
# ─────────────────────────────────────────────────────────────────

@router.post("/reviews/{review_id}/actions", status_code=status.HTTP_201_CREATED)
async def submit_review_action(
    request: Request,
    review_id: str,
    payload: ReviewActionRequest,
    reviewer_id: str = Query(..., description="Reviewer identifier"),
):
    """
    Submit a human review action for an item.

    Actions:
      approve         — accept best candidate (or patched text) as-is
      edit            — accept with manual correction (final_text required)
      reject          — mark as rejected, keep in queue for reassignment
      lock_term       — lock a term in translation memory (term_source + term_target required)
      override_critic — dismiss a critic warning (overridden_critic_id required)
    """
    repo: ReviewRepository = request.app.state.review_repo
    memory_svc: MemoryService = request.app.state.memory_service

    item = await repo.get(review_id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item non trovato")

    if item.status not in (
        ReviewStatus.PENDING.value, ReviewStatus.ESCALATED.value
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Item già processato (status: {item.status})",
        )

    # Validate required fields per action type
    if payload.action_type == ReviewActionType.EDIT and not payload.final_text:
        raise HTTPException(
            status_code=422,
            detail="final_text è obbligatorio per l'azione 'edit'",
        )
    if payload.action_type == ReviewActionType.LOCK_TERM:
        if not payload.term_source or not payload.term_target:
            raise HTTPException(
                status_code=422,
                detail="term_source e term_target sono obbligatori per 'lock_term'",
            )

    # Record the action
    action = ReviewActionRecord(
        review_item_id=review_id,
        book_id=item.book_id,
        chunk_id=item.chunk_id,
        reviewer_id=reviewer_id,
        action_type=payload.action_type,
        final_text=payload.final_text,
        notes=payload.notes,
    )
    await repo.record_action(action)

    # Update item status
    new_status = _resolve_status(payload.action_type)
    await repo.update_status(review_id, new_status)

    # Trigger memory updates (anti-contamination: only approve/edit promote memory)
    try:
        if payload.action_type == ReviewActionType.LOCK_TERM:
            await memory_svc.lock_term(
                book_id=item.book_id,
                source_ja=payload.term_source,          # type: ignore[arg-type]
                target_it=payload.term_target,           # type: ignore[arg-type]
                reviewer_id=reviewer_id,
            )
        else:
            await memory_svc.update_from_review_action(action, item)
    except Exception as exc:
        logger.error(
            "Memory update failed after review action on %s: %s", review_id, exc
        )
        # Memory update failure is non-fatal; action is still recorded

    return {
        "action_id": action.id,
        "review_id": review_id,
        "new_status": new_status.value,
    }


@router.get("/reviews/{review_id}/actions")
async def list_review_actions(review_id: str, request: Request):
    """Audit trail of all actions taken on a review item."""
    repo: ReviewRepository = request.app.state.review_repo
    actions = await repo.list_actions_for_item(review_id)
    return [a.model_dump() for a in actions]


# ─────────────────────────────────────────────────────────────────
# Glossary
# ─────────────────────────────────────────────────────────────────

@router.post("/glossary/{book_id}/lock", status_code=status.HTTP_201_CREATED)
async def lock_glossary_term(
    request: Request,
    book_id: str,
    term_source: str = Query(...),
    term_target: str = Query(...),
    reviewer_id: str = Query(...),
):
    """Lock a Japanese → Italian term directly, without going through a review item."""
    memory_svc: MemoryService = request.app.state.memory_service
    await memory_svc.lock_term(
        book_id=book_id,
        source_ja=term_source,
        target_it=term_target,
        reviewer_id=reviewer_id,
    )
    return {"locked": True, "source": term_source, "target": term_target}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _resolve_status(action_type: str | ReviewActionType) -> ReviewStatus:
    at = action_type if isinstance(action_type, str) else action_type.value
    if at in (ReviewActionType.APPROVE.value, ReviewActionType.EDIT.value):
        return ReviewStatus.APPROVED
    if at == ReviewActionType.REJECT.value:
        return ReviewStatus.REJECTED
    # lock_term, override_critic keep the item as approved (processed)
    return ReviewStatus.APPROVED
