"""
Tests for the memory service anti-contamination policy.

Key invariants tested:
  1. AI-generated translations are stored as CANDIDATE, not APPROVED.
  2. Only human review actions promote records to APPROVED.
  3. Lock-term action sets locked=True.
  4. REJECT action does NOT update memory.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.schemas.contracts import (
    MemoryOrigin,
    MemoryStatus,
    ReviewActionRecord,
    ReviewActionType,
    ReviewItem,
    ReviewStatus,
    TranslationCandidate,
    TranslationMemoryEntry,
)
from app.services.memory_service import MemoryService


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def memory_repo():
    repo = MagicMock()
    repo.upsert_candidate_term = AsyncMock()
    repo.approve_term = AsyncMock()
    repo.get_locked_terms = AsyncMock(return_value=[])
    repo.get_approved_terms = AsyncMock(return_value=[])
    repo.get_character = AsyncMock(return_value=None)
    repo.get_style_memory = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def qdrant_repo():
    repo = MagicMock()
    repo.upsert_segment = AsyncMock()
    repo._embed = AsyncMock(return_value=[0.1] * 1536)
    return repo


@pytest.fixture
def memory_service(memory_repo, qdrant_repo):
    return MemoryService(memory_repo, qdrant_repo)


def _make_review_item() -> ReviewItem:
    return ReviewItem(
        book_id="book1",
        chunk_id="chunk1",
        source_text="彼女は窓の外を見た。",
        best_candidate=TranslationCandidate(
            generator_id="faithful",
            text="Lei guardò fuori dalla finestra.",
        ),
        status=ReviewStatus.PENDING,
    )


def _make_action(action_type: ReviewActionType, final_text: str | None = None) -> ReviewActionRecord:
    return ReviewActionRecord(
        review_item_id="review1",
        book_id="book1",
        chunk_id="chunk1",
        reviewer_id="reviewer1",
        action_type=action_type,
        final_text=final_text,
    )


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_translation_candidate_is_candidate(memory_service, memory_repo):
    """AI translations must be stored as CANDIDATE, never APPROVED."""
    await memory_service.store_translation_candidate(
        book_id="book1",
        source_text="テスト",
        best_translation="Prova",
    )
    memory_repo.upsert_candidate_term.assert_called_once()
    stored: TranslationMemoryEntry = memory_repo.upsert_candidate_term.call_args[0][0]
    assert stored.status == MemoryStatus.CANDIDATE
    assert stored.origin == MemoryOrigin.AI
    assert not stored.locked


@pytest.mark.asyncio
async def test_approve_action_promotes_to_approved(memory_service, memory_repo, qdrant_repo):
    """APPROVE action must call approve_term (APPROVED path)."""
    item = _make_review_item()
    action = _make_action(ReviewActionType.APPROVE)

    await memory_service.update_from_review_action(action, item)

    memory_repo.approve_term.assert_called_once()
    call_kwargs = memory_repo.approve_term.call_args.kwargs
    assert call_kwargs["book_id"] == "book1"
    assert call_kwargs["locked"] is False


@pytest.mark.asyncio
async def test_edit_action_uses_final_text(memory_service, memory_repo):
    """EDIT action must store the human-provided final_text."""
    item = _make_review_item()
    action = _make_action(ReviewActionType.EDIT, final_text="Lei osservò la pioggia fuori.")

    await memory_service.update_from_review_action(action, item)

    memory_repo.approve_term.assert_called_once()
    call_kwargs = memory_repo.approve_term.call_args.kwargs
    assert "osservò la pioggia" in call_kwargs["target_segment"]


@pytest.mark.asyncio
async def test_reject_action_does_not_touch_memory(memory_service, memory_repo):
    """REJECT action must NOT update any memory."""
    item = _make_review_item()
    action = _make_action(ReviewActionType.REJECT)

    await memory_service.update_from_review_action(action, item)

    memory_repo.approve_term.assert_not_called()
    memory_repo.upsert_candidate_term.assert_not_called()


@pytest.mark.asyncio
async def test_lock_term_sets_locked_flag(memory_service, memory_repo):
    """lock_term must call approve_term with locked=True."""
    await memory_service.lock_term(
        book_id="book1",
        source_ja="空の旅人",
        target_it="Viaggiatori del Vuoto",
        reviewer_id="reviewer1",
    )
    memory_repo.approve_term.assert_called_once()
    call_kwargs = memory_repo.approve_term.call_args.kwargs
    assert call_kwargs["locked"] is True
    assert call_kwargs["source_segment"] == "空の旅人"
    assert call_kwargs["target_segment"] == "Viaggiatori del Vuoto"


@pytest.mark.asyncio
async def test_locked_terms_not_returned_when_no_approved(memory_service, memory_repo):
    """get_locked_terms returns empty list when nothing is approved."""
    memory_repo.get_locked_terms = AsyncMock(return_value=[])
    result = await memory_service.get_locked_terms("book1")
    assert result == []
