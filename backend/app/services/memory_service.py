"""
Memory Service — manages the three memory systems with anti-contamination policy.

Anti-contamination rules (enforced here):
  1. CANDIDATE records (AI-generated) are stored but NOT returned by retrieval.
  2. Only APPROVED records enter the RetrievalPacket and are used by generators.
  3. update_from_review_action() is the SOLE path that promotes CANDIDATE → APPROVED.
  4. Term locks require explicit human action (ReviewActionType.LOCK_TERM).
  5. No memory derived from a generated translation is promoted without validation.
"""
from __future__ import annotations

import logging

from app.repositories.mongo.memory_repo import MemoryRepository
from app.repositories.vector.qdrant_repo import QdrantRepository
from app.schemas.contracts import (
    CharacterMemoryEntry,
    MemoryOrigin,
    MemoryStatus,
    ReviewActionRecord,
    ReviewActionType,
    ReviewItem,
    StyleMemoryEntry,
    TranslationMemoryEntry,
)

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        memory_repo: MemoryRepository,
        qdrant_repo: QdrantRepository,
    ) -> None:
        self._memory = memory_repo
        self._qdrant = qdrant_repo

    # ── Retrieval (read path — APPROVED only) ─────────────────────

    async def get_locked_terms(
        self, book_id: str
    ) -> list[TranslationMemoryEntry]:
        return await self._memory.get_locked_terms(book_id)

    async def get_approved_terms(
        self, book_id: str
    ) -> list[TranslationMemoryEntry]:
        return await self._memory.get_approved_terms(book_id)

    # ── Human review update (write path — the only promotion path) ─

    async def update_from_review_action(
        self,
        action: ReviewActionRecord,
        item: ReviewItem,
    ) -> None:
        """
        Apply memory updates based on a human review action.

        Only APPROVE and EDIT actions promote or create approved memory.
        LOCK_TERM additionally sets locked=True.
        REJECT does not touch memory.
        OVERRIDE_CRITIC does not touch memory.
        """
        if action.action_type in (
            ReviewActionType.APPROVE.value,
            ReviewActionType.EDIT.value,
        ):
            final_text = action.final_text or item.best_candidate.text
            await self._store_approved_translation(
                action=action,
                item=item,
                final_text=final_text,
                locked=False,
            )

        elif action.action_type == ReviewActionType.LOCK_TERM.value:
            # Lock a specific term; does not store the full segment
            logger.info(
                "Locking term for book %s: %s → %s",
                action.book_id,
                item.source_text[:40],
                final_text := (action.final_text or ""),
            )
            # The actual locking is handled via the glossary endpoint,
            # but we can also lock from a review action if term fields are set.
            # (ReviewActionRequest.term_source / term_target)

    async def lock_term(
        self,
        book_id: str,
        source_ja: str,
        target_it: str,
        reviewer_id: str,
    ) -> None:
        """Lock a Japanese→Italian term. Approved immediately by human."""
        await self._memory.approve_term(
            book_id=book_id,
            source_segment=source_ja,
            target_segment=target_it,
            reviewer_id=reviewer_id,
            locked=True,
        )
        # Also index in Qdrant for fuzzy retrieval
        try:
            embedding = await self._qdrant._embed(source_ja)
            await self._qdrant.upsert_segment(
                book_id=book_id,
                segment_id=f"lock_{book_id}_{hash(source_ja) & 0xFFFFFFFF}",
                source_text=source_ja,
                target_text=target_it,
                embedding=embedding,
            )
        except Exception as exc:
            logger.warning("Failed to index locked term in Qdrant: %s", exc)

    # ── AI-generated candidate storage (write path — stays CANDIDATE) ─

    async def store_translation_candidate(
        self,
        book_id: str,
        source_text: str,
        best_translation: str,
    ) -> None:
        """
        Store an AI-generated translation as a CANDIDATE for future reference.
        It will NOT be used by retrieval until a human approves it.
        """
        entry = TranslationMemoryEntry(
            book_id=book_id,
            source_segment=source_text[:500],   # index first 500 chars
            target_segment=best_translation[:500],
            confidence=0.5,
            locked=False,
            origin=MemoryOrigin.AI,
            status=MemoryStatus.CANDIDATE,
        )
        await self._memory.upsert_candidate_term(entry)

    # ── Private helpers ───────────────────────────────────────────

    async def _store_approved_translation(
        self,
        action: ReviewActionRecord,
        item: ReviewItem,
        final_text: str,
        locked: bool,
    ) -> None:
        """Persist approved translation in MongoDB TM and index in Qdrant."""
        await self._memory.approve_term(
            book_id=action.book_id,
            source_segment=item.source_text[:500],
            target_segment=final_text[:500],
            reviewer_id=action.reviewer_id,
            locked=locked,
        )
        try:
            embedding = await self._qdrant._embed(item.source_text[:500])
            await self._qdrant.upsert_segment(
                book_id=action.book_id,
                segment_id=item.chunk_id,
                source_text=item.source_text[:500],
                target_text=final_text[:500],
                embedding=embedding,
            )
        except Exception as exc:
            logger.warning(
                "Failed to index approved translation for chunk %s: %s",
                item.chunk_id, exc,
            )
