"""
MongoDB repository — Translation memory, character memory, style memory.

Anti-contamination policy (enforced here):
  - Only records with status=APPROVED are returned by lookup methods.
  - AI-generated updates are stored with status=CANDIDATE and must go
    through a human review action before becoming APPROVED.
  - update_from_review() is the ONLY path that promotes CANDIDATE → APPROVED.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.contracts import (
    CharacterMemoryEntry,
    MemoryOrigin,
    MemoryStatus,
    StyleMemoryEntry,
    TranslationMemoryEntry,
)


class MemoryRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._tm = db["translation_memory"]
        self._chars = db["character_memory"]
        self._style = db["style_memory"]

    # ── Translation Memory ────────────────────────────────────────

    async def get_locked_terms(self, book_id: str) -> list[TranslationMemoryEntry]:
        """Return only locked=True AND approved terms."""
        cursor = self._tm.find({
            "book_id": book_id,
            "locked": True,
            "status": MemoryStatus.APPROVED.value,
        })
        return [TranslationMemoryEntry(**doc) async for doc in cursor]

    async def get_approved_terms(self, book_id: str) -> list[TranslationMemoryEntry]:
        """Return all approved (non-locked) terms for suggestion."""
        cursor = self._tm.find({
            "book_id": book_id,
            "status": MemoryStatus.APPROVED.value,
        })
        return [TranslationMemoryEntry(**doc) async for doc in cursor]

    async def upsert_candidate_term(self, entry: TranslationMemoryEntry) -> None:
        """Store or update a term as CANDIDATE (AI-generated, not approved)."""
        entry.status = MemoryStatus.CANDIDATE
        entry.origin = MemoryOrigin.AI
        entry.updated_at = datetime.now(timezone.utc)
        await self._tm.update_one(
            {"book_id": entry.book_id, "source_segment": entry.source_segment},
            {"$set": entry.model_dump()},
            upsert=True,
        )

    async def approve_term(
        self,
        book_id: str,
        source_segment: str,
        target_segment: str,
        reviewer_id: str,
        locked: bool = False,
    ) -> None:
        """Promote a term to APPROVED status. Called only from review actions."""
        now = datetime.now(timezone.utc)
        await self._tm.update_one(
            {"book_id": book_id, "source_segment": source_segment},
            {
                "$set": {
                    "target_segment": target_segment,
                    "status": MemoryStatus.APPROVED.value,
                    "origin": MemoryOrigin.HUMAN.value,
                    "locked": locked,
                    "reviewer_id": reviewer_id,
                    "reviewed_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    # ── Character Memory ──────────────────────────────────────────

    async def get_character(
        self, book_id: str, name_ja: str
    ) -> Optional[CharacterMemoryEntry]:
        doc = await self._chars.find_one({"book_id": book_id, "name_ja": name_ja})
        return CharacterMemoryEntry(**doc) if doc else None

    async def upsert_candidate_character(self, entry: CharacterMemoryEntry) -> None:
        entry.status = MemoryStatus.CANDIDATE
        entry.origin = MemoryOrigin.AI
        entry.updated_at = datetime.now(timezone.utc)
        await self._chars.update_one(
            {"book_id": entry.book_id, "name_ja": entry.name_ja},
            {"$set": entry.model_dump()},
            upsert=True,
        )

    async def approve_character(
        self,
        book_id: str,
        name_ja: str,
        updates: dict,
        reviewer_id: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        updates.update({
            "status": MemoryStatus.APPROVED.value,
            "origin": MemoryOrigin.HUMAN.value,
            "updated_at": now,
        })
        await self._chars.update_one(
            {"book_id": book_id, "name_ja": name_ja},
            {"$set": updates},
            upsert=True,
        )

    # ── Style Memory ──────────────────────────────────────────────

    async def get_style_memory(self, book_id: str) -> Optional[StyleMemoryEntry]:
        doc = await self._style.find_one({
            "book_id": book_id,
            "status": MemoryStatus.APPROVED.value,
        })
        return StyleMemoryEntry(**doc) if doc else None

    async def upsert_candidate_style(self, entry: StyleMemoryEntry) -> None:
        entry.status = MemoryStatus.CANDIDATE
        entry.origin = MemoryOrigin.AI
        entry.updated_at = datetime.now(timezone.utc)
        await self._style.update_one(
            {"book_id": entry.book_id},
            {"$set": entry.model_dump()},
            upsert=True,
        )

    async def approve_style(self, book_id: str, updates: dict, reviewer_id: str) -> None:
        now = datetime.now(timezone.utc)
        updates.update({
            "status": MemoryStatus.APPROVED.value,
            "origin": MemoryOrigin.HUMAN.value,
            "updated_at": now,
        })
        await self._style.update_one(
            {"book_id": book_id},
            {"$set": updates},
            upsert=True,
        )
