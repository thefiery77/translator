"""
MongoDB repository — ReviewItem and ReviewActionRecord.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.contracts import ReviewActionRecord, ReviewItem, ReviewStatus


class ReviewRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._items = db["review_items"]
        self._actions = db["review_actions"]

    # ── ReviewItem ────────────────────────────────────────────────

    async def create(self, item: ReviewItem) -> ReviewItem:
        await self._items.insert_one(item.model_dump())
        return item

    async def get(self, review_id: str) -> Optional[ReviewItem]:
        doc = await self._items.find_one({"id": review_id})
        return ReviewItem(**doc) if doc else None

    async def list_pending(
        self,
        book_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReviewItem]:
        query: dict = {"status": {"$in": [
            ReviewStatus.PENDING.value,
            ReviewStatus.ESCALATED.value,
        ]}}
        if book_id:
            query["book_id"] = book_id
        cursor = (
            self._items.find(query)
            .sort([("priority", -1), ("created_at", 1)])
            .skip(offset)
            .limit(limit)
        )
        return [ReviewItem(**doc) async for doc in cursor]

    async def update_status(
        self,
        review_id: str,
        status: ReviewStatus,
    ) -> None:
        await self._items.update_one(
            {"id": review_id},
            {
                "$set": {
                    "status": status.value,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    # ── ReviewActionRecord ────────────────────────────────────────

    async def record_action(self, action: ReviewActionRecord) -> ReviewActionRecord:
        await self._actions.insert_one(action.model_dump())
        return action

    async def list_actions_for_item(
        self, review_item_id: str
    ) -> list[ReviewActionRecord]:
        cursor = self._actions.find({"review_item_id": review_item_id})
        return [ReviewActionRecord(**doc) async for doc in cursor]
