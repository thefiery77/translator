"""
MongoDB repository — Books and semantic chunks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.contracts import Book, BookStatus, SemanticChunk

logger = logging.getLogger(__name__)


class BookRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._books = db["books"]
        self._chunks = db["semantic_chunks"]

    # ── Books ──────────────────────────────────────────────────────

    async def create_book(self, book: Book) -> Book:
        logger.debug(f"Creating book: {book.id} (title='{book.title}')")
        await self._books.insert_one(book.model_dump())
        logger.info(f"✓ Book created: {book.id}")
        return book

    async def get_book(self, book_id: str) -> Optional[Book]:
        logger.debug(f"Fetching book: {book_id}")
        doc = await self._books.find_one({"id": book_id})
        if doc:
            logger.debug(f"✓ Book found: {book_id}")
            return Book(**doc)
        logger.warning(f"Book not found: {book_id}")
        return None

    async def list_books(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Book]:
        logger.debug(f"Listing books: status={status}, limit={limit}, offset={offset}")
        query: dict = {}
        if status:
            query["status"] = status
        cursor = self._books.find(query, sort=[("created_at", -1)]).skip(offset).limit(limit)
        books = [Book(**doc) async for doc in cursor]
        logger.info(f"✓ Listed {len(books)} books")
        return books

    async def update_book_status(
        self, book_id: str, status: BookStatus, **extra_fields
    ) -> None:
        logger.debug(f"Updating book status: {book_id} → {status.value}")
        update: dict = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc),
        }
        update.update(extra_fields)
        await self._books.update_one({"id": book_id}, {"$set": update})
        logger.info(f"✓ Book status updated: {book_id} → {status.value}")

    async def increment_completed_chunks(self, book_id: str) -> None:
        logger.debug(f"Incrementing completed chunks: {book_id}")
        await self._books.update_one(
            {"id": book_id},
            {
                "$inc": {"completed_chunks": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        logger.debug(f"✓ Completed chunks incremented: {book_id}")

    # ── Semantic Chunks ────────────────────────────────────────────

    async def create_chunks(self, chunks: list[SemanticChunk]) -> None:
        if not chunks:
            logger.debug("No chunks to create (empty list)")
            return
        logger.debug(f"Creating {len(chunks)} semantic chunks")
        await self._chunks.insert_many([c.model_dump() for c in chunks])
        logger.info(f"✓ Created {len(chunks)} semantic chunks")

    async def get_chunk(self, chunk_id: str) -> Optional[SemanticChunk]:
        logger.debug(f"Fetching chunk: {chunk_id}")
        doc = await self._chunks.find_one({"id": chunk_id})
        return SemanticChunk(**doc) if doc else None

    async def list_chunks(self, book_id: str) -> list[SemanticChunk]:
        logger.debug(f"Listing chunks for book: {book_id}")
        cursor = self._chunks.find(
            {"book_id": book_id}, sort=[("chunk_index", 1)]
        )
        chunks = [SemanticChunk(**doc) async for doc in cursor]
        logger.debug(f"✓ Listed {len(chunks)} chunks for book: {book_id}")
        return chunks

    async def update_chunk_status(self, chunk_id: str, status: str) -> None:
        logger.debug(f"Updating chunk status: {chunk_id} → {status}")
        await self._chunks.update_one(
            {"id": chunk_id}, {"$set": {"status": status}}
        )
        logger.debug(f"✓ Chunk status updated: {chunk_id} → {status}")
