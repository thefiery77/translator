"""
Qdrant repository — vector storage and semantic retrieval for translation memory.

Collections per book:
  translator_tm_{book_id}  — translation memory segments
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_VECTOR_SIZE = 768   # Google text-embedding-001
_TM_COLLECTION_PREFIX = "translator_tm_"


class QdrantRepository:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self._embed_model: Optional[object] = None   # lazy-loaded

    # ── Collection management ─────────────────────────────────────

    async def ensure_collection(self, book_id: str) -> None:
        name = _collection_name(book_id)
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if name not in names:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", name)

    # ── Upsert ────────────────────────────────────────────────────

    async def upsert_segment(
        self,
        book_id: str,
        segment_id: str,
        source_text: str,
        target_text: str,
        embedding: list[float],
    ) -> None:
        await self.ensure_collection(book_id)
        point = PointStruct(
            id=_id_to_uint(segment_id),
            vector=embedding,
            payload={
                "segment_id": segment_id,
                "source": source_text,
                "target": target_text,
                "book_id": book_id,
            },
        )
        await self._client.upsert(
            collection_name=_collection_name(book_id),
            points=[point],
        )

    # ── Search ────────────────────────────────────────────────────

    async def search_similar_segments(
        self,
        book_id: str,
        query_text: str,
        top_k: int = 3,
        min_score: float = 0.80,
    ) -> list[dict]:
        """Return similar approved translation segments for the query."""
        try:
            embedding = await self._embed(query_text)
            results = await self._client.search(
                collection_name=_collection_name(book_id),
                query_vector=embedding,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
            )
            return [
                {
                    "source": r.payload.get("source", ""),
                    "target": r.payload.get("target", ""),
                    "score": r.score,
                }
                for r in results
                if r.payload
            ]
        except Exception as exc:
            logger.warning("Qdrant search failed for book %s: %s", book_id, exc)
            return []

    # ── Embedding helper ──────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        """Embed text using Google embedding-001."""
        import google.generativeai as genai

        genai.configure(api_key=settings.google_api_key)
        response = genai.embed_content(
            model="models/embedding-001",
            content=text[:8000],   # truncate to avoid token overflow
            task_type="SEMANTIC_SIMILARITY",
        )
        return response["embedding"]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _collection_name(book_id: str) -> str:
    # Qdrant collection names must be alphanumeric + underscore/hyphen
    safe_id = book_id.replace("-", "_")
    return f"{_TM_COLLECTION_PREFIX}{safe_id}"


def _id_to_uint(segment_id: str) -> int:
    """Convert a UUID string to a positive integer ID for Qdrant."""
    import hashlib
    h = hashlib.md5(segment_id.encode()).hexdigest()
    return int(h[:16], 16) % (2**63)
