"""
Node: retrieve_context

Builds the minimal RetrievalPacket for the current chunk by combining:
  1. Exact glossary / locked-term lookup from MongoDB
  2. Character profile lookup for detected speakers
  3. Style rules for the detected scene type
  4. Semantic search for similar approved segments (translation memory)

The packet is deliberately small: only terms, characters, and rules that
are relevant to THIS chunk, to prevent context explosion.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.repositories.mongo.memory_repo import MemoryRepository
from app.repositories.vector.qdrant_repo import QdrantRepository
from app.schemas.contracts import RetrievalPacket, MemoryStatus
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Module-level repository singletons (injected at startup)
# ─────────────────────────────────────────────────────────────────

_memory_repo: MemoryRepository | None = None
_qdrant_repo: QdrantRepository | None = None


def init_retrieval_repos(
    memory_repo: MemoryRepository,
    qdrant_repo: QdrantRepository,
) -> None:
    """Call once at application startup."""
    global _memory_repo, _qdrant_repo
    _memory_repo = memory_repo
    _qdrant_repo = qdrant_repo


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def retrieve_context_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    book_id = state["book_id"]
    chunk_id = state["chunk_id"]
    source_text = state["source_text"]
    signals = state.get("source_signals", {})
    speakers: list[str] = signals.get("speakers", [])
    scene_type: str = signals.get("scene_type", "mixed")

    packet = RetrievalPacket()

    if _memory_repo is None or _qdrant_repo is None:
        logger.warning(
            "retrieve_context: repositories not initialised, "
            "returning empty RetrievalPacket for chunk %s",
            chunk_id,
        )
    else:
        try:
            # 1. Locked / approved glossary terms
            locked_entries = await _memory_repo.get_locked_terms(book_id)
            for e in locked_entries:
                packet.locked_terms[e.source_segment] = e.target_segment

            # 2. Suggested (approved but not locked) terms
            suggested_entries = await _memory_repo.get_approved_terms(book_id)
            for e in suggested_entries:
                if e.source_segment not in packet.locked_terms:
                    packet.suggested_terms[e.source_segment] = e.target_segment

            # 3. Character profiles for detected speakers
            for speaker in speakers[:5]:  # cap to 5 speakers to bound context
                char = await _memory_repo.get_character(book_id, speaker)
                if char and char.status == MemoryStatus.APPROVED.value:
                    packet.character_profiles.append({
                        "name_ja": char.name_ja,
                        "name_it": char.name_it,
                        "speech_style": char.speech_style,
                        "formality_level": char.formality_level,
                        "relationships": char.relationships[:3],
                    })

            # 4. Style memory
            style = await _memory_repo.get_style_memory(book_id)
            if style and style.status == MemoryStatus.APPROVED.value:
                packet.style_summary = (
                    f"Tono: {style.tone}. "
                    f"Lunghezza frasi: {style.sentence_length}. "
                    f"Registro: {style.register}."
                ).strip()
                packet.style_rules = [
                    r.description for r in style.rules
                    if _rule_applies(r, scene_type)
                ][:5]  # cap to 5 rules

            # 5. Semantically similar approved segments (translation memory)
            similar = await _qdrant_repo.search_similar_segments(
                book_id=book_id,
                query_text=source_text,
                top_k=3,
                min_score=0.82,
            )
            packet.similar_segments = [
                {"source": s["source"], "target": s["target"], "score": s["score"]}
                for s in similar
            ]

        except Exception as exc:
            logger.error("retrieve_context failed for chunk %s: %s", chunk_id, exc)

    elapsed = time.perf_counter() - t0
    return {
        "retrieval_packet": packet.model_dump(),
        "node_timings": {"retrieve_context": round(elapsed, 3)},
        "audit_trail": [{
            "node": "retrieve_context",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "locked_terms_count": len(packet.locked_terms),
            "characters_count": len(packet.character_profiles),
            "similar_segments_count": len(packet.similar_segments),
        }],
    }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _rule_applies(rule, scene_type: str) -> bool:
    """Return True if the rule is relevant to the current scene type."""
    if not rule.applies_to:
        return True
    return scene_type in [tag if isinstance(tag, str) else tag.value for tag in rule.applies_to]
