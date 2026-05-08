"""
Semantic chunker — scene-aware and dialogue-aware splitting.

Splitting rules (in priority order):
  1. Never break inside an open dialogue block (「...」or 『...』)
  2. Prefer to break at scene boundaries (〇〇〇, ---, blank lines in source)
  3. Group paragraphs until approaching chunk_target_tokens
  4. Hard cap at chunk_max_tokens

Each chunk is enriched with overlap context:
  - preceding_summary: last N tokens of the PREVIOUS chunk (compressed via LLM
    in production; for MVP a simple truncation of the last paragraph)
  - following_teaser: first sentence of the NEXT chunk
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import tiktoken

from app.core.config import get_settings
from app.schemas.contracts import ChunkContext, Paragraph, SemanticChunk

logger = logging.getLogger(__name__)
settings = get_settings()

# Shared tokeniser (approximate — not exact for Japanese but sufficient for sizing)
_ENC = tiktoken.get_encoding("cl100k_base")

# Scene-break markers common in Japanese light novels / fiction
_SCENE_BREAK_RE = re.compile(r"^[〇◆◇■□★☆\-＊\*]{2,}$")

# Japanese dialogue open/close marker tracking
_OPEN_QUOTE = re.compile(r"[「『]")
_CLOSE_QUOTE = re.compile(r"[」』]")


# ─────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ─────────────────────────────────────────────────────────────────
# Scene-break detection
# ─────────────────────────────────────────────────────────────────

def _is_scene_break(para: Paragraph) -> bool:
    return bool(_SCENE_BREAK_RE.match(para.text.strip()))


# ─────────────────────────────────────────────────────────────────
# Dialogue-open state tracker
# ─────────────────────────────────────────────────────────────────

def _quote_balance(text: str) -> int:
    """Return open-quote count minus close-quote count in text."""
    return len(_OPEN_QUOTE.findall(text)) - len(_CLOSE_QUOTE.findall(text))


# ─────────────────────────────────────────────────────────────────
# Core chunking algorithm
# ─────────────────────────────────────────────────────────────────

@dataclass
class _ChunkBuffer:
    paragraphs: list[Paragraph] = field(default_factory=list)
    token_count: int = 0
    open_quote_balance: int = 0  # tracks unclosed 「 across paragraphs

    def add(self, para: Paragraph, tokens: int) -> None:
        self.paragraphs.append(para)
        self.token_count += tokens
        self.open_quote_balance += _quote_balance(para.text)

    def is_empty(self) -> bool:
        return len(self.paragraphs) == 0

    def reset(self) -> None:
        self.paragraphs = []
        self.token_count = 0
        self.open_quote_balance = 0

    def clone_paragraphs(self) -> list[Paragraph]:
        return list(self.paragraphs)


def create_semantic_chunks(
    book_id: str,
    paragraphs: list[Paragraph],
    chapter: int = 0,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[SemanticChunk]:
    """
    Split a list of Paragraph objects into SemanticChunk objects.

    Args:
        book_id: ID of the parent Book document.
        paragraphs: Normalised paragraphs (all from the same chapter).
        chapter: Chapter number for metadata.
        target_tokens: Soft upper bound (defaults to config value).
        max_tokens: Hard upper bound (defaults to config value).

    Returns:
        Ordered list of SemanticChunk objects, ready for persistence.
    """
    target = target_tokens or settings.chunk_target_tokens
    hard_max = max_tokens or settings.chunk_max_tokens

    chunks: list[SemanticChunk] = []
    buf = _ChunkBuffer()
    chunk_index = 0

    for para in paragraphs:
        para_tokens = count_tokens(para.text)

        # ── Hard-max overflow: flush before adding ────────────────────
        if not buf.is_empty() and buf.token_count + para_tokens > hard_max:
            if buf.open_quote_balance <= 0:
                # Safe to cut here
                chunks.append(_flush(buf, book_id, chapter, chunk_index))
                chunk_index += 1
            # else: stay in dialogue — we'll overflow slightly

        # ── Scene break: flush after completing current buffer ────────
        if _is_scene_break(para):
            if not buf.is_empty():
                chunks.append(_flush(buf, book_id, chapter, chunk_index))
                chunk_index += 1
            # The scene-break marker itself is not added to the chunk
            continue

        buf.add(para, para_tokens)

        # ── Soft target reached: flush only if dialogue is closed ─────
        if buf.token_count >= target and buf.open_quote_balance <= 0:
            chunks.append(_flush(buf, book_id, chapter, chunk_index))
            chunk_index += 1

    # Flush remaining
    if not buf.is_empty():
        chunks.append(_flush(buf, book_id, chapter, chunk_index))

    # Enrich with overlap context
    _add_overlap_context(chunks)

    return chunks


def _flush(
    buf: _ChunkBuffer,
    book_id: str,
    chapter: int,
    chunk_index: int,
) -> SemanticChunk:
    paras = buf.clone_paragraphs()
    source_text = "\n".join(p.text for p in paras)
    chunk = SemanticChunk(
        book_id=book_id,
        chapter=chapter,
        chunk_index=chunk_index,
        paragraphs=paras,
        source_text=source_text,
        token_count=buf.token_count,
        context=ChunkContext(),
    )
    buf.reset()
    return chunk


# ─────────────────────────────────────────────────────────────────
# Overlap context (preceding_summary / following_teaser)
# ─────────────────────────────────────────────────────────────────

_OVERLAP_SUMMARY_CHARS = 300   # roughly 1 short paragraph


def _add_overlap_context(chunks: list[SemanticChunk]) -> None:
    """
    Mutates chunks in place to add preceding_summary and following_teaser.

    MVP implementation uses simple text truncation.
    A production implementation could call an LLM to compress the preceding
    chunk into a concise summary before injecting it here.
    """
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_text = chunks[i - 1].source_text
            # Take the last paragraph of the previous chunk as summary
            prev_paras = [p.strip() for p in prev_text.splitlines() if p.strip()]
            summary = prev_paras[-1] if prev_paras else ""
            chunk.context.preceding_summary = summary[:_OVERLAP_SUMMARY_CHARS]

        if i < len(chunks) - 1:
            next_text = chunks[i + 1].source_text
            # Take the first sentence of the next chunk as teaser
            next_lines = [p.strip() for p in next_text.splitlines() if p.strip()]
            teaser = next_lines[0] if next_lines else ""
            # Limit to first sentence (ends at 。! ? etc.)
            match = re.search(r"[。！？!?]", teaser)
            if match:
                teaser = teaser[: match.end()]
            chunk.context.following_teaser = teaser[:200]
