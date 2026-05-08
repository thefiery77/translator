"""
Tests for the semantic chunker.

Covers:
  - Basic splitting on token target
  - Dialogue-block preservation (no mid-dialogue cuts)
  - Scene-break marker removal
  - Overlap context (preceding_summary / following_teaser)
"""
from __future__ import annotations

import pytest

from app.ingestion.chunker import (
    _quote_balance,
    _is_scene_break,
    create_semantic_chunks,
)
from app.schemas.contracts import Paragraph, StyleTag


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _para(text: str, idx: int = 0, chapter: int = 0) -> Paragraph:
    return Paragraph(
        chapter=chapter,
        paragraph_id=f"{chapter}_{idx}",
        text=text,
        style_tags=[StyleTag.NARRATION],
    )


# ─────────────────────────────────────────────────────────────────
# Unit: quote balance
# ─────────────────────────────────────────────────────────────────

def test_quote_balance_closed():
    assert _quote_balance("「こんにちは」") == 0


def test_quote_balance_open():
    assert _quote_balance("「こんにちは") == 1


def test_quote_balance_multi():
    assert _quote_balance("「A」「B") == 1


# ─────────────────────────────────────────────────────────────────
# Unit: scene break detection
# ─────────────────────────────────────────────────────────────────

def test_scene_break_detected():
    assert _is_scene_break(_para("〇〇〇"))
    assert _is_scene_break(_para("---"))
    assert _is_scene_break(_para("◆◆◆"))


def test_scene_break_not_detected():
    assert not _is_scene_break(_para("普通のテキスト"))
    assert not _is_scene_break(_para("「こんにちは」"))


# ─────────────────────────────────────────────────────────────────
# Integration: basic chunking
# ─────────────────────────────────────────────────────────────────

def test_single_chunk_when_small():
    paras = [_para(f"短い文章 {i}。", i) for i in range(5)]
    chunks = create_semantic_chunks("book1", paras, target_tokens=2048)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunks_split_at_target(monkeypatch):
    """Simulate splitting by patching count_tokens to return 500 per paragraph."""
    import app.ingestion.chunker as chunker_mod
    monkeypatch.setattr(chunker_mod, "count_tokens", lambda _: 500)

    paras = [_para(f"段落 {i}。", i) for i in range(10)]
    # target=2048 → flush after 4 paragraphs (4×500=2000 < 2048, 5×500=2500 > 2048)
    chunks = create_semantic_chunks("book1", paras, target_tokens=2048, max_tokens=4096)
    assert len(chunks) >= 2


def test_scene_break_creates_new_chunk():
    paras = [
        _para("最初のシーン。", 0),
        _para("〇〇〇", 1),   # scene break
        _para("次のシーン。", 2),
    ]
    chunks = create_semantic_chunks("book1", paras, target_tokens=4096)
    assert len(chunks) == 2
    # Scene break marker itself should NOT appear in any chunk
    for chunk in chunks:
        for p in chunk.paragraphs:
            assert "〇〇〇" not in p.text


# ─────────────────────────────────────────────────────────────────
# Integration: dialogue preservation
# ─────────────────────────────────────────────────────────────────

def test_dialogue_not_split_mid_block(monkeypatch):
    """
    When a paragraph opens a dialogue (「) and the next closes it (」),
    they must end up in the same chunk.
    """
    import app.ingestion.chunker as chunker_mod
    # Make each paragraph expensive in tokens
    monkeypatch.setattr(chunker_mod, "count_tokens", lambda _: 1200)

    paras = [
        _para("「セリフが始まる", 0),   # open quote → balance = 1
        _para("セリフが続く」", 1),      # close quote → balance = 0
        _para("地の文", 2),
    ]
    chunks = create_semantic_chunks("book1", paras, target_tokens=1500, max_tokens=4096)
    # The first two paragraphs should be in the same chunk
    first_chunk_ids = [p.paragraph_id for p in chunks[0].paragraphs]
    assert "0_0" in first_chunk_ids
    assert "0_1" in first_chunk_ids


# ─────────────────────────────────────────────────────────────────
# Integration: overlap context
# ─────────────────────────────────────────────────────────────────

def test_overlap_context_added(monkeypatch):
    import app.ingestion.chunker as chunker_mod
    monkeypatch.setattr(chunker_mod, "count_tokens", lambda _: 600)

    paras = [_para(f"段落 {i}。テキスト。", i) for i in range(6)]
    chunks = create_semantic_chunks("book1", paras, target_tokens=1500, max_tokens=4096)

    assert len(chunks) >= 2
    # Second chunk should have a preceding_summary
    assert chunks[1].context.preceding_summary is not None
    # First chunk should have a following_teaser
    assert chunks[0].context.following_teaser is not None
    # Last chunk should NOT have a following_teaser
    assert chunks[-1].context.following_teaser is None
