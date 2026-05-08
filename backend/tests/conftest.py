"""
Pytest configuration and shared fixtures.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Shared LLM mock ───────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    """A fake ChatOpenAI response with usage metadata."""
    def _make(content: str, tokens: int = 100):
        msg = MagicMock()
        msg.content = content
        msg.usage_metadata = {"total_tokens": tokens}
        return msg
    return _make


@pytest.fixture
def mock_chat_openai(monkeypatch, mock_llm_response):
    """Patch ChatOpenAI.ainvoke across all nodes."""
    mock = AsyncMock(return_value=mock_llm_response("Risposta simulata", tokens=80))
    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI.ainvoke", mock
    )
    return mock


# ── Minimal workflow state fixture ───────────────────────────────

@pytest.fixture
def base_chunk_state():
    from datetime import datetime, timezone
    return {
        "job_id": "job_test_001",
        "book_id": "book_test_001",
        "chunk_id": "chunk_test_001",
        "retry_count": 0,
        "source_text": "彼女は窓の外を見た。「また雨だ」と彼女は静かに言った。",
        "paragraphs": [],
        "chunk_context": {
            "preceding_summary": "前のシーンのサマリー",
            "following_teaser": "次のシーンの冒頭",
        },
        "source_signals": {},
        "retrieval_packet": {},
        "candidates": [],
        "deterministic_results": [],
        "style_critic_results": [],
        "has_critical_failure": False,
        "style_critic_best_candidate_id": "",
        "style_critic_confidence": 0.0,
        "conflict_decision": {},
        "patch_proposal": None,
        "patch_attempts": 0,
        "review_item_id": None,
        "audit_trail": [],
        "node_timings": {},
        "total_tokens": 0,
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
