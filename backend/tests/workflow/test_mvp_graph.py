"""
Tests for the MVP workflow graph.

Covers:
  - Nominal path: extract_signals → retrieve → generate → critics → accept
  - Routing: deterministic critical failure → escalate
  - Routing: high-confidence style critic → accept directly
  - Routing: low-confidence → conflict analysis → local patch
  - Audit trail accumulation across nodes
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _fake_llm_response(content: str, tokens: int = 50):
    msg = MagicMock()
    msg.content = content
    msg.usage_metadata = {"total_tokens": tokens}
    return msg


def _signals_json(**overrides) -> str:
    data = {
        "speakers": ["田中"],
        "scene_type": "dialogue",
        "formality_level": "teineigo",
        "tension_markers": [],
        "honorific_patterns": ["-san"],
        "onomatopoeia": [],
        "speech_style_hints": ["polite"],
        "cultural_references": [],
    }
    data.update(overrides)
    return json.dumps(data)


def _style_critic_json(candidate_ids: list[str], best: str, confidence: float) -> str:
    evaluations = [
        {
            "candidate_id": cid,
            "generator_id": "faithful",
            "score": confidence,
            "confidence": confidence,
            "issues": [],
            "patch_regions": [],
            "overall_notes": "",
        }
        for cid in candidate_ids
    ]
    return json.dumps({
        "evaluations": evaluations,
        "best_candidate_id": best,
        "overall_confidence": confidence,
    })


# ─────────────────────────────────────────────────────────────────
# Tests: deterministic critics (pure, no LLM)
# ─────────────────────────────────────────────────────────────────

def test_deterministic_critic_glossary_lock():
    from app.workflow.nodes.deterministic_critics import _check_glossary_locks
    locked = {"空の旅人": "Viaggiatori del Vuoto"}
    result = _check_glossary_locks("Ciao mondo.", locked)
    assert not result.passed
    assert any("Viaggiatori del Vuoto" in v.description for v in result.violations)


def test_deterministic_critic_glossary_lock_present():
    from app.workflow.nodes.deterministic_critics import _check_glossary_locks
    locked = {"空の旅人": "Viaggiatori del Vuoto"}
    result = _check_glossary_locks("I Viaggiatori del Vuoto camminavano.", locked)
    assert result.passed
    assert result.violations == []


def test_deterministic_critic_japanese_chars_detected():
    from app.workflow.nodes.deterministic_critics import _check_no_japanese_chars
    result = _check_no_japanese_chars("Ciao 田中。")
    assert not result.passed
    assert result.violations


def test_deterministic_critic_japanese_chars_clean():
    from app.workflow.nodes.deterministic_critics import _check_no_japanese_chars
    result = _check_no_japanese_chars("Guardò fuori dalla finestra.")
    assert result.passed


def test_deterministic_critic_structure_ok():
    from app.workflow.nodes.deterministic_critics import _check_structure_alignment
    src = "行1\n行2\n行3"
    tgt = "Riga1\nRiga2\nRiga3"
    result = _check_structure_alignment(src, tgt)
    assert result.passed


def test_deterministic_critic_structure_mismatch():
    from app.workflow.nodes.deterministic_critics import _check_structure_alignment
    src = "行1"
    tgt = "\n".join([f"Riga {i}" for i in range(20)])  # 20x ratio
    result = _check_structure_alignment(src, tgt)
    assert not result.passed


# ─────────────────────────────────────────────────────────────────
# Tests: conflict analyzer (pure, no LLM)
# ─────────────────────────────────────────────────────────────────

def test_conflict_accept_high_confidence():
    from app.workflow.nodes.analyze_conflicts import _decide
    from app.schemas.contracts import ConflictDecisionType
    decision, _, _ = _decide(
        best_candidate_id="cid1",
        confidence=0.9,
        det_results=[],
        style_results=[],
    )
    assert decision == ConflictDecisionType.ACCEPT_BEST_CANDIDATE


def test_conflict_escalate_low_confidence():
    from app.workflow.nodes.analyze_conflicts import _decide
    from app.schemas.contracts import ConflictDecisionType
    decision, _, _ = _decide(
        best_candidate_id="cid1",
        confidence=0.1,
        det_results=[],
        style_results=[],
    )
    assert decision == ConflictDecisionType.ESCALATE_TO_HUMAN


def test_conflict_local_patch_low_score():
    from app.workflow.nodes.analyze_conflicts import _decide
    from app.schemas.contracts import ConflictDecisionType
    style_results = [{
        "candidate_id": "cid1",
        "score": 0.45,
        "confidence": 0.7,
        "issues": [],
        "patch_regions": [],
    }]
    decision, _, _ = _decide(
        best_candidate_id="cid1",
        confidence=0.7,
        det_results=[],
        style_results=style_results,
    )
    assert decision == ConflictDecisionType.REQUEST_LOCAL_PATCH


def test_conflict_escalate_critical_deterministic():
    from app.workflow.nodes.analyze_conflicts import _decide
    from app.schemas.contracts import ConflictDecisionType, CriticSeverity
    det_results = [{
        "candidate_id": "cid1",
        "critic_id": "glossary_lock_checker",
        "passed": False,
        "violations": [{"severity": CriticSeverity.CRITICAL.value, "description": "mancante"}],
        "severity": CriticSeverity.CRITICAL.value,
    }]
    decision, _, _ = _decide(
        best_candidate_id="cid1",
        confidence=0.9,
        det_results=det_results,
        style_results=[],
    )
    assert decision == ConflictDecisionType.ESCALATE_TO_HUMAN


# ─────────────────────────────────────────────────────────────────
# Tests: source signals node
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_source_signals_happy_path(base_chunk_state):
    with patch("langchain_openai.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(
            return_value=_fake_llm_response(_signals_json())
        )
        from app.workflow.nodes.source_signals import extract_source_signals_node
        result = await extract_source_signals_node(base_chunk_state)

    assert "source_signals" in result
    signals = result["source_signals"]
    assert signals["scene_type"] == "dialogue"
    assert "田中" in signals["speakers"]
    assert len(result["audit_trail"]) == 1


@pytest.mark.asyncio
async def test_extract_source_signals_degraded_on_error(base_chunk_state):
    """Node must return empty signals gracefully if LLM fails."""
    with patch("langchain_openai.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        from app.workflow.nodes.source_signals import extract_source_signals_node
        result = await extract_source_signals_node(base_chunk_state)

    assert "source_signals" in result
    assert result["source_signals"]["speakers"] == []
    assert result["audit_trail"][0].get("error") is not None


# ─────────────────────────────────────────────────────────────────
# Tests: graph routing integration
# ─────────────────────────────────────────────────────────────────

def test_route_after_deterministic_escalate():
    from app.workflow.graph import route_after_deterministic
    state = {"has_critical_failure": True}
    assert route_after_deterministic(state) == "escalate"  # type: ignore[arg-type]


def test_route_after_deterministic_continue():
    from app.workflow.graph import route_after_deterministic
    state = {"has_critical_failure": False}
    assert route_after_deterministic(state) == "continue"  # type: ignore[arg-type]


def test_route_after_style_critic_accept():
    from app.workflow.graph import route_after_style_critic
    state = {"style_critic_confidence": 0.9}
    assert route_after_style_critic(state) == "accept"  # type: ignore[arg-type]


def test_route_after_style_critic_analyze():
    from app.workflow.graph import route_after_style_critic
    state = {"style_critic_confidence": 0.3}
    assert route_after_style_critic(state) == "analyze"  # type: ignore[arg-type]


def test_route_after_conflict_accept():
    from app.workflow.graph import route_after_conflict
    from app.schemas.contracts import ConflictDecisionType
    state = {"conflict_decision": {"decision": ConflictDecisionType.ACCEPT_BEST_CANDIDATE.value}}
    assert route_after_conflict(state) == "accept_best_candidate"  # type: ignore[arg-type]
