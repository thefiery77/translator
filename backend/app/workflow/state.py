"""
LangGraph state for the chunk-level translation workflow.

Each field uses a specific reducer:
  - list with Annotated[list, operator.add]  → values are appended
  - dict with Annotated[dict, _merge_dicts]  → dicts are shallow-merged
  - int with Annotated[int, operator.add]    → values are summed
  - everything else                          → last write wins (replacement)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class ChunkWorkflowState(TypedDict, total=False):
    # ── Identifiers ───────────────────────────────────
    job_id: str
    book_id: str
    chunk_id: str
    retry_count: int

    # ── Input (set at job dispatch; never mutated) ────
    source_text: str
    paragraphs: list[dict]      # serialized list[Paragraph]
    chunk_context: dict         # serialized ChunkContext

    # ── Source signals (extracted from Japanese source only) ──
    source_signals: dict        # serialized SourceSignals

    # ── Retrieval ─────────────────────────────────────
    retrieval_packet: dict      # serialized RetrievalPacket

    # ── Generation ────────────────────────────────────
    candidates: list[dict]      # serialized list[TranslationCandidate]

    # ── Deterministic critics ─────────────────────────
    deterministic_results: list[dict]   # list[DeterministicCriticResult]
    has_critical_failure: bool

    # ── Style critic ──────────────────────────────────
    style_critic_results: list[dict]    # list[StyleCriticResult]
    style_critic_best_candidate_id: str
    style_critic_confidence: float

    # ── Conflict ──────────────────────────────────────
    conflict_decision: dict     # serialized ConflictDecision

    # ── Patch ─────────────────────────────────────────
    patch_proposal: Optional[dict]      # serialized PatchProposal
    patch_attempts: int

    # ── Review ────────────────────────────────────────
    review_item_id: Optional[str]

    # ── Audit (accumulated across all nodes) ──────────
    audit_trail: Annotated[list[dict[str, Any]], operator.add]

    # ── Timings (merged across all nodes) ─────────────
    node_timings: Annotated[dict[str, float], _merge_dicts]

    # ── Token accounting (summed across all nodes) ────
    total_tokens: Annotated[int, operator.add]

    # ── Error ─────────────────────────────────────────
    error: Optional[str]

    # ── Metadata ──────────────────────────────────────
    started_at: str             # ISO-8601 UTC
