"""
LangGraph workflow graph for the chunk-level translation pipeline.

Graph topology (MVP):

  START
    ↓
  extract_source_signals
    ↓
  retrieve_context
    ↓
  run_generators
    ↓
  run_deterministic_critics
    ↓ [route_after_deterministic]
    ├─ "escalate" ──────────────────────────────→ queue_for_review
    └─ "continue" → run_style_critic
                      ↓ [route_after_style_critic]
                      ├─ "accept" ────────────────→ queue_for_review
                      └─ "analyze" → analyze_conflicts
                                       ↓ [route_after_conflict]
                                       ├─ "accept_best_candidate" → queue_for_review
                                       ├─ "request_local_patch"   → synthesize_patch
                                       └─ "escalate_to_human"     → queue_for_review
                                                                       ↑
                                                   synthesize_patch ───┘
  queue_for_review
    ↓
  END
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.workflow.nodes.analyze_conflicts import analyze_conflicts_node
from app.workflow.nodes.generators import run_generators_node
from app.workflow.nodes.human_review import queue_for_review_node
from app.workflow.nodes.patch_synthesizer import synthesize_patch_node
from app.workflow.nodes.retrieval import retrieve_context_node
from app.workflow.nodes.source_signals import extract_source_signals_node
from app.workflow.nodes.deterministic_critics import run_deterministic_critics_node
from app.workflow.nodes.style_critic import run_style_critic_node
from app.workflow.state import ChunkWorkflowState
from app.core.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────

def route_after_deterministic(state: ChunkWorkflowState) -> str:
    """Escalate on critical failure; otherwise continue to LLM critics."""
    if state.get("has_critical_failure"):
        return "escalate"
    return "continue"


def route_after_style_critic(state: ChunkWorkflowState) -> str:
    """If confidence is high enough, accept directly; otherwise analyze conflicts."""
    confidence = state.get("style_critic_confidence", 0.0)
    if confidence >= settings.style_critic_confidence_threshold:
        return "accept"
    return "analyze"


def route_after_conflict(state: ChunkWorkflowState) -> str:
    """Map ConflictDecisionType to graph edge key."""
    decision = state.get("conflict_decision", {})
    return decision.get("decision", "escalate_to_human")


# ─────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Build and compile the chunk translation workflow.

    Args:
        checkpointer: Optional LangGraph checkpointer.  Pass a MemorySaver()
                      for development.  Replace with a persistent checkpointer
                      (e.g. MongoDBSaver) for production.
    """
    builder = StateGraph(ChunkWorkflowState)

    # Nodes
    builder.add_node("extract_source_signals", extract_source_signals_node)
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("run_generators", run_generators_node)
    builder.add_node("run_deterministic_critics", run_deterministic_critics_node)
    builder.add_node("run_style_critic", run_style_critic_node)
    builder.add_node("analyze_conflicts", analyze_conflicts_node)
    builder.add_node("synthesize_patch", synthesize_patch_node)
    builder.add_node("queue_for_review", queue_for_review_node)

    # Linear edges
    builder.add_edge(START, "extract_source_signals")
    builder.add_edge("extract_source_signals", "retrieve_context")
    builder.add_edge("retrieve_context", "run_generators")
    builder.add_edge("run_generators", "run_deterministic_critics")

    # Conditional: after deterministic critics
    builder.add_conditional_edges(
        "run_deterministic_critics",
        route_after_deterministic,
        {
            "escalate": "queue_for_review",
            "continue": "run_style_critic",
        },
    )

    # Conditional: after style critic
    builder.add_conditional_edges(
        "run_style_critic",
        route_after_style_critic,
        {
            "accept": "queue_for_review",
            "analyze": "analyze_conflicts",
        },
    )

    # Conditional: after conflict analysis
    builder.add_conditional_edges(
        "analyze_conflicts",
        route_after_conflict,
        {
            "accept_best_candidate": "queue_for_review",
            "request_local_patch": "synthesize_patch",
            "escalate_to_human": "queue_for_review",
        },
    )

    builder.add_edge("synthesize_patch", "queue_for_review")
    builder.add_edge("queue_for_review", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


# Module-level singleton for the compiled graph.
# Replaced at startup via main.py if a persistent checkpointer is desired.
translation_graph = build_graph()
