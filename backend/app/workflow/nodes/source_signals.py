"""
Node: extract_source_signals

Extracts canonical stylistic and contextual signals from the Japanese source
text using a structured LLM call.

CRITICAL INVARIANT: signals are extracted from the SOURCE (Japanese) only.
They are used to steer stylistic choices at generation time, never injected
as explicit emotions into the translation. This prevents feedback-loop
hallucinations where the LLM confuses the source intent with its own output.

  anxiety    → broken sentences in Italian
  melancholy → slow rhythm, extended syntax
  hostility  → sharp lexicon
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.contracts import (
    FormalityLevel,
    SceneType,
    SourceSignals,
)
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────

_SYSTEM = """\
Sei un esperto di letteratura e linguistica giapponese.
Analizza il testo sorgente giapponese fornito ed estrai segnali strutturati.
Restituisci SOLO JSON valido corrispondente allo schema indicato.
Non commentare, non spiegare. Solo JSON.
Descrivi ciò che osservi nel sorgente: non fare inferenze dalla traduzione."""

_USER_TEMPLATE = """\
TESTO SORGENTE (Giapponese):
{source_text}

Restituisci un oggetto JSON con esattamente queste chiavi:
{{
  "speakers": ["nomi degli speaker identificati nel testo originale giapponese"],
  "scene_type": "<una tra: dialogue, narration, action, description, mixed>",
  "formality_level": "<una tra: teineigo, sonkeigo, kenjoogo, futsuutai, tameguchi>",
  "tension_markers": ["brevi descrizioni di segnali di tensione/emozione nel sorgente"],
  "honorific_patterns": ["onorifici trovati: -san, -kun, -chan, -sama, -sensei, ecc."],
  "onomatopoeia": ["onomatopee presenti"],
  "speech_style_hints": ["indizi di stile: frasi brevi, esitazioni, registro formale, ecc."],
  "cultural_references": ["elementi culturali da adattare: luoghi, concetti, idiomi"]
}}"""


# ─────────────────────────────────────────────────────────────────
# Internal schema for LLM response validation
# ─────────────────────────────────────────────────────────────────

class _LLMSignals:
    __slots__ = (
        "speakers", "scene_type", "formality_level",
        "tension_markers", "honorific_patterns",
        "onomatopoeia", "speech_style_hints", "cultural_references",
    )

    def __init__(self, data: dict):
        self.speakers = list(data.get("speakers") or [])
        self.scene_type = str(data.get("scene_type") or "mixed")
        self.formality_level = str(data.get("formality_level") or "futsuutai")
        self.tension_markers = list(data.get("tension_markers") or [])
        self.honorific_patterns = list(data.get("honorific_patterns") or [])
        self.onomatopoeia = list(data.get("onomatopoeia") or [])
        self.speech_style_hints = list(data.get("speech_style_hints") or [])
        self.cultural_references = list(data.get("cultural_references") or [])


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def extract_source_signals_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    chunk_id = state["chunk_id"]
    source_text = state["source_text"]

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.google_model_critic,   # cheap model is fine here
            temperature=0,
            google_api_key=settings.google_api_key,
        )
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_USER_TEMPLATE.format(source_text=source_text)),
        ]
        response = await llm.ainvoke(messages)
        raw = _parse_json(response.content)
        parsed = _LLMSignals(raw)

        signals = SourceSignals(
            chunk_id=chunk_id,
            speakers=parsed.speakers,
            scene_type=_safe_scene_type(parsed.scene_type),
            formality_level=_safe_formality(parsed.formality_level),
            tension_markers=parsed.tension_markers,
            honorific_patterns=parsed.honorific_patterns,
            onomatopoeia=parsed.onomatopoeia,
            speech_style_hints=parsed.speech_style_hints,
            cultural_references=parsed.cultural_references,
        )

        usage = getattr(response, "usage_metadata", None)
        tokens = (usage.get("total_tokens", 0) if isinstance(usage, dict) else 0)

        elapsed = time.perf_counter() - t0
        return {
            "source_signals": signals.model_dump(),
            "total_tokens": tokens,
            "node_timings": {"extract_source_signals": round(elapsed, 3)},
            "audit_trail": [{
                "node": "extract_source_signals",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(elapsed, 3),
                "speakers_found": len(signals.speakers),
                "scene_type": signals.scene_type,
            }],
        }

    except Exception as exc:
        logger.error("extract_source_signals failed for chunk %s: %s", chunk_id, exc)
        # Graceful degradation: empty signals — workflow continues
        signals = SourceSignals(chunk_id=chunk_id)
        elapsed = time.perf_counter() - t0
        return {
            "source_signals": signals.model_dump(),
            "node_timings": {"extract_source_signals": round(elapsed, 3)},
            "audit_trail": [{
                "node": "extract_source_signals",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }],
        }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    return json.loads(text)


def _safe_scene_type(value: str) -> SceneType:
    try:
        return SceneType(value)
    except ValueError:
        return SceneType.MIXED


def _safe_formality(value: str) -> FormalityLevel:
    try:
        return FormalityLevel(value)
    except ValueError:
        return FormalityLevel.FUTSUUTAI
