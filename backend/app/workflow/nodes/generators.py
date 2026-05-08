"""
Node: run_generators

Runs three specialised translators in parallel via asyncio.gather.
Each generator receives the same source text and RetrievalPacket but uses a
different system prompt that emphasises a different translation priority.

Generators:
  faithful    — semantic fidelity, preserves structure and all ambiguities
  creative    — natural Italian prose, musicality, fluent rhythm
  contextual  — subtext, character relationships, narrative continuity

Outputs a list of three TranslationCandidate objects.
All three are kept in state so critics can evaluate each independently.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.contracts import TranslationCandidate
from app.workflow.state import ChunkWorkflowState

logger = logging.getLogger(__name__)
settings = get_settings()

GeneratorId = Literal["faithful", "creative", "contextual"]

# ─────────────────────────────────────────────────────────────────
# System prompts (Japanese → Italian)
# ─────────────────────────────────────────────────────────────────

_SYS_FAITHFUL = """\
Sei un traduttore letterario professionista specializzato in giapponese → italiano.
Il tuo obiettivo è la fedeltà semantica totale.

Principi:
- Preserva ogni sfumatura di significato del testo originale giapponese
- Mantieni la struttura sintattica quanto compatibile con l'italiano
- Converti le onomatopee giapponesi in equivalenti italiani appropriati
- Rispetta il livello di cortesia (keigo) nella resa italiana
- Non omettere nulla e non aggiungere nulla
- In caso di ambiguità, scegli la resa più letteralmente fedele
- Rispetta ESATTAMENTE i termini bloccati forniti nel contesto"""

_SYS_CREATIVE = """\
Sei un traduttore letterario creativo specializzato in giapponese → italiano.
Il tuo obiettivo è la qualità letteraria e la naturalezza in italiano.

Principi:
- Dai priorità alla musicalità e al ritmo della prosa italiana
- Adatta costruzioni sintattiche giapponesi a strutture italiane naturali e vivide
- Trasforma onomatopee in espressioni intense e idiomatiche in italiano
- Preserva l'intensità emotiva attraverso scelte lessicali acute
- Il testo deve suonare come scritto da un autore italiano, non come una traduzione
- Rispetta ESATTAMENTE i termini bloccati forniti nel contesto"""

_SYS_CONTEXTUAL = """\
Sei un traduttore letterario contestuale specializzato in giapponese → italiano.
Il tuo obiettivo è la continuità narrativa e il sottotesto.

Principi:
- Dai priorità alle dinamiche tra i personaggi e al loro sottotesto implicito
- Rispetta rigorosamente i profili dei personaggi forniti (tono, formalità)
- Mantieni coerenza con i segmenti precedenti e successivi (contesto fornito)
- Adatta i riferimenti culturali giapponesi per il lettore italiano senza perdere il senso
- Preserva le ambiguità intenzionali dell'autore
- Rispetta ESATTAMENTE i termini bloccati forniti nel contesto"""

_SYSTEMS: dict[GeneratorId, str] = {
    "faithful": _SYS_FAITHFUL,
    "creative": _SYS_CREATIVE,
    "contextual": _SYS_CONTEXTUAL,
}

# ─────────────────────────────────────────────────────────────────
# User prompt template (shared across all generators)
# ─────────────────────────────────────────────────────────────────

_USER_TEMPLATE = """\
TESTO SORGENTE (Giapponese):
{source_text}

――――――――――――――――――――――
CONTESTO DELLA SCENA:
- Tipo di scena: {scene_type}
- Livello di formalità: {formality_level}
- Speaker rilevati: {speakers}
- Segnali stilistici: {speech_style_hints}

{preceding_block}\
{following_block}\
{locked_terms_block}\
{characters_block}\
{style_block}\
――――――――――――――――――――――
Fornisci SOLO la traduzione italiana. Nessuna spiegazione, nessuna nota."""


def _build_user_prompt(
    source_text: str,
    signals: dict,
    retrieval: dict,
    chunk_context: dict,
) -> str:
    scene_type = signals.get("scene_type", "mixed")
    formality = signals.get("formality_level", "futsuutai")
    speakers = ", ".join(signals.get("speakers", [])) or "non specificati"
    style_hints = "; ".join(signals.get("speech_style_hints", [])) or "—"

    preceding = chunk_context.get("preceding_summary") or ""
    following = chunk_context.get("following_teaser") or ""

    preceding_block = (
        f"CONTESTO PRECEDENTE (riassunto):\n{preceding}\n\n" if preceding else ""
    )
    following_block = (
        f"CONTESTO SUCCESSIVO (anteprima):\n{following}\n\n" if following else ""
    )

    locked = retrieval.get("locked_terms") or {}
    locked_lines = [f"  {ja} → {it}" for ja, it in locked.items()]
    locked_terms_block = (
        "TERMINI BLOCCATI (usa esattamente queste traduzioni):\n"
        + "\n".join(locked_lines)
        + "\n\n"
        if locked_lines
        else ""
    )

    suggested = retrieval.get("suggested_terms") or {}
    suggested_lines = [f"  {ja} → {it}" for ja, it in list(suggested.items())[:10]]
    if suggested_lines:
        locked_terms_block += (
            "TERMINI SUGGERITI:\n" + "\n".join(suggested_lines) + "\n\n"
        )

    profiles = retrieval.get("character_profiles") or []
    char_lines = []
    for p in profiles:
        line = (
            f"  {p.get('name_ja')} → {p.get('name_it')}"
            f" | stile: {', '.join(p.get('speech_style', []))}"
            f" | formalità: {p.get('formality_level', '')}"
        )
        char_lines.append(line)
    characters_block = (
        "PROFILI PERSONAGGI:\n" + "\n".join(char_lines) + "\n\n" if char_lines else ""
    )

    style_summary = retrieval.get("style_summary") or ""
    style_rules = retrieval.get("style_rules") or []
    style_lines = []
    if style_summary:
        style_lines.append(style_summary)
    style_lines.extend(f"  - {r}" for r in style_rules)
    style_block = (
        "LINEE GUIDA STILISTICHE:\n" + "\n".join(style_lines) + "\n\n"
        if style_lines
        else ""
    )

    return _USER_TEMPLATE.format(
        source_text=source_text,
        scene_type=scene_type,
        formality_level=formality,
        speakers=speakers,
        speech_style_hints=style_hints,
        preceding_block=preceding_block,
        following_block=following_block,
        locked_terms_block=locked_terms_block,
        characters_block=characters_block,
        style_block=style_block,
    )


# ─────────────────────────────────────────────────────────────────
# Single generator coroutine
# ─────────────────────────────────────────────────────────────────

async def _translate(
    generator_id: GeneratorId,
    source_text: str,
    signals: dict,
    retrieval: dict,
    chunk_context: dict,
) -> tuple[TranslationCandidate, int]:
    """Call one LLM generator and return (candidate, token_count)."""
    model_map: dict[GeneratorId, str] = {
        "faithful": settings.google_model_faithful,
        "creative": settings.google_model_creative,
        "contextual": settings.google_model_contextual,
    }
    llm = ChatGoogleGenerativeAI(
        model=model_map[generator_id],
        temperature=0.3 if generator_id == "faithful" else 0.7,
        google_api_key=settings.google_api_key,
    )

    messages = [
        SystemMessage(content=_SYSTEMS[generator_id]),
        HumanMessage(
            content=_build_user_prompt(source_text, signals, retrieval, chunk_context)
        ),
    ]
    response = await llm.ainvoke(messages)
    translation = response.content.strip()

    usage = getattr(response, "usage_metadata", None)
    tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

    candidate = TranslationCandidate(
        generator_id=generator_id,
        text=translation,
        token_count=tokens,
    )
    return candidate, tokens


# ─────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────

async def run_generators_node(state: ChunkWorkflowState) -> dict:
    t0 = time.perf_counter()
    chunk_id = state["chunk_id"]
    source_text = state["source_text"]
    signals = state.get("source_signals", {})
    retrieval = state.get("retrieval_packet", {})
    chunk_context = state.get("chunk_context", {})

    results = await asyncio.gather(
        _translate("faithful", source_text, signals, retrieval, chunk_context),
        _translate("creative", source_text, signals, retrieval, chunk_context),
        _translate("contextual", source_text, signals, retrieval, chunk_context),
        return_exceptions=True,
    )

    candidates: list[dict] = []
    total_tokens = 0
    errors: list[str] = []

    for gen_id, result in zip(("faithful", "creative", "contextual"), results):
        if isinstance(result, Exception):
            logger.error("Generator %s failed for chunk %s: %s", gen_id, chunk_id, result)
            errors.append(f"{gen_id}: {result}")
        else:
            candidate, tokens = result
            candidates.append(candidate.model_dump())
            total_tokens += tokens

    elapsed = time.perf_counter() - t0
    return {
        "candidates": candidates,
        "total_tokens": total_tokens,
        "node_timings": {"run_generators": round(elapsed, 3)},
        "audit_trail": [{
            "node": "run_generators",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 3),
            "candidates_generated": len(candidates),
            "errors": errors,
        }],
    }
