"""
Shared Pydantic v2 contracts for the Japanese → Italian translation pipeline.

All enums, domain models, and DTOs live here so every layer (workflow nodes,
repositories, API routes) shares the same type definitions without circular
imports.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────

class BookStatus(str, Enum):
    INGESTED = "ingested"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    TRANSLATING = "translating"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceFormat(str, Enum):
    EPUB = "epub"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class StyleTag(str, Enum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    INNER_MONOLOGUE = "inner_monologue"
    ACTION = "action"
    DESCRIPTION = "description"


class SceneType(str, Enum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    ACTION = "action"
    DESCRIPTION = "description"
    MIXED = "mixed"


class FormalityLevel(str, Enum):
    """Japanese speech register levels."""
    TEINEIGO = "teineigo"       # polite (-masu/-desu)
    SONKEIGO = "sonkeigo"       # respectful (honorific)
    KENJOOGO = "kenjoogo"       # humble (self-deprecating)
    FUTSUUTAI = "futsuutai"     # plain form
    TAMEGUCHI = "tameguchi"     # casual / familiar


class MemoryOrigin(str, Enum):
    HUMAN = "human"
    AI = "ai"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"   # AI-generated, not yet validated
    APPROVED = "approved"     # Human-validated; used for retrieval


class CriticSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ConflictDecisionType(str, Enum):
    ACCEPT_BEST_CANDIDATE = "accept_best_candidate"
    REQUEST_LOCAL_PATCH = "request_local_patch"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    ESCALATED = "escalated"


class ReviewActionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    LOCK_TERM = "lock_term"
    OVERRIDE_CRITIC = "override_critic"


# ─────────────────────────────────────────────────────────────────
# Book
# ─────────────────────────────────────────────────────────────────

class BookMetadata(BaseModel):
    author: str = ""
    genre: list[str] = Field(default_factory=list)
    published_year: Optional[int] = None

    model_config = ConfigDict(use_enum_values=True)


class Book(BaseModel):
    id: str = Field(default_factory=_uid)
    title: str
    source_language: Literal["ja"] = "ja"
    target_language: Literal["it"] = "it"
    source_format: SourceFormat
    status: BookStatus = BookStatus.INGESTED
    metadata: BookMetadata = Field(default_factory=BookMetadata)
    total_chunks: int = 0
    completed_chunks: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Paragraph — normalised internal format
# ─────────────────────────────────────────────────────────────────

class Paragraph(BaseModel):
    chapter: int
    paragraph_id: str           # e.g. "4_18"
    speaker: Optional[str] = None
    text: str                   # raw Japanese text
    style_tags: list[StyleTag] = Field(default_factory=list)
    char_offset_start: int = 0
    char_offset_end: int = 0

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Semantic Chunk
# ─────────────────────────────────────────────────────────────────

class ChunkContext(BaseModel):
    """Overlap context injected at generation time — never changes the source."""
    preceding_summary: Optional[str] = None   # summary of previous chunk
    following_teaser: Optional[str] = None    # first sentence(s) of next chunk


class SemanticChunk(BaseModel):
    id: str = Field(default_factory=_uid)
    book_id: str
    chapter: int
    chunk_index: int
    paragraphs: list[Paragraph]
    source_text: str            # full concatenated Japanese text
    token_count: int
    context: ChunkContext = Field(default_factory=ChunkContext)
    status: str = "pending"     # pending | translating | completed | failed
    created_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Source Signals — extracted from Japanese source ONLY
# ─────────────────────────────────────────────────────────────────

class SourceSignals(BaseModel):
    """
    Canonical signals extracted from the Japanese source text.

    CRITICAL: These are extracted only from the source, never from the
    generated translation, to prevent feedback-loop hallucinations.
    They are used to steer stylistic choices, NOT to inject explicit emotions.
    """
    chunk_id: str
    speakers: list[str] = Field(default_factory=list)
    scene_type: SceneType = SceneType.MIXED
    formality_level: FormalityLevel = FormalityLevel.FUTSUUTAI
    tension_markers: list[str] = Field(default_factory=list)
    honorific_patterns: list[str] = Field(default_factory=list)
    onomatopoeia: list[str] = Field(default_factory=list)
    speech_style_hints: list[str] = Field(default_factory=list)
    cultural_references: list[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Memory entries
# ─────────────────────────────────────────────────────────────────

class TranslationMemoryEntry(BaseModel):
    id: str = Field(default_factory=_uid)
    book_id: str
    source_segment: str         # Japanese
    target_segment: str         # Italian
    source_language: Literal["ja"] = "ja"
    target_language: Literal["it"] = "it"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    locked: bool = False        # if True, must use this exact translation
    origin: MemoryOrigin = MemoryOrigin.AI
    status: MemoryStatus = MemoryStatus.CANDIDATE
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


class CharacterMemoryEntry(BaseModel):
    id: str = Field(default_factory=_uid)
    book_id: str
    name_ja: str                # Original Japanese name / kanji
    name_it: str                # Italian rendering
    speech_style: list[str] = Field(default_factory=list)
    formality_level: FormalityLevel = FormalityLevel.FUTSUUTAI
    relationships: list[str] = Field(default_factory=list)
    nicknames: list[dict[str, str]] = Field(default_factory=list)
    notes: str = ""
    origin: MemoryOrigin = MemoryOrigin.AI
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


class StyleRule(BaseModel):
    rule_id: str = Field(default_factory=_uid)
    description: str
    examples: list[dict[str, str]] = Field(default_factory=list)
    applies_to: list[StyleTag] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)

    model_config = ConfigDict(use_enum_values=True)


class StyleMemoryEntry(BaseModel):
    id: str = Field(default_factory=_uid)
    book_id: str
    tone: str = ""              # melancholic, tense, lyrical, etc.
    sentence_length: str = ""   # short, long, varied
    register: str = ""          # literary, colloquial, technical
    rules: list[StyleRule] = Field(default_factory=list)
    origin: MemoryOrigin = MemoryOrigin.AI
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Retrieval Packet — minimal context injected per chunk
# ─────────────────────────────────────────────────────────────────

class RetrievalPacket(BaseModel):
    """Minimal context that each generator receives.

    Keep this intentionally small to prevent context explosion.
    Only terms, characters, and rules relevant to the current chunk.
    """
    locked_terms: dict[str, str] = Field(default_factory=dict)     # ja→it, must use
    suggested_terms: dict[str, str] = Field(default_factory=dict)  # ja→it, suggested
    character_profiles: list[dict[str, Any]] = Field(default_factory=list)
    style_summary: str = ""
    style_rules: list[str] = Field(default_factory=list)
    similar_segments: list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Translation Candidates
# ─────────────────────────────────────────────────────────────────

class TranslationCandidate(BaseModel):
    candidate_id: str = Field(default_factory=_uid)
    generator_id: Literal["faithful", "creative", "contextual"]
    text: str
    rationale: str = ""
    token_count: int = 0


# ─────────────────────────────────────────────────────────────────
# Critics
# ─────────────────────────────────────────────────────────────────

class CriticViolation(BaseModel):
    violation_id: str = Field(default_factory=_uid)
    description: str
    evidence: str = ""
    suggestion: Optional[str] = None
    severity: CriticSeverity = CriticSeverity.WARNING

    model_config = ConfigDict(use_enum_values=True)


class DeterministicCriticResult(BaseModel):
    critic_id: str
    passed: bool
    violations: list[CriticViolation] = Field(default_factory=list)
    severity: CriticSeverity = CriticSeverity.WARNING

    model_config = ConfigDict(use_enum_values=True)


class PatchRegionSuggestion(BaseModel):
    region_id: str = Field(default_factory=_uid)
    candidate_id: str
    original_span_text: str
    issue_description: str
    suggested_replacement: Optional[str] = None
    severity: CriticSeverity = CriticSeverity.WARNING

    model_config = ConfigDict(use_enum_values=True)


class StyleCriticResult(BaseModel):
    candidate_id: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    patch_regions: list[PatchRegionSuggestion] = Field(default_factory=list)
    overall_notes: str = ""

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Conflict Analysis
# ─────────────────────────────────────────────────────────────────

class ConflictDecision(BaseModel):
    decision: ConflictDecisionType
    best_candidate_id: str
    conflict_type: Optional[str] = None    # e.g. "style_vs_fidelity"
    reasoning: str = ""
    escalation_reason: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Patch Proposal
# ─────────────────────────────────────────────────────────────────

class PatchRegion(BaseModel):
    region_id: str = Field(default_factory=_uid)
    original_text: str
    replacement_text: str
    rationale: str
    contributing_critics: list[str] = Field(default_factory=list)


class PatchProposal(BaseModel):
    id: str = Field(default_factory=_uid)
    chunk_id: str
    base_candidate_id: str
    original_text: str
    patched_text: str
    regions: list[PatchRegion] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    passes_deterministic: bool = False
    created_at: datetime = Field(default_factory=_now)


# ─────────────────────────────────────────────────────────────────
# Human Review
# ─────────────────────────────────────────────────────────────────

class ReviewItem(BaseModel):
    id: str = Field(default_factory=_uid)
    book_id: str
    chunk_id: str
    source_text: str
    best_candidate: TranslationCandidate
    all_candidates: list[TranslationCandidate] = Field(default_factory=list)
    deterministic_violations: list[CriticViolation] = Field(default_factory=list)
    style_issues: list[dict[str, Any]] = Field(default_factory=list)
    patch_proposal: Optional[PatchProposal] = None
    conflict_decision: Optional[ConflictDecision] = None
    escalation_reason: Optional[str] = None
    status: ReviewStatus = ReviewStatus.PENDING
    priority: int = 0           # higher = review first
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


class ReviewActionRequest(BaseModel):
    """Incoming payload from a human reviewer via the API."""
    action_type: ReviewActionType
    final_text: Optional[str] = None
    notes: Optional[str] = None
    # For lock_term action
    term_source: Optional[str] = None
    term_target: Optional[str] = None
    # For override_critic action
    overridden_critic_id: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


class ReviewActionRecord(BaseModel):
    """Immutable audit record of a human review action."""
    id: str = Field(default_factory=_uid)
    review_item_id: str
    book_id: str
    chunk_id: str
    reviewer_id: str
    action_type: ReviewActionType
    final_text: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
# Job status (lightweight tracking document in MongoDB)
# ─────────────────────────────────────────────────────────────────

class TranslationJobStatus(BaseModel):
    job_id: str = Field(default_factory=_uid)
    book_id: str
    chunk_id: str
    status: str = "pending"     # pending | running | completed | failed | pending_review
    current_node: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
