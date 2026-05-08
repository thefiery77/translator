from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────
    google_api_key: str = ""
    google_model_faithful: str = "gemini-2.0-flash"
    google_model_creative: str = "gemini-2.0-flash"
    google_model_contextual: str = "gemini-2.0-flash"
    google_model_critic: str = "gemini-1.5-flash"
    google_model_patch: str = "gemini-2.0-flash"

    # ── Language pair ─────────────────────────────────
    source_language: str = "ja"
    target_language: str = "it"

    # ── Databases ─────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "translator"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    redis_url: str = "redis://localhost:6379/0"

    # ── Chunking ──────────────────────────────────────
    chunk_target_tokens: int = 2048
    chunk_max_tokens: int = 4096
    chunk_overlap_tokens: int = 128

    # ── Critics ───────────────────────────────────────
    style_critic_confidence_threshold: float = 0.75
    deterministic_critic_max_violations: int = 3

    # ── Workflow ──────────────────────────────────────
    max_patch_attempts: int = 2

    # ── Observability ─────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
