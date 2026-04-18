"""
config.py — Central settings loaded from environment variables.
All app components import from here; never read os.environ directly.
"""
from functools import lru_cache
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────────
    app_name: str = "SmartHealthAssistant"
    environment: str = "development"
    debug: bool = True
    cors_origins: List[str] = ["http://localhost:5173"]

    # ── Supabase ───────────────────────────────────────────────────────────────
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str

    # ── JWT ────────────────────────────────────────────────────────────────────
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    jwt_refresh_expiry_days: int = 7

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    openai_api_key: str

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str

    # ── Pinecone ───────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "gale-medical-encyclopedia"

    # ── RAG ────────────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 1500
    rag_top_k: int = 5
    rag_min_score: float = 0.72
    ml_fallback_threshold: float = 0.60

    # ── Storage ────────────────────────────────────────────────────────────────
    storage_bucket: str = "lab-reports"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings — only loaded once per process."""
    return Settings()


settings = get_settings()
