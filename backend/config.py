"""Central settings object. Every env var in .env.example is declared here —
agent code must read model names/thresholds from a `Settings` instance, never
hardcode them (see PROJECT_PLAN.md SS4)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    classifier_model: str = Field(default="claude-sonnet-5", alias="CLASSIFIER_MODEL")
    kyc_model: str = Field(default="claude-sonnet-5", alias="KYC_MODEL")
    claims_model: str = Field(default="claude-sonnet-5", alias="CLAIMS_MODEL")
    policy_rag_model: str = Field(default="claude-sonnet-5", alias="POLICY_RAG_MODEL")
    fraud_model: str = Field(default="claude-sonnet-5", alias="FRAUD_MODEL")
    orchestrator_model: str = Field(default="claude-sonnet-5", alias="ORCHESTRATOR_MODEL")
    escalation_model: str = Field(default="claude-opus-5", alias="ESCALATION_MODEL")

    # --- Orchestrator decision thresholds (assignment Component 8) ---
    fraud_approve_max: float = Field(default=0.3, alias="FRAUD_APPROVE_MAX")
    fraud_escalate_min: float = Field(default=0.3, alias="FRAUD_ESCALATE_MIN")
    confidence_escalate_max: float = Field(default=0.6, alias="CONFIDENCE_ESCALATE_MAX")
    fraud_model_escalation_low: float = Field(default=0.2, alias="FRAUD_MODEL_ESCALATION_LOW")
    fraud_model_escalation_high: float = Field(default=0.5, alias="FRAUD_MODEL_ESCALATION_HIGH")
    # risk_level buckets: LOW < fraud_escalate_min <= MEDIUM < fraud_risk_high_min <= HIGH.
    # Deliberately reuses fraud_escalate_min as the LOW/MEDIUM boundary so
    # "MEDIUM risk" lines up exactly with the Orchestrator's ESCALATE trigger.
    fraud_risk_high_min: float = Field(default=0.6, alias="FRAUD_RISK_HIGH_MIN")

    # --- LangSmith tracing (optional) ---
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="medishield-intake", alias="LANGCHAIN_PROJECT")

    # --- Storage ---
    case_db_path: str = Field(default="./storage/cases.db", alias="CASE_DB_PATH")
    upload_dir: str = Field(default="./storage/uploads", alias="UPLOAD_DIR")
    chroma_persist_dir: str = Field(default="./backend/rag/chroma_db", alias="CHROMA_PERSIST_DIR")
    llm_cache_path: str = Field(default="./storage/llm_cache.db", alias="LLM_CACHE_PATH")
    checkpoint_db_path: str = Field(default="./storage/checkpoints.db", alias="CHECKPOINT_DB_PATH")
    token_usage_log_path: str = Field(default="./storage/token_usage.jsonl", alias="TOKEN_USAGE_LOG_PATH")

    # --- Dev cost controls ---
    # When true, every agent call is forced onto `testing_model` regardless of
    # its configured production model — for cheap "does the pipeline run"
    # smoke tests, not for prompt/behavior tuning (see IMPLEMENTATION_CHALLENGES.md).
    use_cheap_models_for_testing: bool = Field(default=False, alias="USE_CHEAP_MODELS_FOR_TESTING")
    testing_model: str = Field(default="claude-haiku-4-5", alias="TESTING_MODEL")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    # --- Security ---
    api_auth_tokens: str = Field(default="dev-local-token", alias="API_AUTH_TOKENS")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def api_auth_tokens_list(self) -> list[str]:
        return [token.strip() for token in self.api_auth_tokens.split(",") if token.strip()]

    def resolved_path(self, relative: str) -> Path:
        """Resolve a storage-style path relative to the repo root."""
        path = Path(relative)
        return path if path.is_absolute() else REPO_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
