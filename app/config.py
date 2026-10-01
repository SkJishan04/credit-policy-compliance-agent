"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "credit_policies"

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Data
    policy_data_dir: str = "./data/policies"

    # Retrieval
    top_k_retrieval: int = 4

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Frontend
    backend_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"

    @property
    def policy_data_path(self) -> Path:
        return Path(self.policy_data_dir)

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()