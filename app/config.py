"""
Application configuration.

Centralizes environment-driven settings (API keys, model names,
vector store paths, chunking parameters, etc.) using pydantic BaseSettings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    TODO: Define configuration fields, e.g.:
    - app_name: str
    - embedding_model_name: str
    - vector_store_path: str
    - chunk_size: int
    - chunk_overlap: int
    - translation_provider: str
    - log_level: str
    """

    app_name: str = "RAG Application"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    TODO: Add validation / environment-specific overrides if needed.
    """
    return Settings()
