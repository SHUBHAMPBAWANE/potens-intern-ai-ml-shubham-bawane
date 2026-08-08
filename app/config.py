"""
Application configuration.

Centralizes environment-driven settings using Pydantic Settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    """

    app_name: str = "RAG Application"

    # Gemini
    gemini_api_key: str

    # Gemini model
    gemini_model_name: str = "gemini-3.5-flash"

    # RAG settings
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    vector_store_path: str = "data/chroma"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """

    return Settings()