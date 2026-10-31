from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration shared by the API, ingestion, and retrieval layers."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "omnidoc-vision"
    environment: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "0.0.0.0"
    port: int = Field(default=8001, ge=1, le=65535)

    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_api_key: SecretStr = SecretStr("EMPTY")
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vlm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    vlm_max_tokens: int = Field(default=1024, ge=1)

    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"

    qdrant_host: str = "localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_collection: str = "omnidoc_vision_chunks"
    bm25_index_dir: Path = Path("data/bm25_index")

    redis_url: str = "redis://localhost:6379/0"

    raw_pdfs_dir: Path = Path("data/raw_pdfs")
    crops_dir: Path = Path("data/crops")
    max_image_dim: int = Field(default=1280, ge=1, le=1280)
    max_pdf_size_mb: int = Field(default=50, ge=1)
    max_pdf_pages: int = Field(default=100, ge=1)
    max_images_per_query: int = Field(default=2, ge=0, le=2)
    docling_device: str = "cpu"

    @field_validator("bm25_index_dir", "raw_pdfs_dir", "crops_dir", mode="after")
    @classmethod
    def _resolve_data_path(cls, value: Path) -> Path:
        """Keep configured data paths stable regardless of the process cwd."""
        return value if value.is_absolute() else PROJECT_ROOT / value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
