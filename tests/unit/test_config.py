from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config import PROJECT_ROOT, Settings, get_settings

EXPECTED_FIELDS = {
    "app_name",
    "environment",
    "log_level",
    "host",
    "port",
    "vlm_base_url",
    "vlm_api_key",
    "vlm_model_name",
    "vlm_temperature",
    "vlm_max_tokens",
    "embedding_model_name",
    "reranker_model_name",
    "embedding_device",
    "reranker_device",
    "qdrant_host",
    "qdrant_port",
    "qdrant_collection",
    "bm25_index_dir",
    "redis_url",
    "raw_pdfs_dir",
    "crops_dir",
    "max_image_dim",
    "max_pdf_size_mb",
    "max_pdf_pages",
    "max_images_per_query",
    "docling_device",
}


@pytest.fixture(autouse=True)
def _isolated_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_model_covers_env_template() -> None:
    assert set(Settings.model_fields) == EXPECTED_FIELDS


def test_settings_defaults_match_env_template() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "omnidoc-vision"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8001
    assert settings.vlm_base_url == "http://localhost:8000/v1"
    assert settings.vlm_model_name == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert settings.vlm_temperature == 0.1
    assert settings.vlm_max_tokens == 1024
    assert settings.embedding_model_name == "BAAI/bge-m3"
    assert settings.reranker_model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.embedding_device == "cpu"
    assert settings.reranker_device == "cpu"
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_collection == "omnidoc_vision_chunks"
    assert settings.bm25_index_dir == PROJECT_ROOT / "data/bm25_index"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.raw_pdfs_dir == PROJECT_ROOT / "data/raw_pdfs"
    assert settings.crops_dir == PROJECT_ROOT / "data/crops"
    assert settings.max_image_dim == 1280
    assert settings.max_pdf_size_mb == 50
    assert settings.max_pdf_pages == 100
    assert settings.max_images_per_query == 2
    assert settings.docling_device == "cpu"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_IMAGE_DIM", "640")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VLM_API_KEY", "local-secret")

    settings = Settings(_env_file=None)

    assert settings.max_image_dim == 640
    assert settings.log_level == "DEBUG"
    assert settings.vlm_api_key.get_secret_value() == "local-secret"


def test_env_template_loads_and_matches_settings() -> None:
    template = PROJECT_ROOT / ".env.example"
    settings = Settings(_env_file=template)

    assert settings.port == 8001
    assert settings.raw_pdfs_dir == PROJECT_ROOT / "data/raw_pdfs"
    assert settings.crops_dir == PROJECT_ROOT / "data/crops"
    assert settings.bm25_index_dir == PROJECT_ROOT / "data/bm25_index"


def test_relative_data_paths_are_project_root_relative() -> None:
    settings = Settings(_env_file=None, crops_dir=Path("custom/crops"))

    assert settings.crops_dir == PROJECT_ROOT / "custom/crops"


def test_absolute_data_paths_are_preserved(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, crops_dir=tmp_path)

    assert settings.crops_dir == tmp_path


def test_secret_is_redacted_in_repr() -> None:
    settings = Settings(_env_file=None, vlm_api_key="super-secret")

    assert settings.vlm_api_key.get_secret_value() == "super-secret"
    assert "super-secret" not in repr(settings)


def test_security_bounds_reject_unsafe_values() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_image_dim=2048)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_images_per_query=3)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, port=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vlm_temperature=3.0)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "cached-app")

    first = get_settings()
    monkeypatch.setenv("APP_NAME", "changed-app")
    second = get_settings()

    assert first is second
    assert first.app_name == "cached-app"
