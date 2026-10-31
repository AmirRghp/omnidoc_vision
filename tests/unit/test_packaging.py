"""Guard the package metadata and environment template against regressions."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dependency_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(re.split(r"[<>=!~ ]", line, maxsplit=1)[0].lower())
    return names


def test_project_metadata_references_existing_readme_and_wheel_package() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert (PROJECT_ROOT / metadata["project"]["readme"]).is_file()
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src"]


def test_dependency_manifests_are_aligned() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = {
        re.split(r"[<>=!~ ]", dependency, maxsplit=1)[0].lower()
        for dependency in metadata["project"]["dependencies"]
    }
    requirements = _dependency_names(PROJECT_ROOT / "requirements.txt")

    assert project_dependencies <= requirements
    assert "python-multipart" in project_dependencies
    assert "python-multipart" in requirements
    assert "sentence-transformers" not in requirements


def test_env_template_has_only_assignments_and_comments() -> None:
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert "=" in stripped, stripped


def test_env_template_does_not_share_api_and_vllm_port() -> None:
    values: dict[str, str] = {}
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", maxsplit=1)
            values[key.strip()] = value.strip()

    assert values["PORT"] == "8001"
    assert ":8000/" in values["VLM_BASE_URL"]


def test_docker_and_compose_configs_include_auto_ingest_and_services() -> None:
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    compose_path = PROJECT_ROOT / "docker-compose.yml"

    assert dockerfile_path.is_file()
    assert compose_path.is_file()

    dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile_text
    assert "USER omnidoc" in dockerfile_text
    assert "HEALTHCHECK" in dockerfile_text

    compose_text = compose_path.read_text(encoding="utf-8")
    for required_service in ("qdrant:", "redis:", "langfuse:", "init-ingest:", "omnidoc-api:"):
        assert required_service in compose_text
    assert "build_benchmark_pdfs" in compose_text
    assert "service_completed_successfully" in compose_text


def test_github_actions_ci_workflow_runs_ruff_mypy_and_pytest() -> None:
    ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.is_file()

    ci_text = ci_path.read_text(encoding="utf-8")
    assert "ruff check ." in ci_text
    assert "mypy src" in ci_text
    assert "pytest" in ci_text


def test_portfolio_readme_contains_trilogy_links_vram_benchmarks_and_limitations() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for repo_name in ("omnidoc-rag", "omniops-sre-agent", "omnidoc-vision"):
        assert repo_name in readme_text

    assert "16GB VRAM" in readme_text
    assert "Three-Way Multimodal Ablation" in readme_text
    assert "Document Degradation & Noise Robustness" in readme_text
    assert "Limitations & Failure Modes" in readme_text

