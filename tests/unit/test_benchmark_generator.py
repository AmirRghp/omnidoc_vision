"""Contract tests for the deterministic benchmark corpus generator."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader
from reportlab import rl_config

from scripts.benchmark.cache import build_cache_pdf
from scripts.benchmark.cloud import build_cloud_database_pdf
from scripts.benchmark.content import (
    CACHE_LFU_HIT,
    CACHE_LRU_HIT,
    CACHE_PDF,
    CLOUD_PDF,
    CLOUD_PEAK_THROUGHPUT,
    CLOUD_TABLE_1,
    GPU_AWQ_P95,
    GPU_AWQ_THROUGHPUT,
    GPU_PDF,
    GPU_TABLE_1,
)
from scripts.benchmark.dataset import build_golden_dataset, validate_dataset, write_dataset
from scripts.benchmark.gpu import build_gpu_pdf


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    rl_config.invariant = 1
    build_cloud_database_pdf(tmp_path / CLOUD_PDF)
    build_cache_pdf(tmp_path / CACHE_PDF)
    build_gpu_pdf(tmp_path / GPU_PDF)
    return tmp_path


def _reader(directory: Path, name: str) -> PdfReader:
    return PdfReader(str(directory / name))


def test_pdfs_have_expected_pages_and_visual_pages(corpus: Path) -> None:
    for name in (CLOUD_PDF, CACHE_PDF, GPU_PDF):
        reader = _reader(corpus, name)
        assert len(reader.pages) == 5
        assert any(len(page.images) > 0 for page in reader.pages)


def test_golden_dataset_contract_and_page_modalities(corpus: Path) -> None:
    dataset = build_golden_dataset()
    validate_dataset(dataset)

    questions = dataset["questions"]
    assert len(questions) == 30
    assert {question["category"] for question in questions} == {"table", "visual", "text"}
    assert all(question["expected_pages"] for question in questions)

    readers = {
        CLOUD_PDF: _reader(corpus, CLOUD_PDF),
        CACHE_PDF: _reader(corpus, CACHE_PDF),
        GPU_PDF: _reader(corpus, GPU_PDF),
    }
    for question in questions:
        page_number = question["expected_pages"][0]
        page = readers[question["doc_name"]].pages[page_number - 1]
        modality = question["expected_modalities"][0]
        if modality == "figure":
            assert len(page.images) > 0, question["id"]
        elif modality == "table":
            text = page.extract_text() or ""
            assert "Table" in text, question["id"]
        else:
            assert (page.extract_text() or "").strip(), question["id"]


def test_pdf_and_dataset_outputs_are_deterministic(tmp_path: Path) -> None:
    rl_config.invariant = 1
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        build_cloud_database_pdf(directory / CLOUD_PDF)
        build_cache_pdf(directory / CACHE_PDF)
        build_gpu_pdf(directory / GPU_PDF)

    for name in (CLOUD_PDF, CACHE_PDF, GPU_PDF):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    write_dataset(build_golden_dataset(), first / "dataset.json")
    write_dataset(build_golden_dataset(), second / "dataset.json")
    assert (first / "dataset.json").read_bytes() == (second / "dataset.json").read_bytes()


def test_tables_use_the_shared_series() -> None:
    assert CLOUD_TABLE_1[2][5] == f"{CLOUD_PEAK_THROUGHPUT:,}"
    assert CACHE_LRU_HIT[2] == 88.4
    assert CACHE_LRU_HIT[4] == 94.1
    assert CACHE_LFU_HIT[4] == 96.3
    for row_index, (p95, throughput) in enumerate(
        zip(GPU_AWQ_P95, GPU_AWQ_THROUGHPUT, strict=True), start=2
    ):
        assert GPU_TABLE_1[row_index][2] == str(p95)
        assert GPU_TABLE_1[row_index][3] == f"{throughput:,}"


def test_gpu_recommendation_is_not_described_as_hard_maximum(corpus: Path) -> None:
    reader = _reader(corpus, GPU_PDF)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    assert "recommended batch size of 8" in text
    assert "maximum batch size of 8" not in text
    assert "Batch 16 fits the 16 GB ceiling" in text
