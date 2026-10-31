"""Generate the self-contained OmniDoc-Vision benchmark corpus.

The script creates three deterministic, five-page engineering PDFs and a
30-question bilingual golden dataset. It uses ReportLab for document layout
and Matplotlib for charts and topology diagrams; no network or GPU is needed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader  # noqa: E402
from reportlab import rl_config  # noqa: E402

from scripts.benchmark.cache import build_cache_pdf  # noqa: E402
from scripts.benchmark.cloud import build_cloud_database_pdf  # noqa: E402
from scripts.benchmark.content import (  # noqa: E402
    CACHE_PDF,
    CLOUD_PDF,
    GPU_PDF,
    TOTAL_PAGES,
)
from scripts.benchmark.dataset import (  # noqa: E402
    build_golden_dataset,
    validate_dataset,
    write_dataset,
)
from scripts.benchmark.gpu import build_gpu_pdf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DATASET_PATH = PROJECT_ROOT / "data" / "golden_multimodal_dataset.json"
PDF_NAMES = (CLOUD_PDF, CACHE_PDF, GPU_PDF)


def _validate_pdf_artifacts(directory: Path) -> None:
    for name in PDF_NAMES:
        reader = PdfReader(str(directory / name))
        if len(reader.pages) != TOTAL_PAGES:
            raise ValueError(f"{name} must contain {TOTAL_PAGES} pages")
        if not any((page.extract_text() or "").strip() for page in reader.pages):
            raise ValueError(f"{name} contains no extractable text")


def _write_atomically(directory: Path) -> None:
    for name in PDF_NAMES:
        (directory / name).replace(RAW_PDF_DIR / name)


def main() -> None:
    """Generate all artifacts, validating PDFs and the golden dataset first."""
    rl_config.invariant = 1
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(dir=RAW_PDF_DIR) as temporary_dir:
            staging = Path(temporary_dir)
            build_cloud_database_pdf(staging / CLOUD_PDF)
            build_cache_pdf(staging / CACHE_PDF)
            build_gpu_pdf(staging / GPU_PDF)
            _validate_pdf_artifacts(staging)
            _write_atomically(staging)

        dataset = build_golden_dataset()
        validate_dataset(dataset)
        temporary_dataset = DATASET_PATH.with_name(f"{DATASET_PATH.name}.tmp")
        write_dataset(dataset, temporary_dataset)
        temporary_dataset.replace(DATASET_PATH)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Benchmark generation failed: {exc}") from exc

    print(f"Generated 3 PDFs in {RAW_PDF_DIR}")
    print("Generated 30 QA pairs (table=10, visual=10, text=10)")
    print(f"Wrote dataset to {DATASET_PATH}")


if __name__ == "__main__":
    main()
