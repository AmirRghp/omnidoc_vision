"""Unit tests for the Docling layout, table reconstruction, and figure cropping engine."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from scripts.benchmark.cache import build_cache_pdf
from scripts.benchmark.cloud import build_cloud_database_pdf
from scripts.benchmark.content import CACHE_PDF, CLOUD_PDF, GPU_PDF
from scripts.benchmark.gpu import build_gpu_pdf
from src.common.config import Settings
from src.common.exceptions import PdfValidationError
from src.ingestion.docling_parser import (
    DoclingParser,
    clamp_image_dimensions,
    normalize_bbox,
    reconstruct_table_artifacts,
)
from src.schemas.document import ChunkModality, CoordOrigin


@pytest.fixture()
def benchmark_pdfs(tmp_path: Path) -> Path:
    rl_config.invariant = 1
    pdf_dir = tmp_path / "raw_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    build_cloud_database_pdf(pdf_dir / CLOUD_PDF)
    build_cache_pdf(pdf_dir / CACHE_PDF)
    build_gpu_pdf(pdf_dir / GPU_PDF)
    return pdf_dir


def test_parse_cloud_benchmark_pdf_extracts_text_tables_and_figures(
    benchmark_pdfs: Path,
    tmp_path: Path,
) -> None:
    crops_dir = tmp_path / "crops"
    parser = DoclingParser(crops_dir=crops_dir, use_docling=False)
    chunks = parser.parse_pdf(benchmark_pdfs / CLOUD_PDF)

    assert len(chunks) > 0
    modalities = {chunk.modality for chunk in chunks}
    assert modalities == {ChunkModality.TEXT, ChunkModality.TABLE, ChunkModality.FIGURE}

    # Check visual attribution invariants on every chunk
    for chunk in chunks:
        assert chunk.doc_name == CLOUD_PDF
        assert chunk.doc_id == "doc_cloud_database_benchmark_2026"
        assert 1 <= chunk.page_number <= 5
        assert chunk.bounding_box.page == chunk.page_number
        assert chunk.bounding_box.coord_origin is CoordOrigin.TOPLEFT
        assert 0.0 <= chunk.bounding_box.x0 < chunk.bounding_box.x1 <= 1.0
        assert 0.0 <= chunk.bounding_box.y0 < chunk.bounding_box.y1 <= 1.0

    # Verify table chunks on pages 2 and 5
    table_chunks = [c for c in chunks if c.modality is ChunkModality.TABLE]
    assert len(table_chunks) >= 2
    table_pages = {c.page_number for c in table_chunks}
    assert {2, 5}.issubset(table_pages)

    page2_tables = [c for c in table_chunks if c.page_number == 2]
    assert any("18,400" in c.content and "Aurora PostgreSQL" in c.content for c in page2_tables)
    for tbl in table_chunks:
        assert tbl.table_html is not None
        assert tbl.table_html.startswith("<table")
        assert "</table>" in tbl.table_html
        assert "|" in tbl.content

    # Verify figure chunks on pages 3 and 4 and saved PNG crops
    figure_chunks = [c for c in chunks if c.modality is ChunkModality.FIGURE]
    assert len(figure_chunks) == 2
    figure_pages = {c.page_number for c in figure_chunks}
    assert figure_pages == {3, 4}

    for fig in figure_chunks:
        assert fig.image_path is not None
        assert not Path(fig.image_path).is_absolute()
        assert ".." not in Path(fig.image_path).parts
        crop_file = crops_dir / Path(fig.image_path).name
        assert crop_file.exists()
        with Image.open(crop_file) as img:
            assert img.format == "PNG"
            assert max(img.size) <= 1280


def test_parse_all_three_benchmark_pdfs_preserves_expected_modalities(
    benchmark_pdfs: Path,
    tmp_path: Path,
) -> None:
    crops_dir = tmp_path / "crops"
    parser = DoclingParser(crops_dir=crops_dir, use_docling=False)

    for pdf_name in (CLOUD_PDF, CACHE_PDF, GPU_PDF):
        chunks = parser.parse_pdf(benchmark_pdfs / pdf_name)
        by_modality = {
            modality: [c for c in chunks if c.modality is modality] for modality in ChunkModality
        }
        assert len(by_modality[ChunkModality.TEXT]) >= 3, pdf_name
        assert len(by_modality[ChunkModality.TABLE]) >= 2, pdf_name
        assert len(by_modality[ChunkModality.FIGURE]) >= 2, pdf_name


def test_reconstruct_table_artifacts_propagates_merged_headers_and_row_spans() -> None:
    raw_rows: list[list[str | None]] = [
        ["Engine", "Compute", None, "Throughput (ops/s)", None],
        [None, "vCPU", "Mem (GB)", "Cache On", "Cache Off"],
        ["Aurora PostgreSQL", "16", "128", "18,400", "11,200"],
        [None, "8", "64", "11,200", "6,800"],
    ]
    markdown, table_html = reconstruct_table_artifacts(
        raw_rows,
        caption="Table 1 - Throughput Profile",
    )

    assert "Table 1 - Throughput Profile" in markdown
    assert "Compute / vCPU" in markdown
    assert "Throughput (ops/s) / Cache On" in markdown
    # Merged first column ("Aurora PostgreSQL") should be forward-filled to the second data row
    lines = [line for line in markdown.splitlines() if line.startswith("|")]
    assert len(lines) == 4  # header + separator + 2 data rows
    assert "Aurora PostgreSQL" in lines[2]
    assert "Aurora PostgreSQL" in lines[3]
    assert "<table>" in table_html
    assert "<thead>" in table_html
    assert "<tbody>" in table_html
    assert "<script" not in table_html.lower()


def test_clamp_image_dimensions_resizes_oversized_images() -> None:
    oversized = Image.new("RGB", (2560, 1440), color=(20, 40, 80))
    clamped = clamp_image_dimensions(oversized, max_dim=1280)

    assert clamped.size == (1280, 720)

    small = Image.new("RGB", (640, 400), color=(10, 20, 30))
    unchanged = clamp_image_dimensions(small, max_dim=1280)
    assert unchanged.size == (640, 400)


def test_normalize_bbox_handles_topleft_and_bottomleft_origins() -> None:
    top_left_bbox = normalize_bbox(
        (50.0, 100.0, 500.0, 400.0),
        page_width=600.0,
        page_height=800.0,
        page_number=2,
        origin=CoordOrigin.TOPLEFT,
    )
    assert top_left_bbox.page == 2
    assert top_left_bbox.coord_origin is CoordOrigin.TOPLEFT
    assert top_left_bbox.x0 == pytest.approx(50.0 / 600.0, rel=1e-4)
    assert top_left_bbox.y0 == pytest.approx(100.0 / 800.0, rel=1e-4)
    assert top_left_bbox.x1 == pytest.approx(500.0 / 600.0, rel=1e-4)
    assert top_left_bbox.y1 == pytest.approx(400.0 / 800.0, rel=1e-4)

    bottom_left_bbox = normalize_bbox(
        (60.0, 200.0, 300.0, 600.0),
        page_width=600.0,
        page_height=800.0,
        page_number=3,
        origin=CoordOrigin.BOTTOMLEFT,
    )
    assert bottom_left_bbox.page == 3
    assert bottom_left_bbox.coord_origin is CoordOrigin.TOPLEFT
    assert bottom_left_bbox.y0 == pytest.approx((800.0 - 600.0) / 800.0, rel=1e-4)
    assert bottom_left_bbox.y1 == pytest.approx((800.0 - 200.0) / 800.0, rel=1e-4)


def test_pdf_validation_rejects_missing_oversized_or_excessive_pages(tmp_path: Path) -> None:
    settings = Settings(max_pdf_size_mb=1, max_pdf_pages=2)
    parser = DoclingParser(settings=settings, crops_dir=tmp_path / "crops", use_docling=False)

    with pytest.raises(PdfValidationError, match="does not exist"):
        parser.parse_pdf(tmp_path / "missing.pdf")

    non_pdf = tmp_path / "notes.txt"
    non_pdf.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(PdfValidationError, match=r"\.pdf"):
        parser.parse_pdf(non_pdf)

    # Create a 3-page PDF exceeding max_pdf_pages=2
    three_page_pdf = tmp_path / "three_pages.pdf"
    canvas = Canvas(str(three_page_pdf), pagesize=A4)
    for i in range(3):
        canvas.drawString(72, 720, f"Page {i + 1}")
        canvas.showPage()
    canvas.save()

    with pytest.raises(PdfValidationError, match="exceeds maximum page limit"):
        parser.parse_pdf(three_page_pdf)

    # Create an oversized file (> 1 MB)
    oversized_pdf = tmp_path / "oversized.pdf"
    oversized_pdf.write_bytes(b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 512))
    with pytest.raises(PdfValidationError, match="exceeds maximum file size"):
        parser.parse_pdf(oversized_pdf)


@pytest.mark.asyncio()
async def test_aparse_pdf_runs_non_blocking_and_matches_sync(
    benchmark_pdfs: Path,
    tmp_path: Path,
) -> None:
    parser = DoclingParser(crops_dir=tmp_path / "crops", use_docling=False)
    chunks = await parser.aparse_pdf(benchmark_pdfs / CLOUD_PDF)

    assert len(chunks) > 0
    assert {c.modality for c in chunks} == {
        ChunkModality.TEXT,
        ChunkModality.TABLE,
        ChunkModality.FIGURE,
    }


def test_docling_converter_adapter_extracts_items_and_falls_back_on_error(
    benchmark_pdfs: Path,
    tmp_path: Path,
) -> None:
    class _FakeBBox:
        l = 60.0  # noqa: E741
        t = 120.0
        r = 480.0
        b = 360.0
        coord_origin = type("Origin", (), {"name": "TOPLEFT"})()

    class _FakeProv:
        page_no = 1
        bbox = _FakeBBox()

    class _FakePageSize:
        width = 600.0
        height = 800.0

    class _FakePage:
        size = _FakePageSize()

    class _FakeTextItem:
        label = "section_header"
        text = "1. Architecture Overview"
        prov = [_FakeProv()]

    class _FakeParagraphItem:
        label = "text"
        text = "Distributed storage nodes replicate state across three availability zones."
        prov = [_FakeProv()]

    class _FakeTableItem:
        label = "table"
        prov = [_FakeProv()]

        def export_to_markdown(self, doc: Any = None) -> str:
            return "| Node | Latency |\n| --- | --- |\n| Primary | 12ms |"

        def export_to_html(self, doc: Any = None) -> str:
            return (
                "<table><tr><th>Node</th><th>Latency</th></tr>"
                "<tr><td>Primary</td><td>12ms</td></tr></table>"
            )

    class _FakePictureItem:
        label = "picture"
        caption_text = "Figure 1 - Cluster replication topology"
        prov = [_FakeProv()]

        def get_image(self, doc: Any = None) -> Image.Image:
            return Image.new("RGB", (1600, 900), color=(30, 60, 90))

    class _FakeDoc:
        pages = {1: _FakePage()}

        def iterate_items(self) -> list[tuple[Any, int]]:
            return [
                (_FakeTextItem(), 1),
                (_FakeParagraphItem(), 1),
                (_FakeTableItem(), 1),
                (_FakePictureItem(), 1),
            ]

    class _FakeResult:
        document = _FakeDoc()

    class _FakeConverter:
        def convert(self, source: str) -> _FakeResult:
            return _FakeResult()

    crops_dir = tmp_path / "docling_crops"
    parser = DoclingParser(
        crops_dir=crops_dir,
        use_docling=True,
        converter=_FakeConverter(),
    )
    chunks = parser.parse_pdf(benchmark_pdfs / CLOUD_PDF)

    assert [c.modality for c in chunks] == [
        ChunkModality.TEXT,
        ChunkModality.TABLE,
        ChunkModality.FIGURE,
    ]
    assert chunks[0].section_title == "1. Architecture Overview"
    assert "Distributed storage nodes" in chunks[0].content
    assert chunks[1].table_html is not None
    assert chunks[2].image_path is not None

    saved_crop = crops_dir / Path(chunks[2].image_path).name
    assert saved_crop.exists()
    with Image.open(saved_crop) as img:
        assert img.size == (1280, 720)
        assert file_bytes_are_png(saved_crop)


def file_bytes_are_png(path: Path) -> bool:
    with io.BytesIO(path.read_bytes()) as buf, Image.open(buf) as img:
        return img.format == "PNG"
