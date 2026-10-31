"""Unit tests for Chainlit UI helpers, formatting, and crop resolution (Task 6.1)."""

from __future__ import annotations

from pathlib import Path

from app import (
    build_citations_markdown,
    build_telemetry_badge,
    get_starters,
    resolve_crop_paths,
)
from src.schemas.api import (
    CRAGStatus,
    LatencyBreakdown,
    MultimodalQueryResponse,
    RoutingPath,
    VisualCitation,
)
from src.schemas.document import BoundingBox, ChunkModality


def _make_dummy_response(
    routing_path: RoutingPath,
    crag_status: CRAGStatus = CRAGStatus.CONFIDENT,
    vlm_images: int = 1,
) -> MultimodalQueryResponse:
    return MultimodalQueryResponse(
        answer="پاسخ تحلیلی به پرسش کاربر درباره عملکرد دیتابیس.",
        routing_path=routing_path,
        crag_status=crag_status,
        rewritten_query=None,
        vlm_images_inspected=vlm_images,
        latency_ms=LatencyBreakdown(
            retrieval_ms=42.5,
            rerank_ms=31.0,
            generation_ms=250.2,
            total_ms=323.7,
        ),
        citations=[],
    )


def test_build_telemetry_badge_visual_path() -> None:
    """Verify telemetry badge for VISUAL_VLM_PATH contains violet class and image count."""
    resp = _make_dummy_response(RoutingPath.VISUAL_VLM_PATH, vlm_images=2)
    badge_html = build_telemetry_badge(resp)

    assert "route-pill visual-path" in badge_html
    assert "Visual VLM Reasoning" in badge_html
    assert "2 تصویر" in badge_html
    assert "323.7ms" in badge_html
    assert "crag-badge confident" in badge_html


def test_build_telemetry_badge_fast_path() -> None:
    """Verify telemetry badge for FAST_TEXT_TABLE_PATH contains cyan class and latency breakdown."""
    resp = _make_dummy_response(RoutingPath.FAST_TEXT_TABLE_PATH, CRAGStatus.LOW_CONFIDENCE)
    badge_html = build_telemetry_badge(resp)

    assert "route-pill fast-path" in badge_html
    assert "Fast Text/Table Path" in badge_html
    assert "بازیابی: 42.5ms" in badge_html
    assert "رتبه‌بندی: 31.0ms" in badge_html
    assert "crag-badge low-confidence" in badge_html


def test_resolve_crop_paths_finds_existing_figures(tmp_path: Path) -> None:
    """Verify existing figure crop files are resolved with appropriate labels."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_file = crops_dir / "bench_p03_fig01.png"
    crop_file.write_bytes(b"\x89PNG dummy")

    citations = [
        VisualCitation(
            chunk_id="bench_p03_fig01",
            doc_name="cloud_benchmark.pdf",
            page_number=3,
            modality=ChunkModality.FIGURE,
            section_title="Latency Curve",
            rerank_score=0.95,
            bounding_box=BoundingBox(page=3, x0=0.1, y0=0.2, x1=0.8, y1=0.7),
            image_url="/api/v1/crops/bench_p03_fig01.png",
            content_preview="[FIGURE] P99 Latency vs Concurrency",
        ),
        VisualCitation(
            chunk_id="bench_p02_tbl01",
            doc_name="cloud_benchmark.pdf",
            page_number=2,
            modality=ChunkModality.TABLE,
            section_title="DB Comparison",
            rerank_score=0.89,
            bounding_box=BoundingBox(page=2, x0=0.1, y0=0.2, x1=0.8, y1=0.5),
            image_url=None,
            content_preview="| Qdrant | 14200 QPS |",
        ),
    ]

    resolved = resolve_crop_paths(citations, crops_dir)
    assert len(resolved) == 1
    path, label = resolved[0]
    assert path == crop_file
    assert "صفحه 3" in label
    assert "bench_p03_fig01" in label


def test_resolve_crop_paths_rejects_path_traversal(tmp_path: Path) -> None:
    """Verify malicious filenames with traversal patterns are ignored."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    citation = VisualCitation(
        chunk_id="malicious_chunk",
        doc_name="report.pdf",
        page_number=1,
        modality=ChunkModality.FIGURE,
        section_title="Attack",
        rerank_score=0.99,
        bounding_box=BoundingBox(page=1, x0=0.1, y0=0.1, x1=0.9, y1=0.9),
        image_url="/api/v1/crops/../../secret.png",
        content_preview="malicious",
    )

    resolved = resolve_crop_paths([citation], crops_dir)
    assert len(resolved) == 0


def test_build_citations_markdown_empty() -> None:
    """Verify empty citations list produces empty markdown."""
    assert build_citations_markdown([]) == ""


def test_build_citations_markdown_formatting() -> None:
    """Verify markdown output formats details/summary tags and bounding box badges."""
    citations = [
        VisualCitation(
            chunk_id="chk_01",
            doc_name="distributed_cache.pdf",
            page_number=4,
            modality=ChunkModality.FIGURE,
            section_title="Cluster Topology",
            rerank_score=0.942,
            bounding_box=BoundingBox(page=4, x0=0.15, y0=0.25, x1=0.85, y1=0.75),
            image_url="/api/v1/crops/chk_01.png",
            content_preview="[FIGURE] 3-node Redis cluster with sentinel failover",
        )
    ]
    md = build_citations_markdown(citations)
    assert "### 📚 مستندات و شواهد استنادشده" in md
    assert "<details class='citation-card'>" in md
    assert "distributed_cache.pdf" in md
    assert "صفحه 4" in md
    assert "Cluster Topology" in md
    assert "[0.150, 0.250, 0.850, 0.750]" in md
    assert "0.942" in md


def test_get_starters_returns_three_starters() -> None:
    """Verify starter suggestions provide 3 diverse bilingual technical queries."""
    starters = get_starters()
    assert len(starters) == 3
    labels = [s.label for s in starters]
    assert any("مسیر بینایی" in lbl for lbl in labels)
    assert any("مسیر سریع جدول" in lbl for lbl in labels)
    assert any("مسیر دیاگرام" in lbl for lbl in labels)
