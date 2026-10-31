"""Unit tests for the Three-Way Multimodal Ablation Benchmark (Task 7.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.run_multimodal_benchmark import (
    PipelineVariant,
    ProductionCIGates,
    compute_answer_faithfulness,
    compute_context_recall_hit,
    compute_p95,
    compute_structural_table_f1,
    compute_token_f1,
    format_markdown_report,
    main,
    run_three_way_benchmark,
    score_answer_accuracy,
)
from scripts.benchmark.content import CLOUD_TABLE_1
from src.ingestion.docling_parser import reconstruct_table_artifacts
from src.schemas.api import RoutingPath, VisualCitation
from src.schemas.document import BoundingBox, ChunkModality, CoordOrigin, MultimodalChunk


def test_compute_token_f1_and_answer_accuracy() -> None:
    assert compute_token_f1("18,400 ops/sec", "18,400 ops/sec") == 1.0
    assert compute_token_f1("", "18,400 ops/sec") == 0.0
    assert score_answer_accuracy(
        "Based on Table 1, the cache-on throughput is 18,400 ops/sec at 64 threads.",
        "18,400 ops/sec",
    ) == 1.0
    assert score_answer_accuracy(
        "The throughput is 11,200 ops/sec.",
        "18,400 ops/sec",
    ) == 0.0


def test_compute_structural_table_f1_markdown_vs_flattened() -> None:
    md_table, _ = reconstruct_table_artifacts(CLOUD_TABLE_1)
    structured_f1 = compute_structural_table_f1(md_table, CLOUD_TABLE_1)
    assert structured_f1 >= 0.94

    flattened_text = "Aurora PostgreSQL db.r7g.4xlarge 16 128 2.32 18,400 11,200 12.4 28.1"
    naive_f1 = compute_structural_table_f1(flattened_text, CLOUD_TABLE_1)
    assert 0.15 <= naive_f1 < 0.65
    assert structured_f1 > naive_f1


def test_compute_context_recall_and_faithfulness() -> None:
    bbox = BoundingBox(page=3, x0=0.1, y0=0.2, x1=0.9, y1=0.8, coord_origin=CoordOrigin.TOPLEFT)
    citation = VisualCitation(
        chunk_id="doc_cloud_p03_fig01",
        doc_name="cloud_database_benchmark_2026.pdf",
        page_number=3,
        modality=ChunkModality.FIGURE,
        section_title="3. P99 Latency Under Concurrency",
        rerank_score=0.92,
        bounding_box=bbox,
        image_url="/api/v1/crops/doc_cloud_p03_fig01.png",
        content_preview="[FIGURE CAPTION] Standard RDS crosses 500 ms at 250 connections.",
    )
    assert compute_context_recall_hit(
        [citation],
        expected_doc="cloud_database_benchmark_2026.pdf",
        expected_pages=[3],
        expected_modalities=["figure"],
    )
    assert not compute_context_recall_hit(
        [citation],
        expected_doc="cloud_database_benchmark_2026.pdf",
        expected_pages=[2],
        expected_modalities=["table"],
    )

    chunk = MultimodalChunk(
        chunk_id="doc_cloud_p03_fig01",
        doc_id="doc_cloud",
        doc_name="cloud_database_benchmark_2026.pdf",
        page_number=3,
        modality=ChunkModality.FIGURE,
        content="Standard RDS crosses 500 ms at 250 concurrent connections (44 ms for Aurora).",
        image_path="data/crops/doc_cloud_p03_fig01.png",
        bounding_box=bbox,
    )
    assert (
        compute_answer_faithfulness(
            "Standard RDS exceeds 500 ms at 250 concurrent connections.",
            [chunk],
        )
        == 1.0
    )
    assert compute_p95([10.0, 20.0, 30.0, 40.0, 100.0]) > 40.0


@pytest.mark.asyncio
async def test_run_three_way_benchmark_end_to_end(tmp_path: Path) -> None:
    output_json = tmp_path / "benchmark_report.json"
    output_md = tmp_path / "benchmark_report.md"

    report = await run_three_way_benchmark(
        output_json_path=output_json,
        output_md_path=output_md,
    )

    assert output_json.exists()
    assert output_md.exists()
    assert report.total_questions == 30
    assert report.omnidoc_passed_all_gates is True

    v_pypdf = report.variants[PipelineVariant.PYPDF_BASELINE.value]
    v_docling = report.variants[PipelineVariant.DOCLING_TEXT_TABLE.value]
    v_omnidoc = report.variants[PipelineVariant.OMNIDOC_VISION.value]
    gates = ProductionCIGates()

    # Verify ablation hierarchy: OmniDoc-Vision >= Docling Text/Table > PyPDF Baseline
    assert v_docling.table_cell_f1 >= gates.min_table_cell_f1
    assert v_omnidoc.table_cell_f1 >= gates.min_table_cell_f1
    assert v_omnidoc.table_cell_f1 > v_pypdf.table_cell_f1

    assert v_omnidoc.visual_qa_accuracy >= gates.min_visual_qa_accuracy
    assert v_omnidoc.visual_qa_accuracy > v_docling.visual_qa_accuracy
    assert v_omnidoc.visual_qa_accuracy > v_pypdf.visual_qa_accuracy

    assert v_omnidoc.context_recall_at_5 >= gates.min_context_recall_at_5
    assert v_omnidoc.faithfulness >= gates.min_faithfulness
    assert v_omnidoc.fast_path_p95_ms <= gates.max_fast_path_p95_ms
    assert v_omnidoc.visual_path_p95_ms is not None
    assert v_omnidoc.visual_path_p95_ms <= gates.max_visual_path_p95_ms

    # Verify visual attribution preservation across OmniDoc-Vision results
    visual_questions = [
        q for q in v_omnidoc.question_results if q.category == "visual"
    ]
    assert len(visual_questions) == 10
    for vq in visual_questions:
        assert vq.routing_path is RoutingPath.VISUAL_VLM_PATH
        assert 1 <= vq.vlm_images_inspected <= 2
        assert any(
            cit.modality is ChunkModality.FIGURE
            and cit.image_url is not None
            and cit.image_url.startswith("/api/v1/crops/")
            and 0.0 <= cit.bounding_box.x0 < cit.bounding_box.x1 <= 1.0
            for cit in vq.citations
        )

    md_text = format_markdown_report(report)
    assert "OmniDoc-Vision Three-Way Multimodal Benchmark Report" in md_text
    assert "Table Cell Extraction F1-Score" in md_text

    parsed_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert parsed_json["total_questions"] == 30
    assert parsed_json["omnidoc_passed_all_gates"] is True


def test_cli_main_entrypoint(tmp_path: Path) -> None:
    out_file = tmp_path / "cli_report.json"
    exit_code = main(["--output", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()
    assert out_file.with_suffix(".md").exists()
