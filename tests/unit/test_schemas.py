from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.schemas.api import (
    ChunkCounts,
    CRAGStatus,
    IngestRequest,
    IngestResponse,
    LatencyBreakdown,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    RoutingPath,
    VisualCitation,
)
from src.schemas.document import BoundingBox, ChunkModality, CoordOrigin, MultimodalChunk

VALID_BBOX = {"page": 2, "x0": 85.0, "y0": 190.0, "x1": 510.0, "y1": 410.0}

TEXT_CHUNK = {
    "chunk_id": "doc_demo_p02_txt01",
    "doc_id": "doc_demo",
    "doc_name": "demo.pdf",
    "page_number": 2,
    "modality": "text",
    "section_title": "2.1 Throughput",
    "content": "The measured throughput was 12,400 ops/sec.",
    "bounding_box": VALID_BBOX,
}

TABLE_CHUNK = {
    **TEXT_CHUNK,
    "chunk_id": "doc_demo_p02_tbl01",
    "modality": "table",
    "content": "| Engine | Throughput |\n| --- | --- |\n| Qdrant | 12400 |",
    "table_html": "<table><tr><td>Qdrant</td><td>12400</td></tr></table>",
}

FIGURE_CHUNK = {
    **TEXT_CHUNK,
    "chunk_id": "doc_demo_p02_fig01",
    "modality": "figure",
    "content": "[FIGURE] P99 latency rises to 620 ms at 250 connections.",
    "image_path": "data/crops/doc_demo_p02_fig01.png",
}

QUERY_RESPONSE = {
    "answer": "بر اساس نمودار، از ۲۵۰ اتصال هم‌زمان تاخیر جهش می‌کند.",
    "routing_path": "VISUAL_VLM_PATH",
    "crag_status": "CONFIDENT",
    "rewritten_query": None,
    "vlm_images_inspected": 1,
    "latency_ms": {
        "retrieval_ms": 84.2,
        "rerank_ms": 62.1,
        "generation_ms": 785.4,
        "total_ms": 931.7,
    },
    "citations": [
        {
            "chunk_id": "doc_cloud_db_bench_2026_p04_fig01",
            "doc_name": "cloud_database_benchmark_2026.pdf",
            "page_number": 4,
            "modality": "figure",
            "section_title": "4.2 P99 Latency vs. Concurrency",
            "rerank_score": 0.942,
            "bounding_box": {
                "page": 4,
                "x0": 112.5,
                "y0": 240.0,
                "x1": 525.0,
                "y1": 598.5,
            },
            "image_url": "/api/v1/crops/doc_cloud_db_bench_2026_p04_fig01.png",
            "content_preview": "[FIGURE] P99 Write Latency (ms) vs Concurrent Connections.",
        }
    ],
}


def test_chunk_modality_values_are_lowercase() -> None:
    assert {modality.value for modality in ChunkModality} == {"text", "table", "figure"}
    assert ChunkModality.FIGURE == "figure"


def test_bounding_box_accepts_valid_geometry() -> None:
    bbox = BoundingBox(**VALID_BBOX)

    assert bbox.page == 2
    assert bbox.x0 == 85.0
    assert bbox.x1 == 510.0
    assert bbox.coord_origin is CoordOrigin.TOPLEFT


@pytest.mark.parametrize(
    "overrides",
    [
        {"x1": 85.0},
        {"y1": 190.0},
        {"x1": 10.0},
        {"y1": 10.0},
        {"x0": -1.0},
        {"page": 0},
    ],
)
def test_bounding_box_rejects_invalid_geometry(overrides: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        BoundingBox(**{**VALID_BBOX, **overrides})


def test_bounding_box_is_immutable() -> None:
    bbox = BoundingBox(**VALID_BBOX)

    with pytest.raises(ValidationError):
        bbox.x0 = 1.0  # type: ignore[misc]


def test_text_chunk_serializes_for_storage() -> None:
    chunk = MultimodalChunk(**TEXT_CHUNK)
    payload = chunk.model_dump(mode="json")

    assert payload["modality"] == "text"
    assert payload["bounding_box"]["page"] == 2
    assert payload["table_html"] is None
    assert payload["image_path"] is None
    assert isinstance(payload["ingestion_timestamp"], str)
    assert chunk.ingestion_timestamp.tzinfo is not None


def test_chunk_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MultimodalChunk(**TEXT_CHUNK, unexpected="value")


def test_table_chunk_requires_html() -> None:
    chunk = MultimodalChunk(**TABLE_CHUNK)
    assert chunk.table_html is not None

    without_html = {key: value for key, value in TABLE_CHUNK.items() if key != "table_html"}
    with pytest.raises(ValidationError):
        MultimodalChunk(**without_html)

    with pytest.raises(ValidationError):
        MultimodalChunk(**{**TABLE_CHUNK, "table_html": ""})


def test_table_chunk_accepts_optional_crop() -> None:
    chunk = MultimodalChunk(**{**TABLE_CHUNK, "image_path": "data/crops/table.png"})
    assert chunk.image_path == "data/crops/table.png"


def test_figure_chunk_requires_image_path() -> None:
    chunk = MultimodalChunk(**FIGURE_CHUNK)
    assert chunk.image_path is not None

    without_image = {key: value for key, value in FIGURE_CHUNK.items() if key != "image_path"}
    with pytest.raises(ValidationError):
        MultimodalChunk(**without_image)

    with pytest.raises(ValidationError):
        MultimodalChunk(**{**FIGURE_CHUNK, "table_html": "<table></table>"})


def test_text_chunk_rejects_visual_payloads() -> None:
    with pytest.raises(ValidationError):
        MultimodalChunk(**{**TEXT_CHUNK, "image_path": "data/crops/text.png"})

    with pytest.raises(ValidationError):
        MultimodalChunk(**{**TEXT_CHUNK, "table_html": "<table></table>"})


def test_chunk_page_must_match_bounding_box_page() -> None:
    with pytest.raises(ValidationError):
        MultimodalChunk(**{**TEXT_CHUNK, "page_number": 3})


def test_chunk_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        MultimodalChunk(**{**TEXT_CHUNK, "ingestion_timestamp": datetime(2026, 1, 1)})

    chunk = MultimodalChunk(
        **{**TEXT_CHUNK, "ingestion_timestamp": datetime(2026, 1, 1, tzinfo=UTC)}
    )
    assert chunk.ingestion_timestamp.year == 2026


def test_query_request_defaults_and_stripping() -> None:
    request = MultimodalQueryRequest(query="  سوال درباره تاخیر P99  ")

    assert request.query == "سوال درباره تاخیر P99"
    assert request.top_k == 5
    assert request.modality_filter is None
    assert request.force_visual_route is False


def test_query_request_validates_bounds_and_modality() -> None:
    assert MultimodalQueryRequest(query="q", modality_filter="figure").modality_filter is (
        ChunkModality.FIGURE
    )

    with pytest.raises(ValidationError):
        MultimodalQueryRequest(query="   ")
    with pytest.raises(ValidationError):
        MultimodalQueryRequest(query="q", top_k=0)
    with pytest.raises(ValidationError):
        MultimodalQueryRequest(query="q", top_k=21)
    with pytest.raises(ValidationError):
        MultimodalQueryRequest(query="q", modality_filter="image")
    with pytest.raises(ValidationError):
        MultimodalQueryRequest(query="q", unexpected=True)


def test_ingest_request_defaults_and_validation() -> None:
    request = IngestRequest(pdf_path="  data/raw_pdfs/demo.pdf  ")

    assert request.pdf_path == "data/raw_pdfs/demo.pdf"
    assert request.generate_vlm_captions is True

    with pytest.raises(ValidationError):
        IngestRequest(pdf_path="   ")


def test_chunk_counts_require_consistent_total() -> None:
    counts = ChunkCounts(text=18, table=4, figure=3, total=25)
    assert counts.total == 25

    with pytest.raises(ValidationError):
        ChunkCounts(text=18, table=4, figure=3, total=24)
    with pytest.raises(ValidationError):
        ChunkCounts(text=-1, table=4, figure=3, total=6)


def test_ingest_response_contract() -> None:
    response = IngestResponse(
        doc_id="doc_demo",
        doc_name="demo.pdf",
        pages_processed=6,
        chunks_created=ChunkCounts(text=18, table=4, figure=3, total=25),
        elapsed_ms=3420.5,
    )
    payload = response.model_dump(mode="json")

    assert payload["status"] == "indexed"
    assert payload["chunks_created"]["total"] == 25

    with pytest.raises(ValidationError):
        IngestResponse(
            doc_id="doc_demo",
            doc_name="demo.pdf",
            pages_processed=-1,
            chunks_created=ChunkCounts(text=1, table=0, figure=0, total=1),
            elapsed_ms=10.0,
        )


def test_visual_citation_requires_image_for_figures() -> None:
    citation = QUERY_RESPONSE["citations"][0]  # type: ignore[index]
    parsed = VisualCitation.model_validate(citation)
    assert parsed.image_url is not None

    with pytest.raises(ValidationError):
        VisualCitation.model_validate({**citation, "image_url": None})

    text_citation = {
        **citation,
        "modality": "text",
        "image_url": None,
        "content_preview": "A text passage.",
    }
    assert VisualCitation.model_validate(text_citation).image_url is None


def test_visual_citation_validates_score_and_page() -> None:
    citation = QUERY_RESPONSE["citations"][0]  # type: ignore[index]

    with pytest.raises(ValidationError):
        VisualCitation.model_validate({**citation, "rerank_score": 1.01})
    with pytest.raises(ValidationError):
        VisualCitation.model_validate({**citation, "rerank_score": -0.01})
    with pytest.raises(ValidationError):
        VisualCitation.model_validate({**citation, "page_number": 5})


def test_query_response_parses_documented_example() -> None:
    response = MultimodalQueryResponse.model_validate(QUERY_RESPONSE)
    payload = response.model_dump(mode="json")

    assert response.routing_path is RoutingPath.VISUAL_VLM_PATH
    assert response.crag_status is CRAGStatus.CONFIDENT
    assert response.vlm_images_inspected == 1
    assert payload["routing_path"] == "VISUAL_VLM_PATH"
    assert payload["latency_ms"]["total_ms"] == 931.7

    citation = payload["citations"][0]
    assert citation["page_number"] == 4
    assert citation["modality"] == "figure"
    assert citation["bounding_box"] == {
        "page": 4,
        "x0": 112.5,
        "y0": 240.0,
        "x1": 525.0,
        "y1": 598.5,
        "coord_origin": "TOPLEFT",
    }
    assert citation["image_url"] == "/api/v1/crops/doc_cloud_db_bench_2026_p04_fig01.png"


def test_query_response_defaults_and_negative_latency() -> None:
    response = MultimodalQueryResponse(
        answer="Direct answer",
        routing_path=RoutingPath.FAST_TEXT_TABLE_PATH,
        crag_status=CRAGStatus.LOW_CONFIDENCE,
        latency_ms=LatencyBreakdown(
            retrieval_ms=10.0,
            rerank_ms=5.0,
            generation_ms=20.0,
            total_ms=35.0,
        ),
    )

    assert response.citations == []
    assert response.rewritten_query is None
    assert response.vlm_images_inspected == 0

    with pytest.raises(ValidationError):
        LatencyBreakdown(retrieval_ms=-1.0, rerank_ms=0.0, generation_ms=0.0, total_ms=0.0)


def test_api_enums_use_documented_wire_values() -> None:
    assert RoutingPath.FAST_TEXT_TABLE_PATH == "FAST_TEXT_TABLE_PATH"
    assert RoutingPath.VISUAL_VLM_PATH == "VISUAL_VLM_PATH"
    assert CRAGStatus.CONFIDENT == "CONFIDENT"
    assert CRAGStatus.LOW_CONFIDENCE == "LOW_CONFIDENCE"
