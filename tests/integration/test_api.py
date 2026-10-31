"""Integration tests for FastAPI endpoints (Task 5.1).

Covers /ingest, /query, /crops/{filename}, and /health.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from main import create_app
from src.api.routes import (
    get_captioner_dep,
    get_indexer_dep,
    get_parser_dep,
    get_settings_dep,
    get_workflow_dep,
)
from src.common.config import Settings
from src.ingestion.indexer import IndexingStats
from src.schemas.api import (
    CRAGStatus,
    LatencyBreakdown,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    RoutingPath,
    VisualCitation,
)
from src.schemas.document import BoundingBox, ChunkModality, MultimodalChunk


class DummyParser:
    """Mock parser producing deterministic MultimodalChunk outputs."""

    def __init__(self, doc_name: str = "test_doc.pdf") -> None:
        self.doc_name = doc_name

    async def aparse_pdf(self, pdf_path: Path | str) -> list[MultimodalChunk]:
        path = Path(pdf_path)
        bbox = BoundingBox(page=1, x0=0.1, y0=0.1, x1=0.9, y1=0.5)
        return [
            MultimodalChunk(
                chunk_id="chk_txt_01",
                doc_id=f"doc_{path.stem}",
                doc_name=path.name,
                page_number=1,
                modality=ChunkModality.TEXT,
                section_title="1. Introduction",
                content="OmniDoc-Vision hybrid retrieval test content.",
                bounding_box=bbox,
            ),
            MultimodalChunk(
                chunk_id="chk_tbl_01",
                doc_id=f"doc_{path.stem}",
                doc_name=path.name,
                page_number=1,
                modality=ChunkModality.TABLE,
                section_title="2. Latency Benchmarks",
                content="| Concurrency | Latency (ms) |\n|---|---|\n| 100 | 25 |",
                table_html="<table><tr><td>100</td><td>25</td></tr></table>",
                bounding_box=bbox,
            ),
            MultimodalChunk(
                chunk_id="chk_fig_01",
                doc_id=f"doc_{path.stem}",
                doc_name=path.name,
                page_number=1,
                modality=ChunkModality.FIGURE,
                section_title="3. Architecture Diagram",
                content="[FIGURE] System topology overview diagram.",
                image_path="data/crops/chk_fig_01.png",
                bounding_box=bbox,
            ),
        ]


class DummyCaptioner:
    """Mock captioner appending synthetic technical captions to figure chunks."""

    async def caption_chunks(
        self,
        chunks: Sequence[MultimodalChunk],
        *,
        max_concurrency: int = 4,
    ) -> list[MultimodalChunk]:
        del max_concurrency
        result: list[MultimodalChunk] = []
        for c in chunks:
            if c.modality == ChunkModality.FIGURE:
                updated = c.model_copy(
                    update={"content": f"{c.content} Dense technical visual description."}
                )
                result.append(updated)
            else:
                result.append(c)
        return result


class DummyIndexer:
    """Mock indexer recording chunk counts."""

    def __init__(self) -> None:
        self.indexed_chunks: list[MultimodalChunk] = []
        self.qdrant_client = DummyQdrantClient()

    async def index_chunks(self, chunks: Sequence[MultimodalChunk]) -> IndexingStats:
        self.indexed_chunks.extend(chunks)
        return IndexingStats(
            total_chunks=len(chunks),
            indexed_dense=len(chunks),
            indexed_sparse=len(chunks),
            doc_ids=list({c.doc_id for c in chunks}),
        )


class DummyQdrantClient:
    """Mock Qdrant client responding to collection inspection."""

    async def get_collections(self) -> Any:
        return {"collections": [{"name": "omnidoc_vision_chunks"}]}


class DummyWorkflow:
    """Mock LangGraph AdaptiveRAGWorkflow producing verified MultimodalQueryResponse."""

    def __init__(self, force_visual: bool = False) -> None:
        self.force_visual = force_visual

    async def run_query(
        self,
        request: MultimodalQueryRequest | str,
        *,
        top_k: int = 5,
        modality_filter: ChunkModality | str | None = None,
        force_visual_route: bool = False,
    ) -> MultimodalQueryResponse:
        del top_k, modality_filter
        is_visual = self.force_visual
        if isinstance(request, MultimodalQueryRequest) and request.force_visual_route:
            is_visual = True

        citations: list[VisualCitation] = []
        if is_visual:
            citations.append(
                VisualCitation(
                    chunk_id="doc_bench_p03_fig01",
                    doc_name="cloud_database_benchmark_2026.pdf",
                    page_number=3,
                    modality=ChunkModality.FIGURE,
                    section_title="P99 Write Latency",
                    rerank_score=0.95,
                    bounding_box=BoundingBox(page=3, x0=0.1, y0=0.2, x1=0.9, y1=0.7),
                    image_url="/api/v1/crops/doc_bench_p03_fig01.png",
                    content_preview="[FIGURE] P99 Latency vs Concurrent Connections.",
                )
            )
            routing_path = RoutingPath.VISUAL_VLM_PATH
            answer = "بر اساس نمودار صفحه ۳، جهش تاخیر از مرز ۲۵۰ اتصال هم‌زمان رخ می‌دهد."
            images_inspected = 1
        else:
            citations.append(
                VisualCitation(
                    chunk_id="doc_bench_p02_tbl01",
                    doc_name="cloud_database_benchmark_2026.pdf",
                    page_number=2,
                    modality=ChunkModality.TABLE,
                    section_title="Latency Benchmarks",
                    rerank_score=0.88,
                    bounding_box=BoundingBox(page=2, x0=0.1, y0=0.2, x1=0.9, y1=0.5),
                    image_url=None,
                    content_preview="| Concurrency | Latency |\n| 100 | 25 ms |",
                )
            )
            routing_path = RoutingPath.FAST_TEXT_TABLE_PATH
            answer = "طبق جدول صفحه ۲، تاخیر در ۱۰۰ اتصال معادل ۲۵ میلی‌ثانیه است."
            images_inspected = 0

        return MultimodalQueryResponse(
            answer=answer,
            routing_path=routing_path,
            crag_status=CRAGStatus.CONFIDENT,
            rewritten_query=None,
            vlm_images_inspected=images_inspected,
            latency_ms=LatencyBreakdown(
                retrieval_ms=45.2,
                rerank_ms=30.1,
                generation_ms=210.5,
                total_ms=285.8,
            ),
            citations=citations,
        )


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provide isolated directory settings for integration tests."""
    raw_dir = tmp_path / "raw_pdfs"
    crops_dir = tmp_path / "crops"
    bm25_dir = tmp_path / "bm25_index"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        raw_pdfs_dir=raw_dir,
        crops_dir=crops_dir,
        bm25_index_dir=bm25_dir,
        environment="testing",
    )


@pytest.fixture
def app_with_mocks(test_settings: Settings) -> FastAPI:
    """Provide FastAPI app with injected mock dependencies."""
    app = create_app(settings=test_settings)

    dummy_parser = DummyParser()
    dummy_captioner = DummyCaptioner()
    dummy_indexer = DummyIndexer()
    dummy_workflow = DummyWorkflow()

    app.dependency_overrides[get_settings_dep] = lambda: test_settings
    app.dependency_overrides[get_parser_dep] = lambda: dummy_parser
    app.dependency_overrides[get_captioner_dep] = lambda: dummy_captioner
    app.dependency_overrides[get_indexer_dep] = lambda: dummy_indexer
    app.dependency_overrides[get_workflow_dep] = lambda: dummy_workflow

    return app


@pytest.fixture
async def client(app_with_mocks: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Asynchronous HTTP test client using ASGI transport."""
    transport = ASGITransport(app=app_with_mocks)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoints(client: httpx.AsyncClient) -> None:
    """Verify readiness status on both /api/v1/health and /health."""
    for endpoint in ["/api/v1/health", "/health"]:
        response = await client.get(endpoint)
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in ["healthy", "degraded"]
        assert payload["version"] == "0.1.0"
        assert "services" in payload
        assert "qdrant" in payload["services"]
        assert "bm25" in payload["services"]
        assert "redis" in payload["services"]
        assert "vlm" in payload["services"]


@pytest.mark.asyncio
async def test_crops_serving_success(client: httpx.AsyncClient, test_settings: Settings) -> None:
    """Verify valid PNG crops are served with proper content-type."""
    crop_path = test_settings.crops_dir / "sample_figure.png"
    # Write a minimal 1x1 PNG header
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00"
        b"\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    crop_path.write_bytes(png_bytes)

    response = await client.get("/api/v1/crops/sample_figure.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == png_bytes


@pytest.mark.asyncio
async def test_crops_serving_not_found(client: httpx.AsyncClient) -> None:
    """Verify missing crop filenames return HTTP 404."""
    response = await client.get("/api/v1/crops/missing_chart.png")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_crops_serving_path_traversal_rejection(client: httpx.AsyncClient) -> None:
    """Verify strict rejection of path traversal attacks with HTTP 400."""
    malicious_inputs = [
        "../secret.png",
        "..%2Fsecret.png",
        "subfolder/image.png",
        "..\\windows_traversal.png",
    ]
    for filename in malicious_inputs:
        response = await client.get(f"/api/v1/crops/{filename}")
        assert response.status_code in [400, 404]
        if response.status_code == 400:
            detail = response.json()["detail"].lower()
            assert "forbidden" in detail or "invalid" in detail


@pytest.mark.asyncio
async def test_ingest_endpoint_success(client: httpx.AsyncClient, test_settings: Settings) -> None:
    """Verify successful PDF ingestion, chunking, captioning, and response schema."""
    pdf_path = test_settings.raw_pdfs_dir / "benchmark.pdf"
    pdf_path.write_bytes(b"%PDF-1.5 dummy binary content")

    payload = {
        "pdf_path": "benchmark.pdf",
        "generate_vlm_captions": True,
    }
    response = await client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "indexed"
    assert data["doc_name"] == "benchmark.pdf"
    assert data["pages_processed"] == 1
    assert data["chunks_created"] == {
        "text": 1,
        "table": 1,
        "figure": 1,
        "total": 3,
    }
    assert data["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_ingest_endpoint_file_not_found(client: httpx.AsyncClient) -> None:
    """Verify missing PDF returns HTTP 404."""
    payload = {"pdf_path": "non_existent.pdf"}
    response = await client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ingest_endpoint_path_traversal_schema_rejection(
    client: httpx.AsyncClient,
) -> None:
    """Verify path traversal attempts in pdf_path are rejected with HTTP 422."""
    payload = {"pdf_path": "../../etc/shadow.pdf"}
    response = await client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 422
    err_body = response.json()
    assert "error" in err_body
    assert err_body["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_ingest_endpoint_rejects_non_pdf(
    client: httpx.AsyncClient, test_settings: Settings
) -> None:
    """Verify non-PDF documents return HTTP 400."""
    txt_path = test_settings.raw_pdfs_dir / "report.txt"
    txt_path.write_text("plain text", encoding="utf-8")

    payload = {"pdf_path": "report.txt"}
    response = await client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 400
    assert "only pdf" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_endpoint_fast_text_route(client: httpx.AsyncClient) -> None:
    """Verify fast text/table routing path returns valid response and table citations."""
    payload = {
        "query": "میانگین تاخیر در ۱۰۰ اتصال چقدر است؟",
        "top_k": 5,
        "force_visual_route": False,
    }
    response = await client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["routing_path"] == "FAST_TEXT_TABLE_PATH"
    assert data["crag_status"] == "CONFIDENT"
    assert data["vlm_images_inspected"] == 0
    assert len(data["citations"]) == 1
    assert data["citations"][0]["modality"] == "table"
    assert data["citations"][0]["image_url"] is None


@pytest.mark.asyncio
async def test_query_endpoint_visual_route(client: httpx.AsyncClient) -> None:
    """Verify visual VLM routing path includes valid crop URL and image inspections."""
    payload = {
        "query": "بر اساس نمودار تاخیر P99، جهش در چه نقطه‌ای رخ می‌دهد؟",
        "top_k": 5,
        "force_visual_route": True,
    }
    response = await client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["routing_path"] == "VISUAL_VLM_PATH"
    assert data["vlm_images_inspected"] == 1
    assert len(data["citations"]) == 1
    citation = data["citations"][0]
    assert citation["modality"] == "figure"
    assert citation["image_url"] == "/api/v1/crops/doc_bench_p03_fig01.png"
    assert citation["bounding_box"]["page"] == 3


@pytest.mark.asyncio
async def test_query_endpoint_validation_error(client: httpx.AsyncClient) -> None:
    """Verify invalid payloads (empty query, invalid top_k) return HTTP 422."""
    invalid_payload = {
        "query": "",
        "top_k": 50,
    }
    response = await client.post("/api/v1/query", json=invalid_payload)
    assert response.status_code == 422
    err_body = response.json()
    assert err_body["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_ingest_domain_exception_returns_structured_400(
    app_with_mocks: FastAPI,
    client: httpx.AsyncClient,
    test_settings: Settings,
) -> None:
    """Verify PdfValidationError raised inside /ingest propagates to 400 handler."""
    from src.common.exceptions import PdfValidationError

    class FailingParser:
        async def aparse_pdf(self, pdf_path: Path | str) -> list[MultimodalChunk]:
            raise PdfValidationError(f"Corrupted PDF header in {pdf_path}")

    pdf_path = test_settings.raw_pdfs_dir / "corrupted.pdf"
    pdf_path.write_bytes(b"%PDF-broken")
    app_with_mocks.dependency_overrides[get_parser_dep] = lambda: FailingParser()

    response = await client.post("/api/v1/ingest", json={"pdf_path": "corrupted.pdf"})
    assert response.status_code == 400
    err_body = response.json()
    assert err_body["error"]["code"] == "PDF_VALIDATION_ERROR"
    assert "Corrupted PDF header" in err_body["error"]["message"]
