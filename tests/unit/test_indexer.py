"""Unit tests for the Unified Hybrid Indexer (Qdrant + BM25s)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient

from src.common.config import Settings
from src.common.exceptions import IndexingError
from src.ingestion.indexer import (
    DENSE_VECTOR_DIM,
    BGEM3Embedder,
    HybridIndexer,
    MockEmbedder,
    normalize_bilingual_text,
)
from src.schemas.document import BoundingBox, ChunkModality, CoordOrigin, MultimodalChunk


@pytest.fixture
def sample_chunks() -> list[MultimodalChunk]:
    """Create a sample batch of text, table, and figure MultimodalChunks."""
    bbox1 = BoundingBox(page=1, x0=0.1, y0=0.1, x1=0.9, y1=0.2, coord_origin=CoordOrigin.TOPLEFT)
    chunk1 = MultimodalChunk(
        chunk_id="doc_bench_p01_txt01",
        doc_id="doc_bench",
        doc_name="benchmark.pdf",
        page_number=1,
        modality=ChunkModality.TEXT,
        section_title="Overview",
        content="Distributed database throughput evaluation under high concurrency.",
        bounding_box=bbox1,
    )

    bbox2 = BoundingBox(page=2, x0=0.1, y0=0.2, x1=0.9, y1=0.6, coord_origin=CoordOrigin.TOPLEFT)
    chunk2 = MultimodalChunk(
        chunk_id="doc_bench_p02_tbl01",
        doc_id="doc_bench",
        doc_name="benchmark.pdf",
        page_number=2,
        modality=ChunkModality.TABLE,
        section_title="Benchmark Results",
        content="| Engine | QPS | P99 (ms) |\n| Aurora | 45000 | 38 |",
        table_html=(
            "<table><tr><th>Engine</th><th>QPS</th></tr>"
            "<tr><td>Aurora</td><td>45000</td></tr></table>"
        ),
        bounding_box=bbox2,
    )

    bbox3 = BoundingBox(page=3, x0=0.1, y0=0.3, x1=0.9, y1=0.8, coord_origin=CoordOrigin.TOPLEFT)
    chunk3 = MultimodalChunk(
        chunk_id="doc_bench_p03_fig01",
        doc_id="doc_bench",
        doc_name="benchmark.pdf",
        page_number=3,
        modality=ChunkModality.FIGURE,
        section_title="Latency Curve",
        content="[FIGURE CAPTION & TECHNICAL SUMMARY] P99 write latency vs connections.",
        image_path="data/crops/doc_bench_p03_fig01.png",
        bounding_box=bbox3,
    )

    bbox4 = BoundingBox(page=1, x0=0.1, y0=0.1, x1=0.9, y1=0.3, coord_origin=CoordOrigin.TOPLEFT)
    chunk4 = MultimodalChunk(
        chunk_id="doc_other_p01_txt01",
        doc_id="doc_other",
        doc_name="other.pdf",
        page_number=1,
        modality=ChunkModality.TEXT,
        section_title="Cache Topology",
        content="Distributed Redis caching layer with cluster replication.",
        bounding_box=bbox4,
    )

    return [chunk1, chunk2, chunk3, chunk4]


@pytest.fixture
def in_memory_qdrant() -> AsyncQdrantClient:
    """Create an isolated in-memory AsyncQdrantClient."""
    return AsyncQdrantClient(location=":memory:")


@pytest.mark.asyncio
async def test_mock_embedder_properties() -> None:
    """MockEmbedder produces deterministic, L2-normalized 1024-dim vectors."""
    embedder = MockEmbedder(dimension=DENSE_VECTOR_DIM)
    assert embedder.dimension == DENSE_VECTOR_DIM

    texts = [
        "Distributed database throughput evaluation.",
        "Distributed database throughput evaluation.",
        "Different unrelated sentence.",
    ]
    vectors = await embedder.embed_texts(texts)
    assert len(vectors) == 3
    assert len(vectors[0]) == DENSE_VECTOR_DIM

    # L2-normalization check
    norm0 = float(np.linalg.norm(vectors[0]))
    assert pytest.approx(norm0, rel=1e-5) == 1.0

    # Determinism check: identical text -> identical vector
    assert vectors[0] == vectors[1]

    # Distinct texts -> distinct vectors
    assert vectors[0] != vectors[2]

    # Empty sequence
    assert await embedder.embed_texts([]) == []


@pytest.mark.asyncio
async def test_ensure_collection_creates_and_is_idempotent(
    in_memory_qdrant: AsyncQdrantClient,
    tmp_path: Path,
) -> None:
    """ensure_collection creates the Qdrant collection and payload indexes idempotently."""
    settings = Settings(
        environment="testing",
        qdrant_collection="test_collection_idempotent",
        bm25_index_dir=tmp_path / "bm25",
    )
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=in_memory_qdrant,
        embedder=MockEmbedder(),
    )

    await indexer.ensure_collection()
    assert await in_memory_qdrant.collection_exists("test_collection_idempotent")

    # Calling again must not fail or raise
    await indexer.ensure_collection()
    await indexer.close()


@pytest.mark.asyncio
async def test_index_chunks_stores_payload_and_sparse_index(
    in_memory_qdrant: AsyncQdrantClient,
    sample_chunks: list[MultimodalChunk],
    tmp_path: Path,
) -> None:
    """index_chunks stores dense vectors with rich payloads in Qdrant and indexes BM25s."""
    bm25_dir = tmp_path / "bm25"
    settings = Settings(
        environment="testing",
        qdrant_collection="test_multimodal_hybrid",
        bm25_index_dir=bm25_dir,
    )
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=in_memory_qdrant,
        embedder=MockEmbedder(),
    )

    stats = await indexer.index_chunks(sample_chunks)
    assert stats.total_chunks == 4
    assert stats.indexed_dense == 4
    assert stats.indexed_sparse == 4
    assert sorted(stats.doc_ids) == ["doc_bench", "doc_other"]

    # Verify points in Qdrant
    point_id_txt = str(uuid.uuid5(uuid.NAMESPACE_DNS, sample_chunks[0].chunk_id))
    retrieved = await in_memory_qdrant.retrieve(
        collection_name="test_multimodal_hybrid",
        ids=[point_id_txt],
        with_payload=True,
    )
    assert len(retrieved) == 1
    payload = retrieved[0].payload
    assert payload is not None
    assert payload["chunk_id"] == "doc_bench_p01_txt01"
    assert payload["modality"] == "text"
    assert payload["section_title"] == "Overview"
    assert payload["bounding_box"]["page"] == 1

    # Verify table chunk payload
    point_id_tbl = str(uuid.uuid5(uuid.NAMESPACE_DNS, sample_chunks[1].chunk_id))
    retrieved_tbl = await in_memory_qdrant.retrieve(
        collection_name="test_multimodal_hybrid",
        ids=[point_id_tbl],
        with_payload=True,
    )
    assert retrieved_tbl[0].payload is not None
    assert retrieved_tbl[0].payload["modality"] == "table"
    assert "<table>" in retrieved_tbl[0].payload["table_html"]

    # Verify figure chunk payload
    point_id_fig = str(uuid.uuid5(uuid.NAMESPACE_DNS, sample_chunks[2].chunk_id))
    retrieved_fig = await in_memory_qdrant.retrieve(
        collection_name="test_multimodal_hybrid",
        ids=[point_id_fig],
        with_payload=True,
    )
    assert retrieved_fig[0].payload is not None
    assert retrieved_fig[0].payload["modality"] == "figure"
    assert retrieved_fig[0].payload["image_path"] == "data/crops/doc_bench_p03_fig01.png"

    # Verify BM25s directory and full MultimodalChunk payload in corpus_metadata.json
    assert bm25_dir.exists()
    metadata_file = bm25_dir / "corpus_metadata.json"
    assert metadata_file.exists()
    bm25_records = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert len(bm25_records) == 4
    records_by_id = {rec["chunk_id"]: rec for rec in bm25_records}
    assert records_by_id["doc_bench_p02_tbl01"]["table_html"] is not None
    assert (
        records_by_id["doc_bench_p03_fig01"]["image_path"]
        == "data/crops/doc_bench_p03_fig01.png"
    )
    assert records_by_id["doc_bench_p03_fig01"]["bounding_box"]["page"] == 3
    assert records_by_id["doc_bench_p01_txt01"]["section_title"] == "Overview"

    await indexer.close()


@pytest.mark.asyncio
async def test_index_chunks_empty(
    in_memory_qdrant: AsyncQdrantClient,
    tmp_path: Path,
) -> None:
    """Empty chunks sequence returns 0 stats safely."""
    settings = Settings(
        environment="testing",
        qdrant_collection="test_empty",
        bm25_index_dir=tmp_path / "bm25",
    )
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=in_memory_qdrant,
        embedder=MockEmbedder(),
    )
    stats = await indexer.index_chunks([])
    assert stats.total_chunks == 0
    assert stats.indexed_dense == 0
    await indexer.close()


@pytest.mark.asyncio
async def test_delete_document_removes_from_qdrant_and_bm25(
    in_memory_qdrant: AsyncQdrantClient,
    sample_chunks: list[MultimodalChunk],
    tmp_path: Path,
) -> None:
    """delete_document removes all chunks for a doc_id leaving other documents intact."""
    bm25_dir = tmp_path / "bm25"
    settings = Settings(
        environment="testing",
        qdrant_collection="test_delete_doc",
        bm25_index_dir=bm25_dir,
    )
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=in_memory_qdrant,
        embedder=MockEmbedder(),
    )

    await indexer.index_chunks(sample_chunks)

    # Delete doc_bench (has 3 chunks)
    deleted = await indexer.delete_document("doc_bench")
    assert deleted == 3

    # doc_other point should still exist
    point_id_other = str(uuid.uuid5(uuid.NAMESPACE_DNS, sample_chunks[3].chunk_id))
    retrieved_other = await in_memory_qdrant.retrieve(
        collection_name="test_delete_doc",
        ids=[point_id_other],
        with_payload=True,
    )
    assert len(retrieved_other) == 1
    assert retrieved_other[0].payload is not None
    assert retrieved_other[0].payload["doc_id"] == "doc_other"

    # doc_bench point should not exist
    point_id_bench = str(uuid.uuid5(uuid.NAMESPACE_DNS, sample_chunks[0].chunk_id))
    retrieved_bench = await in_memory_qdrant.retrieve(
        collection_name="test_delete_doc",
        ids=[point_id_bench],
        with_payload=True,
    )
    assert len(retrieved_bench) == 0

    await indexer.close()


@pytest.mark.asyncio
async def test_index_chunks_qdrant_error_raises_indexing_error(
    tmp_path: Path,
    sample_chunks: list[MultimodalChunk],
) -> None:
    """Failures during Qdrant upsert are wrapped in IndexingError."""
    closed_client = AsyncQdrantClient(location=":memory:")
    await closed_client.close()

    settings = Settings(
        environment="testing",
        qdrant_collection="test_error",
        bm25_index_dir=tmp_path / "bm25",
    )
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=closed_client,
        embedder=MockEmbedder(),
    )

    with pytest.raises(IndexingError):
        await indexer.index_chunks(sample_chunks)


def test_normalize_bilingual_text_handles_persian_variants_and_digits() -> None:
    """Persian Yeh/Kaf, ZWNJ, and Eastern digits normalize consistently for BM25."""
    raw = "كلاس\u200cهای دیتابیس در ۲۵۰ اتصال و تاخیر ۱۸.۲ms"
    normalized = normalize_bilingual_text(raw)
    assert "کلاس های" in normalized
    assert "250" in normalized
    assert "18.2ms" in normalized


@pytest.mark.asyncio
async def test_bge_m3_embedder_runtime_failure_falls_back_to_mock_embedder(
    in_memory_qdrant: AsyncQdrantClient,
    sample_chunks: list[MultimodalChunk],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If BGEM3Embedder fails to load at runtime, HybridIndexer falls back to MockEmbedder."""
    broken_bge = BGEM3Embedder(model_name="nonexistent/bge-m3", device="cpu")

    def _fail_load() -> None:
        raise IndexingError("Model weights missing")

    monkeypatch.setattr(broken_bge, "_load_model", _fail_load)
    settings = Settings(
        environment="development",
        qdrant_collection="test_bge_fallback",
        bm25_index_dir=tmp_path / "bm25",
    )
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=in_memory_qdrant,
        embedder=broken_bge,
    )

    stats = await indexer.index_chunks(sample_chunks[:1])
    assert stats.indexed_dense == 1
    assert isinstance(indexer.embedder, MockEmbedder)
    await indexer.close()

