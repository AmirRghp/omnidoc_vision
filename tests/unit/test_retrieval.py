"""Unit tests for parallel dense + sparse retrieval, RRF fusion, and BGE reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient

from src.common.config import Settings
from src.common.exceptions import RetrievalError
from src.ingestion.indexer import HybridIndexer, MockEmbedder
from src.retrieval import (
    BGEReranker,
    DenseRetriever,
    HybridRetriever,
    MockReranker,
    ScoredChunk,
    SparseRetriever,
    reciprocal_rank_fusion,
)
from src.schemas.document import BoundingBox, ChunkModality, CoordOrigin, MultimodalChunk


def _make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc_cloud",
    doc_name: str = "cloud_database_benchmark_2026.pdf",
    page_number: int = 1,
    modality: ChunkModality = ChunkModality.TEXT,
    section_title: str = "Overview",
    content: str = "Sample technical content",
) -> MultimodalChunk:
    bbox = BoundingBox(
        page=page_number,
        x0=0.1,
        y0=0.2,
        x1=0.85,
        y1=0.65,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    table_html = (
        "<table><tr><th>Engine</th><th>Throughput</th></tr>"
        "<tr><td>Aurora</td><td>18400</td></tr></table>"
        if modality is ChunkModality.TABLE
        else None
    )
    image_path = (
        f"data/crops/{chunk_id}.png" if modality is ChunkModality.FIGURE else None
    )
    return MultimodalChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        page_number=page_number,
        modality=modality,
        section_title=section_title,
        content=content,
        table_html=table_html,
        image_path=image_path,
        bounding_box=bbox,
    )


@pytest.fixture
def multimodal_corpus() -> list[MultimodalChunk]:
    """Build a diverse 8-chunk multimodal corpus covering English and Persian queries."""
    return [
        _make_chunk(
            "doc_cloud_p01_txt01",
            page_number=1,
            modality=ChunkModality.TEXT,
            section_title="1. Executive Summary",
            content=(
                "Aurora PostgreSQL sustains 18,400 ops/sec at 64 threads with warm buffer cache."
            ),
        ),
        _make_chunk(
            "doc_cloud_p01_txt02",
            page_number=1,
            modality=ChunkModality.TEXT,
            section_title="1.2 Persian Summary",
            content=(
                "در جدول مقایسه دیتابیس‌ها، توان عملیاتی Aurora در حالت کش فعال ۱۸۴۰۰ است."
            ),
        ),
        _make_chunk(
            "doc_cloud_p02_tbl01",
            page_number=2,
            modality=ChunkModality.TABLE,
            section_title="2. Throughput and Memory Profile",
            content=(
                "| Engine | Instance | vCPU | Mem (GB) | Cache On | Cache Off |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                "| Aurora PostgreSQL | db.r7g.4xlarge | 16 | 128 | 18,400 | 11,200 |"
            ),
        ),
        _make_chunk(
            "doc_cloud_p03_fig01",
            page_number=3,
            modality=ChunkModality.FIGURE,
            section_title="3. P99 Latency Under Concurrency",
            content=(
                "[FIGURE CAPTION & TECHNICAL SUMMARY] P99 write latency spikes to 620 ms at "
                "250 concurrent connections for Standard RDS while Aurora stays at 31 ms."
            ),
        ),
        _make_chunk(
            "doc_cloud_p04_fig02",
            page_number=4,
            modality=ChunkModality.FIGURE,
            section_title="4. Thread Scaling",
            content=(
                "[FIGURE CAPTION & TECHNICAL SUMMARY] Read throughput vs thread count from "
                "8 to 128 threads across Aurora, Cloud SQL, and Standard RDS."
            ),
        ),
        _make_chunk(
            "doc_cloud_p05_tbl02",
            page_number=5,
            modality=ChunkModality.TABLE,
            section_title="5. Pooling Configuration",
            content=(
                "| Engine | Pool Mode | Max Connections | Cost |\n"
                "| --- | --- | --- | --- |\n"
                "| Aurora PostgreSQL | PgBouncer transaction | 5,000 | $1,680 |"
            ),
        ),
        _make_chunk(
            "doc_cache_p02_tbl01",
            doc_id="doc_cache",
            doc_name="distributed_cache_architecture.pdf",
            page_number=2,
            modality=ChunkModality.TABLE,
            section_title="2. Eviction Policy Comparison",
            content=(
                "| Policy | 64 GB Hit Ratio | 256 GB Hit Ratio | Latency (us) |\n"
                "| --- | --- | --- | --- |\n"
                "| LFU | 91.2% | 96.3% | 430 |"
            ),
        ),
        _make_chunk(
            "doc_gpu_p03_fig01",
            doc_id="doc_gpu",
            doc_name="gpu_inference_capacity_report.pdf",
            page_number=3,
            modality=ChunkModality.FIGURE,
            section_title="3. AWQ vs FP16 VRAM Footprint",
            content=(
                "[FIGURE CAPTION & TECHNICAL SUMMARY] Bar chart comparing Qwen2.5-VL 4-bit AWQ "
                "(9.4 GB at batch 8) vs FP16 OOM above 16 GB VRAM ceiling."
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_dense_retriever_search_and_modality_filter(
    multimodal_corpus: list[MultimodalChunk],
    tmp_path: Path,
) -> None:
    """DenseRetriever returns top_k chunks and respects modality_filter and doc_name."""
    qdrant = AsyncQdrantClient(location=":memory:")
    settings = Settings(
        environment="testing",
        qdrant_collection="test_dense_retrieval",
        bm25_index_dir=tmp_path / "bm25",
    )
    embedder = MockEmbedder()
    indexer = HybridIndexer(settings=settings, qdrant_client=qdrant, embedder=embedder)
    await indexer.index_chunks(multimodal_corpus)

    dense = DenseRetriever(settings=settings, qdrant_client=qdrant, embedder=embedder)

    all_hits = await dense.search("Aurora PostgreSQL throughput", top_k=5)
    assert len(all_hits) == 5
    assert all(isinstance(hit, ScoredChunk) for hit in all_hits)
    assert all(hit.dense_score is not None for hit in all_hits)

    # Filter strictly by FIGURE modality
    figure_hits = await dense.search(
        "P99 latency chart",
        top_k=10,
        modality_filter="figure",
    )
    assert len(figure_hits) == 3
    assert {h.chunk.modality for h in figure_hits} == {ChunkModality.FIGURE}
    assert all(h.chunk.image_path is not None for h in figure_hits)

    # Filter by TABLE modality and doc_name
    table_hits = await dense.search(
        "eviction hit ratio",
        top_k=10,
        modality_filter=ChunkModality.TABLE,
        doc_name="distributed_cache_architecture.pdf",
    )
    assert len(table_hits) == 1
    assert table_hits[0].chunk.chunk_id == "doc_cache_p02_tbl01"
    assert table_hits[0].chunk.table_html is not None

    # Invalid modality filter raises RetrievalError
    with pytest.raises(RetrievalError, match="Unsupported modality_filter"):
        await dense.search("query", modality_filter="invalid_modality")

    await dense.close()


@pytest.mark.asyncio
async def test_sparse_retriever_bilingual_and_modality_filter(
    multimodal_corpus: list[MultimodalChunk],
    tmp_path: Path,
) -> None:
    """SparseRetriever matches English and Persian queries and preserves full payload."""
    qdrant = AsyncQdrantClient(location=":memory:")
    bm25_dir = tmp_path / "bm25"
    settings = Settings(
        environment="testing",
        qdrant_collection="test_sparse_retrieval",
        bm25_index_dir=bm25_dir,
    )
    indexer = HybridIndexer(settings=settings, qdrant_client=qdrant, embedder=MockEmbedder())
    await indexer.index_chunks(multimodal_corpus)

    sparse = SparseRetriever(settings=settings, bm25_dir=bm25_dir)

    # English metric lookup
    en_hits = await sparse.search("620 ms 250 concurrent connections", top_k=3)
    assert len(en_hits) >= 1
    assert en_hits[0].chunk.chunk_id == "doc_cloud_p03_fig01"
    assert en_hits[0].chunk.bounding_box.page == 3
    assert en_hits[0].sparse_score is not None

    # Persian query with Arabic/Persian character variants and Eastern digits
    fa_hits = await sparse.search("توان عملياتي دیتابیس در حالت كش فعال ۱۸۴۰۰", top_k=3)
    assert len(fa_hits) >= 1
    assert fa_hits[0].chunk.chunk_id == "doc_cloud_p01_txt02"

    # Modality filter on sparse search
    tbl_hits = await sparse.search("PgBouncer 5,000", top_k=5, modality_filter="table")
    assert len(tbl_hits) >= 1
    assert all(h.chunk.modality is ChunkModality.TABLE for h in tbl_hits)
    assert tbl_hits[0].chunk.chunk_id == "doc_cloud_p05_tbl02"

    await indexer.close()


def test_reciprocal_rank_fusion_math_and_deduplication(
    multimodal_corpus: list[MultimodalChunk],
) -> None:
    """RRF (k=60) computes exact 1/(k+rank) sums, deduplicates, and caps at top_n."""
    c1, c2, c3 = multimodal_corpus[0], multimodal_corpus[1], multimodal_corpus[2]

    dense_list = [
        ScoredChunk(chunk=c1, score=0.95, dense_score=0.95),  # rank 1 -> 1/61
        ScoredChunk(chunk=c2, score=0.80, dense_score=0.80),  # rank 2 -> 1/62
    ]
    sparse_list = [
        ScoredChunk(chunk=c2, score=12.4, sparse_score=12.4),  # rank 1 -> 1/61
        ScoredChunk(chunk=c3, score=8.1, sparse_score=8.1),  # rank 2 -> 1/62
    ]

    fused = reciprocal_rank_fusion([dense_list, sparse_list], k=60, top_n=15)
    assert len(fused) == 3

    # c2 appears in both lists (rank 2 + rank 1) -> highest RRF score
    assert fused[0].chunk.chunk_id == c2.chunk_id
    expected_c2_rrf = (1.0 / 62.0) + (1.0 / 61.0)
    assert fused[0].rrf_score == pytest.approx(expected_c2_rrf, rel=1e-7)
    assert fused[0].dense_score == 0.80
    assert fused[0].sparse_score == 12.4

    # c1 has rank 1 in dense (1/61), c3 has rank 2 in sparse (1/62)
    assert fused[1].chunk.chunk_id == c1.chunk_id
    assert fused[1].rrf_score == pytest.approx(1.0 / 61.0, rel=1e-7)
    assert fused[2].chunk.chunk_id == c3.chunk_id
    assert fused[2].rrf_score == pytest.approx(1.0 / 62.0, rel=1e-7)

    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([dense_list], k=0)


@pytest.mark.asyncio
async def test_bge_reranker_normalizes_scores_and_sorts(
    multimodal_corpus: list[MultimodalChunk],
) -> None:
    """BGEReranker normalizes raw cross-encoder scores into [0, 1] and sorts Top-K."""

    class _FakeFlagReranker:
        def compute_score(
            self,
            pairs: list[list[str]],
            normalize: bool = True,
        ) -> list[float]:
            # Return highest score for the figure chunk about 250 concurrent connections
            scores: list[float] = []
            for _query, text in pairs:
                if "250 concurrent connections" in text:
                    scores.append(2.5)  # raw logit > 1 to verify sigmoid clamping into [0, 1]
                elif "18,400" in text:
                    scores.append(0.72)
                else:
                    scores.append(-1.5)
            return scores

    reranker = BGEReranker()
    reranker._model = _FakeFlagReranker()

    candidates = [
        ScoredChunk(chunk=c, score=0.03, rrf_score=0.03) for c in multimodal_corpus[:4]
    ]
    top_chunks = await reranker.rerank(
        "At what concurrency does P99 write latency spike to 620 ms?",
        candidates,
        top_k=2,
    )

    assert len(top_chunks) == 2
    assert top_chunks[0].chunk.chunk_id == "doc_cloud_p03_fig01"
    assert top_chunks[0].rerank_score is not None
    assert 0.0 <= top_chunks[0].rerank_score <= 1.0
    assert top_chunks[0].rerank_score > top_chunks[1].rerank_score


@pytest.mark.asyncio
async def test_hybrid_retriever_end_to_end_pipeline_and_fallback(
    multimodal_corpus: list[MultimodalChunk],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HybridRetriever runs parallel dense+sparse, RRF (k=60), and reranks to Top-5."""
    qdrant = AsyncQdrantClient(location=":memory:")
    bm25_dir = tmp_path / "bm25"
    settings = Settings(
        environment="testing",
        qdrant_collection="test_hybrid_e2e",
        bm25_index_dir=bm25_dir,
    )
    embedder = MockEmbedder()
    indexer = HybridIndexer(settings=settings, qdrant_client=qdrant, embedder=embedder)
    await indexer.index_chunks(multimodal_corpus)

    dense = DenseRetriever(settings=settings, qdrant_client=qdrant, embedder=embedder)
    sparse = SparseRetriever(settings=settings, bm25_dir=bm25_dir)

    # Also test runtime fallback when BGEReranker fails to load model weights
    broken_reranker = BGEReranker(model_name="nonexistent/reranker")

    def _fail_load() -> Any:
        raise RetrievalError("Missing reranker weights")

    monkeypatch.setattr(broken_reranker, "_load_model", _fail_load)

    hybrid = HybridRetriever(
        settings=settings,
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=broken_reranker,
    )

    result = await hybrid.retrieve(
        "P99 write latency 620 ms at 250 concurrent connections",
        top_k=5,
    )
    assert len(result.chunks) == 5
    assert result.retrieval_ms >= 0.0
    assert result.rerank_ms >= 0.0
    assert isinstance(hybrid.reranker, MockReranker)

    # Top chunk should be the figure chunk matching the query terms
    top_hit = result.chunks[0]
    assert top_hit.chunk.chunk_id == "doc_cloud_p03_fig01"
    assert top_hit.chunk.modality is ChunkModality.FIGURE
    assert top_hit.chunk.image_path == "data/crops/doc_cloud_p03_fig01.png"
    assert top_hit.rerank_score is not None
    assert 0.0 <= top_hit.rerank_score <= 1.0

    # Test modality_filter="table" on HybridRetriever.search
    table_results = await hybrid.search(
        "LFU 96.3% 256 GB Hit Ratio",
        top_k=3,
        modality_filter="table",
    )
    assert len(table_results) == 3
    assert all(item.chunk.modality is ChunkModality.TABLE for item in table_results)
    assert table_results[0].chunk.chunk_id == "doc_cache_p02_tbl01"

    await hybrid.close()
