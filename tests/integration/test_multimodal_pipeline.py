"""End-to-end integration tests for the OmniDoc-Vision multimodal pipeline.

Covers PDF generation -> DoclingParser extraction -> Approach A VisualCaptioner ->
Hybrid Qdrant + BM25s Indexing -> RRF + Reranking -> LangGraph AdaptiveRAGWorkflow
(Fast Text/Table Path and Visual VLM Path) and RedisCacheService.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from scripts.benchmark.cloud import build_cloud_database_pdf
from src.common.config import Settings
from src.graph.workflow import AdaptiveRAGWorkflow
from src.ingestion.docling_parser import DoclingParser
from src.ingestion.indexer import DENSE_VECTOR_DIM, HybridIndexer, MockEmbedder
from src.ingestion.visual_captioner import VisualCaptioner
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import HybridRetriever, MockReranker
from src.retrieval.sparse import SparseRetriever
from src.schemas.api import CRAGStatus, MultimodalQueryRequest, RoutingPath
from src.schemas.document import ChunkModality
from src.services.cache_service import RedisCacheService, build_query_cache_key
from src.services.vlm_service import VLMService


@pytest.mark.asyncio
async def test_end_to_end_multimodal_ingestion_retrieval_and_adaptive_routing(
    tmp_path: Path,
) -> None:
    """Verify full pipeline from generated PDF to grounded Fast and Visual responses."""
    raw_dir = tmp_path / "raw_pdfs"
    crops_dir = tmp_path / "crops"
    bm25_dir = tmp_path / "bm25_index"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = raw_dir / "cloud_database_benchmark_2026.pdf"
    build_cloud_database_pdf(pdf_path)
    assert pdf_path.exists()

    settings = Settings(
        environment="testing",
        raw_pdfs_dir=raw_dir,
        crops_dir=crops_dir,
        bm25_index_dir=bm25_dir,
        qdrant_collection="test_e2e_multimodal_pipeline",
    )

    # 1. Structural PDF Parsing & Figure Cropping
    parser = DoclingParser(settings=settings, crops_dir=crops_dir, use_docling=False)
    raw_chunks = await parser.aparse_pdf(pdf_path)
    assert len(raw_chunks) >= 5
    modalities = {c.modality for c in raw_chunks}
    assert ChunkModality.TEXT in modalities
    assert ChunkModality.TABLE in modalities
    assert ChunkModality.FIGURE in modalities

    # 2. Approach A Visual Captioning (offline fallback)
    captioner = VisualCaptioner(settings=settings, offline_mode=True)
    captioned_chunks = await captioner.caption_chunks(raw_chunks)
    figure_chunks = [c for c in captioned_chunks if c.modality is ChunkModality.FIGURE]
    assert len(figure_chunks) >= 1
    assert all("[FIGURE CAPTION & TECHNICAL SUMMARY]" in c.content for c in figure_chunks)

    # 3. Unified Qdrant + BM25s Hybrid Indexing
    qdrant_client = AsyncQdrantClient(location=":memory:")
    embedder = MockEmbedder(dimension=DENSE_VECTOR_DIM)
    indexer = HybridIndexer(
        settings=settings,
        qdrant_client=qdrant_client,
        embedder=embedder,
    )
    stats = await indexer.index_chunks(captioned_chunks)
    assert stats.total_chunks == len(captioned_chunks)
    assert stats.indexed_dense == len(captioned_chunks)
    assert stats.indexed_sparse == len(captioned_chunks)

    # 4. Hybrid Retriever + Adaptive RAG Workflow
    dense_retriever = DenseRetriever(
        settings=settings,
        qdrant_client=qdrant_client,
        embedder=embedder,
    )
    sparse_retriever = SparseRetriever(settings=settings, bm25_dir=bm25_dir)
    hybrid_retriever = HybridRetriever(
        settings=settings,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=MockReranker(),
    )
    vlm_service = VLMService(settings=settings, offline_mode=True)
    workflow = AdaptiveRAGWorkflow(
        retriever=hybrid_retriever,
        vlm_service=vlm_service,
        settings=settings,
    )

    try:
        # 5a. Query filtered to TABLE modality -> must route to FAST_TEXT_TABLE_PATH
        fast_req = MultimodalQueryRequest(
            query="Aurora PostgreSQL db.r7g.4xlarge throughput Cache On",
            top_k=3,
            modality_filter=ChunkModality.TABLE,
            force_visual_route=False,
        )
        fast_resp = await workflow.run_query(fast_req)
        assert fast_resp.routing_path is RoutingPath.FAST_TEXT_TABLE_PATH
        assert fast_resp.vlm_images_inspected == 0
        assert fast_resp.crag_status in (CRAGStatus.CONFIDENT, CRAGStatus.LOW_CONFIDENCE)
        assert len(fast_resp.citations) >= 1
        assert all(c.modality is ChunkModality.TABLE for c in fast_resp.citations)

        # 5b. Query retrieving FIGURE chunks -> must route to VISUAL_VLM_PATH
        visual_req = MultimodalQueryRequest(
            query="P99 write latency concurrent connections chart figure",
            top_k=5,
            modality_filter=ChunkModality.FIGURE,
            force_visual_route=False,
        )
        visual_resp = await workflow.run_query(visual_req)
        assert visual_resp.routing_path is RoutingPath.VISUAL_VLM_PATH
        assert 1 <= visual_resp.vlm_images_inspected <= 2
        assert len(visual_resp.citations) >= 1
        assert visual_resp.citations[0].modality is ChunkModality.FIGURE
        assert visual_resp.citations[0].image_url is not None
        assert visual_resp.citations[0].image_url.startswith("/api/v1/crops/")

        # 6. Verify RedisCacheService round-trip and invalidation
        cache = RedisCacheService(settings=settings)
        cache_key = build_query_cache_key(
            visual_req.query,
            visual_req.modality_filter,
            force_visual_route=visual_req.force_visual_route,
            top_k=visual_req.top_k,
        )
        assert await cache.get(cache_key) is None
        await cache.set(cache_key, visual_resp)
        cached_hit = await cache.get(cache_key)
        assert cached_hit is not None
        assert cached_hit.routing_path is RoutingPath.VISUAL_VLM_PATH
        assert cached_hit.citations[0].chunk_id == visual_resp.citations[0].chunk_id

        await cache.invalidate_all()
        assert await cache.get(cache_key) is None
        await cache.close()
    finally:
        await workflow.close()
        await indexer.close()
