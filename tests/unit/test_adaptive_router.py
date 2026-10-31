"""Unit tests for the LangGraph CRAG Evaluator and Adaptive Multimodal Router (Task 4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.graph.nodes import (
    CRAG_CONFIDENCE_THRESHOLD,
    adaptive_router_edge,
    crag_conditional_edge,
    crag_evaluate_node,
    fast_text_table_generation_node,
    retrieve_node,
    rewrite_query_node,
    select_visual_image_paths,
    visual_vlm_generation_node,
)
from src.graph.state import (
    VisionGraphState,
    build_initial_state,
    chunk_to_visual_citation,
    state_to_query_response,
)
from src.graph.workflow import AdaptiveRAGWorkflow, build_vision_graph
from src.retrieval.dense import ScoredChunk
from src.retrieval.reranker import HybridRetrievalResult
from src.schemas.api import CRAGStatus, MultimodalQueryRequest, RoutingPath
from src.schemas.document import BoundingBox, ChunkModality, MultimodalChunk


def _make_scored_chunk(
    chunk_id: str,
    modality: ChunkModality,
    score: float,
    *,
    page: int = 2,
    doc_name: str = "cloud_database_benchmark_2026.pdf",
    content: str = "Qdrant scalar quantization achieves 14,200 QPS at 6.4 GB RAM.",
    image_path: str | None = None,
    table_html: str | None = None,
) -> ScoredChunk:
    resolved_table_html = table_html
    resolved_image_path = image_path
    if modality is ChunkModality.TABLE and resolved_table_html is None:
        resolved_table_html = "<table><tr><td>Qdrant</td><td>14200</td></tr></table>"
    if modality is ChunkModality.FIGURE and resolved_image_path is None:
        resolved_image_path = f"data/crops/{chunk_id}.png"

    chunk = MultimodalChunk(
        chunk_id=chunk_id,
        doc_id="doc_cloud_db_bench_2026",
        doc_name=doc_name,
        page_number=page,
        modality=modality,
        section_title=f"Section for {chunk_id}",
        content=content,
        table_html=resolved_table_html,
        image_path=resolved_image_path,
        bounding_box=BoundingBox(page=page, x0=0.12, y0=0.20, x1=0.88, y1=0.65),
    )
    return ScoredChunk(
        chunk=chunk,
        score=score,
        dense_score=score,
        sparse_score=score,
        rrf_score=0.032,
        rerank_score=score,
    )


class FakeHybridRetriever:
    """Deterministic fake retriever supporting multi-turn CRAG responses."""

    def __init__(self, responses: Sequence[HybridRetrievalResult]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, int, ChunkModality | str | None]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        modality_filter: ChunkModality | str | None = None,
    ) -> HybridRetrievalResult:
        self.calls.append((query, top_k, modality_filter))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


class FakeVLMService:
    """Deterministic fake VLM service tracking fast vs visual calls and rewrites."""

    def __init__(self, rewritten_text: str = "network topology server cluster diagram") -> None:
        self.rewritten_text = rewritten_text
        self.fast_calls: list[tuple[str, list[ScoredChunk]]] = []
        self.visual_calls: list[tuple[str, list[ScoredChunk], list[str]]] = []
        self.rewrite_calls: list[str] = []

    async def generate_fast_text(
        self,
        query: str,
        chunks: Sequence[MultimodalChunk | ScoredChunk],
    ) -> str:
        scored = [c for c in chunks if isinstance(c, ScoredChunk)]
        self.fast_calls.append((query, scored))
        return f"[FAST_ANSWER] Grounded answer for '{query}' from {len(chunks)} text/table chunks."

    async def generate_visual_reasoning(
        self,
        query: str,
        chunks: Sequence[MultimodalChunk | ScoredChunk],
        image_paths: Sequence[Path | str] | None = None,
    ) -> str:
        scored = [c for c in chunks if isinstance(c, ScoredChunk)]
        paths = [str(p) for p in (image_paths or [])]
        self.visual_calls.append((query, scored, paths))
        return f"[VISUAL_ANSWER] Inspected {len(paths)} chart(s) for '{query}'."

    async def rewrite_query(self, query: str) -> str:
        self.rewrite_calls.append(query)
        return self.rewritten_text


def test_adaptive_router_edge_selects_fast_path_for_text_and_table_only() -> None:
    chunks = [
        _make_scored_chunk("c_text_1", ChunkModality.TEXT, 0.91, page=1),
        _make_scored_chunk("c_table_1", ChunkModality.TABLE, 0.87, page=2),
    ]
    state: VisionGraphState = {
        "query": "What is the scalar quantization throughput?",
        "retrieved_chunks": chunks,
        "force_visual_route": False,
    }

    assert adaptive_router_edge(state) is RoutingPath.FAST_TEXT_TABLE_PATH


def test_adaptive_router_edge_selects_visual_path_when_figure_present() -> None:
    chunks = [
        _make_scored_chunk("c_text_1", ChunkModality.TEXT, 0.89, page=1),
        _make_scored_chunk("c_fig_1", ChunkModality.FIGURE, 0.84, page=3),
    ]
    state: VisionGraphState = {
        "query": "At what concurrency does P99 latency spike?",
        "retrieved_chunks": chunks,
        "force_visual_route": False,
    }

    assert adaptive_router_edge(state) is RoutingPath.VISUAL_VLM_PATH


def test_adaptive_router_edge_respects_force_visual_route() -> None:
    chunks = [_make_scored_chunk("c_text_1", ChunkModality.TEXT, 0.92, page=1)]
    state: VisionGraphState = {
        "query": "Explain the architecture.",
        "retrieved_chunks": chunks,
        "force_visual_route": True,
    }

    assert adaptive_router_edge(state) is RoutingPath.VISUAL_VLM_PATH


def test_select_visual_image_paths_caps_at_two_unique_images() -> None:
    chunks = [
        _make_scored_chunk(
            "fig_1", ChunkModality.FIGURE, 0.95, page=1, image_path="data/crops/fig_1.png"
        ),
        _make_scored_chunk(
            "fig_1_dup", ChunkModality.FIGURE, 0.90, page=1, image_path="data/crops/fig_1.png"
        ),
        _make_scored_chunk(
            "fig_2", ChunkModality.FIGURE, 0.85, page=2, image_path="data/crops/fig_2.png"
        ),
        _make_scored_chunk(
            "fig_3", ChunkModality.FIGURE, 0.80, page=3, image_path="data/crops/fig_3.png"
        ),
    ]

    selected = select_visual_image_paths(chunks)
    assert selected == ["data/crops/fig_1.png", "data/crops/fig_2.png"]


@pytest.mark.asyncio
async def test_workflow_routes_text_and_table_query_to_fast_path() -> None:
    chunks = [
        _make_scored_chunk("tbl_p2", ChunkModality.TABLE, 0.93, page=2),
        _make_scored_chunk("txt_p2", ChunkModality.TEXT, 0.81, page=2),
    ]
    retriever = FakeHybridRetriever(
        [HybridRetrievalResult(chunks=chunks, retrieval_ms=42.5, rerank_ms=28.0)]
    )
    vlm = FakeVLMService()
    workflow = AdaptiveRAGWorkflow(retriever=retriever, vlm_service=vlm)

    request = MultimodalQueryRequest(
        query="در جدول مقایسه دیتابیس‌ها، توان عملیاتی Qdrant چقدر است؟",
        top_k=5,
    )
    response = await workflow.run_query(request)

    assert response.routing_path is RoutingPath.FAST_TEXT_TABLE_PATH
    assert response.crag_status is CRAGStatus.CONFIDENT
    assert response.rewritten_query is None
    assert response.vlm_images_inspected == 0
    assert len(vlm.fast_calls) == 1
    assert len(vlm.visual_calls) == 0
    assert response.latency_ms.retrieval_ms == 42.5
    assert response.latency_ms.rerank_ms == 28.0
    assert response.latency_ms.total_ms >= 70.5
    assert len(response.citations) == 2
    assert response.citations[0].modality is ChunkModality.TABLE
    assert response.citations[0].page_number == 2
    assert response.citations[0].bounding_box.page == 2


@pytest.mark.asyncio
async def test_workflow_routes_figure_query_to_visual_path_and_caps_images_at_two() -> None:
    chunks = [
        _make_scored_chunk(
            "fig_p3_1",
            ChunkModality.FIGURE,
            0.94,
            page=3,
            image_path="data/crops/cloud_p03_fig01.png",
        ),
        _make_scored_chunk(
            "fig_p3_2",
            ChunkModality.FIGURE,
            0.88,
            page=3,
            image_path="data/crops/cloud_p03_fig02.png",
        ),
        _make_scored_chunk(
            "fig_p4_3",
            ChunkModality.FIGURE,
            0.79,
            page=4,
            image_path="data/crops/cloud_p04_fig03.png",
        ),
    ]
    retriever = FakeHybridRetriever(
        [HybridRetrievalResult(chunks=chunks, retrieval_ms=55.0, rerank_ms=35.0)]
    )
    vlm = FakeVLMService()
    workflow = AdaptiveRAGWorkflow(retriever=retriever, vlm_service=vlm)

    response = await workflow.run_query(
        "بر اساس نمودار تاخیر P99 در صفحه ۳، در چه تعداد اتصالی خطا جهش می‌کند؟"
    )

    assert response.routing_path is RoutingPath.VISUAL_VLM_PATH
    assert response.crag_status is CRAGStatus.CONFIDENT
    assert response.vlm_images_inspected == 2
    assert len(vlm.visual_calls) == 1
    assert vlm.visual_calls[0][2] == [
        "data/crops/cloud_p03_fig01.png",
        "data/crops/cloud_p03_fig02.png",
    ]
    assert len(vlm.fast_calls) == 0
    assert response.citations[0].image_url == "/api/v1/crops/cloud_p03_fig01.png"
    assert response.citations[0].bounding_box.x0 == 0.12


@pytest.mark.asyncio
async def test_crag_triggers_single_query_rewrite_when_score_below_threshold() -> None:
    low_score_chunks = [
        _make_scored_chunk("txt_low", ChunkModality.TEXT, 0.24, page=1),
    ]
    high_score_chunks = [
        _make_scored_chunk(
            "arch_p1_fig1",
            ChunkModality.FIGURE,
            0.86,
            page=1,
            image_path="data/crops/arch_p1_fig1.png",
        ),
    ]
    retriever = FakeHybridRetriever(
        [
            HybridRetrievalResult(chunks=low_score_chunks, retrieval_ms=30.0, rerank_ms=20.0),
            HybridRetrievalResult(chunks=high_score_chunks, retrieval_ms=35.0, rerank_ms=25.0),
        ]
    )
    vlm = FakeVLMService(
        rewritten_text="network topology architecture diagram server cluster connection flow"
    )
    workflow = AdaptiveRAGWorkflow(retriever=retriever, vlm_service=vlm)

    response = await workflow.run_query("اون عکس شبکه که سرورها به هم وصلن چی میگه؟")

    assert len(vlm.rewrite_calls) == 1
    assert len(retriever.calls) == 2
    assert (
        retriever.calls[1][0]
        == "network topology architecture diagram server cluster connection flow"
    )
    assert (
        response.rewritten_query
        == "network topology architecture diagram server cluster connection flow"
    )
    assert response.crag_status is CRAGStatus.CONFIDENT
    assert response.routing_path is RoutingPath.VISUAL_VLM_PATH
    assert response.vlm_images_inspected == 1
    assert response.latency_ms.retrieval_ms == 65.0
    assert response.latency_ms.rerank_ms == 45.0


@pytest.mark.asyncio
async def test_crag_does_not_loop_more_than_once_when_scores_remain_low() -> None:
    low_chunks_1 = [_make_scored_chunk("low_1", ChunkModality.TEXT, 0.18, page=1)]
    low_chunks_2 = [_make_scored_chunk("low_2", ChunkModality.TEXT, 0.29, page=2)]

    retriever = FakeHybridRetriever(
        [
            HybridRetrievalResult(chunks=low_chunks_1, retrieval_ms=20.0, rerank_ms=10.0),
            HybridRetrievalResult(chunks=low_chunks_2, retrieval_ms=22.0, rerank_ms=12.0),
        ]
    )
    vlm = FakeVLMService(rewritten_text="refined technical query")
    workflow = AdaptiveRAGWorkflow(retriever=retriever, vlm_service=vlm)

    response = await workflow.run_query("vague question")

    assert len(vlm.rewrite_calls) == 1
    assert len(retriever.calls) == 2
    assert response.crag_status is CRAGStatus.LOW_CONFIDENCE
    assert response.routing_path is RoutingPath.FAST_TEXT_TABLE_PATH
    assert response.rewritten_query == "refined technical query"


@pytest.mark.asyncio
async def test_individual_nodes_and_state_conversion_preserve_visual_attribution() -> None:
    assert CRAG_CONFIDENCE_THRESHOLD == 0.35

    fig_chunk = _make_scored_chunk(
        "doc_p4_fig2",
        ChunkModality.FIGURE,
        0.91,
        page=4,
        image_path="data/crops/doc_p4_fig2.png",
    )
    citation = chunk_to_visual_citation(fig_chunk)
    assert citation.image_url == "/api/v1/crops/doc_p4_fig2.png"
    assert citation.page_number == 4
    assert citation.bounding_box.page == 4

    retriever = FakeHybridRetriever(
        [HybridRetrievalResult(chunks=[fig_chunk], retrieval_ms=15.0, rerank_ms=10.0)]
    )
    vlm = FakeVLMService()

    state = build_initial_state("Test query", top_k=3)
    state.update(await retrieve_node(state, retriever))
    state.update(await crag_evaluate_node(state))
    assert crag_conditional_edge(state) == RoutingPath.VISUAL_VLM_PATH

    state.update(await visual_vlm_generation_node(state, vlm))
    response = state_to_query_response(state)
    assert response.routing_path is RoutingPath.VISUAL_VLM_PATH
    assert response.vlm_images_inspected == 1

    # Also exercise rewrite_query_node and fast_text_table_generation_node directly
    rewrite_update = await rewrite_query_node(state, vlm)
    assert rewrite_update["rewrite_count"] == 1
    fast_update = await fast_text_table_generation_node(state, vlm)
    assert fast_update["routing_path"] is RoutingPath.FAST_TEXT_TABLE_PATH

    compiled = build_vision_graph(retriever=retriever, vlm_service=vlm)
    final_state = await compiled.ainvoke(build_initial_state("Direct graph invocation"))
    assert final_state["routing_path"] is RoutingPath.VISUAL_VLM_PATH


@pytest.mark.asyncio
async def test_workflow_ainvoke_and_close_lifecycle() -> None:
    closed = False

    class ClosableRetriever(FakeHybridRetriever):
        async def close(self) -> None:
            nonlocal closed
            closed = True

    chunk = _make_scored_chunk("txt_p1", ChunkModality.TEXT, 0.85, page=1)
    retriever = ClosableRetriever(
        [HybridRetrievalResult(chunks=[chunk], retrieval_ms=12.0, rerank_ms=8.0)]
    )
    vlm = FakeVLMService()
    workflow = AdaptiveRAGWorkflow(retriever=retriever, vlm_service=vlm)

    state = await workflow.ainvoke(build_initial_state("Test lifecycle"))
    assert state["routing_path"] is RoutingPath.FAST_TEXT_TABLE_PATH
    assert workflow.graph is not None

    await workflow.close()
    assert closed is True

