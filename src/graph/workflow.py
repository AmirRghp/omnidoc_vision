"""Compiled LangGraph state machine for CRAG evaluation and adaptive multimodal routing."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Protocol, cast

from src.common.config import Settings, get_settings
from src.graph.nodes import (
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EVALUATE_NODE,
    FAST_TEXT_TABLE_GENERATION_NODE,
    MAX_REWRITE_COUNT,
    RETRIEVE_NODE,
    REWRITE_QUERY_NODE,
    VISUAL_VLM_GENERATION_NODE,
    RetrieverProtocol,
    VLMServiceProtocol,
    crag_conditional_edge,
    crag_evaluate_node,
    fast_text_table_generation_node,
    retrieve_node,
    rewrite_query_node,
    visual_vlm_generation_node,
)
from src.graph.state import (
    VisionGraphState,
    build_initial_state,
    state_to_query_response,
)
from src.retrieval.reranker import HybridRetriever
from src.schemas.api import MultimodalQueryRequest, MultimodalQueryResponse, RoutingPath
from src.schemas.document import ChunkModality
from src.services.vlm_service import VLMService


class CompiledVisionGraph(Protocol):
    """Protocol representing a compiled LangGraph state machine."""

    async def ainvoke(
        self,
        input: VisionGraphState,
        config: Any = None,
        **kwargs: Any,
    ) -> VisionGraphState:
        """Asynchronously execute the compiled state graph."""
        ...


class _FallbackCompiledGraph:
    """Lightweight deterministic state machine runner when langgraph is unavailable."""

    def __init__(
        self,
        *,
        retriever: RetrieverProtocol,
        vlm_service: VLMServiceProtocol,
        crag_threshold: float,
        max_rewrites: int,
        max_images_per_query: int,
    ) -> None:
        self._retriever = retriever
        self._vlm_service = vlm_service
        self._crag_threshold = crag_threshold
        self._max_rewrites = max_rewrites
        self._max_images_per_query = max_images_per_query

    async def ainvoke(
        self,
        input: VisionGraphState,
        config: Any = None,
        **kwargs: Any,
    ) -> VisionGraphState:
        del config, kwargs
        state = VisionGraphState(**input)

        while True:
            state.update(await retrieve_node(state, self._retriever))
            state.update(await crag_evaluate_node(state, threshold=self._crag_threshold))
            next_target = crag_conditional_edge(
                state,
                threshold=self._crag_threshold,
                max_rewrites=self._max_rewrites,
            )
            if next_target == REWRITE_QUERY_NODE:
                state.update(await rewrite_query_node(state, self._vlm_service))
                continue
            break

        if next_target is RoutingPath.VISUAL_VLM_PATH:
            state.update(
                await visual_vlm_generation_node(
                    state,
                    self._vlm_service,
                    max_images=self._max_images_per_query,
                )
            )
        else:
            state.update(await fast_text_table_generation_node(state, self._vlm_service))

        return state


def build_vision_graph(
    retriever: RetrieverProtocol | None = None,
    vlm_service: VLMServiceProtocol | None = None,
    settings: Settings | None = None,
    *,
    crag_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
    max_rewrites: int = MAX_REWRITE_COUNT,
) -> CompiledVisionGraph:
    """Build and compile the LangGraph CRAG + Adaptive Multimodal Routing state machine."""
    cfg = settings or get_settings()
    active_retriever: RetrieverProtocol = retriever or HybridRetriever(settings=cfg)
    active_vlm: VLMServiceProtocol = vlm_service or VLMService(settings=cfg)
    max_images = min(2, cfg.max_images_per_query)

    if importlib.util.find_spec("langgraph") is None:
        return _FallbackCompiledGraph(
            retriever=active_retriever,
            vlm_service=active_vlm,
            crag_threshold=crag_threshold,
            max_rewrites=max_rewrites,
            max_images_per_query=max_images,
        )

    langgraph_mod = importlib.import_module("langgraph.graph")
    state_graph_cls = langgraph_mod.StateGraph
    start_edge = langgraph_mod.START
    end_edge = langgraph_mod.END

    async def _retrieve(state: VisionGraphState) -> VisionGraphState:
        return await retrieve_node(state, active_retriever)

    async def _evaluate(state: VisionGraphState) -> VisionGraphState:
        return await crag_evaluate_node(state, threshold=crag_threshold)

    async def _rewrite(state: VisionGraphState) -> VisionGraphState:
        return await rewrite_query_node(state, active_vlm)

    async def _fast_gen(state: VisionGraphState) -> VisionGraphState:
        return await fast_text_table_generation_node(state, active_vlm)

    async def _visual_gen(state: VisionGraphState) -> VisionGraphState:
        return await visual_vlm_generation_node(
            state,
            active_vlm,
            max_images=max_images,
        )

    def _route_after_crag(state: VisionGraphState) -> str:
        target = crag_conditional_edge(
            state,
            threshold=crag_threshold,
            max_rewrites=max_rewrites,
        )
        return target.value if isinstance(target, RoutingPath) else str(target)

    graph = state_graph_cls(VisionGraphState)
    graph.add_node(RETRIEVE_NODE, _retrieve)
    graph.add_node(CRAG_EVALUATE_NODE, _evaluate)
    graph.add_node(REWRITE_QUERY_NODE, _rewrite)
    graph.add_node(FAST_TEXT_TABLE_GENERATION_NODE, _fast_gen)
    graph.add_node(VISUAL_VLM_GENERATION_NODE, _visual_gen)

    graph.add_edge(start_edge, RETRIEVE_NODE)
    graph.add_edge(RETRIEVE_NODE, CRAG_EVALUATE_NODE)
    graph.add_conditional_edges(
        CRAG_EVALUATE_NODE,
        _route_after_crag,
        {
            REWRITE_QUERY_NODE: REWRITE_QUERY_NODE,
            RoutingPath.FAST_TEXT_TABLE_PATH.value: FAST_TEXT_TABLE_GENERATION_NODE,
            RoutingPath.VISUAL_VLM_PATH.value: VISUAL_VLM_GENERATION_NODE,
        },
    )
    graph.add_edge(REWRITE_QUERY_NODE, RETRIEVE_NODE)
    graph.add_edge(FAST_TEXT_TABLE_GENERATION_NODE, end_edge)
    graph.add_edge(VISUAL_VLM_GENERATION_NODE, end_edge)

    return cast(CompiledVisionGraph, graph.compile())


class AdaptiveRAGWorkflow:
    """High-level orchestrator wrapping the compiled LangGraph multimodal workflow."""

    def __init__(
        self,
        retriever: RetrieverProtocol | None = None,
        vlm_service: VLMServiceProtocol | None = None,
        settings: Settings | None = None,
        *,
        crag_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        max_rewrites: int = MAX_REWRITE_COUNT,
    ) -> None:
        self._settings = settings or get_settings()
        self._retriever: RetrieverProtocol = retriever or HybridRetriever(settings=self._settings)
        self._vlm_service: VLMServiceProtocol = vlm_service or VLMService(settings=self._settings)
        self._graph = build_vision_graph(
            retriever=self._retriever,
            vlm_service=self._vlm_service,
            settings=self._settings,
            crag_threshold=crag_threshold,
            max_rewrites=max_rewrites,
        )

    @property
    def graph(self) -> CompiledVisionGraph:
        """Return the compiled state machine."""
        return self._graph

    async def ainvoke(self, state: VisionGraphState) -> VisionGraphState:
        """Invoke the underlying state machine with a VisionGraphState dictionary."""
        return await self._graph.ainvoke(state)

    async def run_query(
        self,
        request: MultimodalQueryRequest | str,
        *,
        top_k: int = 5,
        modality_filter: ChunkModality | str | None = None,
        force_visual_route: bool = False,
    ) -> MultimodalQueryResponse:
        """Execute end-to-end retrieval, CRAG evaluation, adaptive routing, and generation."""
        initial_state = build_initial_state(
            request,
            top_k=top_k,
            modality_filter=modality_filter,
            force_visual_route=force_visual_route,
        )
        final_state = await self._graph.ainvoke(initial_state)
        return state_to_query_response(final_state)

    async def close(self) -> None:
        """Close underlying retriever resources if supported."""
        close_fn = getattr(self._retriever, "close", None)
        if callable(close_fn):
            await close_fn()
