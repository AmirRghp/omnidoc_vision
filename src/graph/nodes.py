"""LangGraph node implementations and conditional routing edges for CRAG and multimodal routing."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Protocol

from src.common.logger import get_logger
from src.graph.state import (
    CRAG_CONFIDENCE_THRESHOLD,
    MAX_REWRITE_COUNT,
    VisionGraphState,
    build_citations,
)
from src.retrieval.dense import ScoredChunk
from src.retrieval.reranker import HybridRetrievalResult
from src.schemas.api import CRAGStatus, RoutingPath
from src.schemas.document import ChunkModality, MultimodalChunk
from src.services.vlm_service import unwrap_chunk

logger = get_logger(__name__)

RETRIEVE_NODE: Final[str] = "retrieve_node"
CRAG_EVALUATE_NODE: Final[str] = "crag_evaluate_node"
REWRITE_QUERY_NODE: Final[str] = "rewrite_query_node"
FAST_TEXT_TABLE_GENERATION_NODE: Final[str] = "fast_text_table_generation_node"
VISUAL_VLM_GENERATION_NODE: Final[str] = "visual_vlm_generation_node"

MAX_VISUAL_IMAGES: Final[int] = 2

CRAGRouteTarget = Literal[
    "rewrite_query_node",
    RoutingPath.FAST_TEXT_TABLE_PATH,
    RoutingPath.VISUAL_VLM_PATH,
]

__all__ = [
    "CRAG_CONFIDENCE_THRESHOLD",
    "CRAG_EVALUATE_NODE",
    "CRAGRouteTarget",
    "FAST_TEXT_TABLE_GENERATION_NODE",
    "MAX_REWRITE_COUNT",
    "MAX_VISUAL_IMAGES",
    "RETRIEVE_NODE",
    "REWRITE_QUERY_NODE",
    "RetrieverProtocol",
    "VISUAL_VLM_GENERATION_NODE",
    "VLMServiceProtocol",
    "adaptive_router_edge",
    "crag_conditional_edge",
    "crag_evaluate_node",
    "fast_text_table_generation_node",
    "retrieve_node",
    "rewrite_query_node",
    "select_visual_image_paths",
    "visual_vlm_generation_node",
]


class RetrieverProtocol(Protocol):
    """Structural interface for hybrid retrieval used by retrieve_node."""

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        modality_filter: ChunkModality | str | None = None,
    ) -> HybridRetrievalResult:
        """Retrieve and rerank multimodal chunks for the given query."""
        ...


class VLMServiceProtocol(Protocol):
    """Structural interface for VLM fast text, visual reasoning, and CRAG query rewriting."""

    async def generate_fast_text(
        self,
        query: str,
        chunks: Sequence[MultimodalChunk | ScoredChunk],
    ) -> str:
        """Generate grounded response from text and reconstructed table chunks."""
        ...

    async def generate_visual_reasoning(
        self,
        query: str,
        chunks: Sequence[MultimodalChunk | ScoredChunk],
        image_paths: Sequence[Path | str] | None = None,
    ) -> str:
        """Generate grounded multimodal response inspecting up to 2 cropped images."""
        ...

    async def rewrite_query(self, query: str) -> str:
        """Rewrite an ambiguous query into a keyword-rich technical search query."""
        ...


def _extract_chunk_score(item: ScoredChunk) -> float:
    score = item.rerank_score if item.rerank_score is not None else item.score
    return float(score)


def select_visual_image_paths(
    chunks: Sequence[ScoredChunk | MultimodalChunk],
    *,
    max_images: int = MAX_VISUAL_IMAGES,
    include_non_figure_crops: bool = False,
) -> list[str]:
    """Select up to max_images (hard-capped at 2) unique image paths from retrieved chunks."""
    cap = max(0, min(MAX_VISUAL_IMAGES, max_images))
    if cap == 0:
        return []

    selected: list[str] = []
    for item in chunks:
        chunk = unwrap_chunk(item)
        if (
            chunk.modality is ChunkModality.FIGURE
            and chunk.image_path
            and chunk.image_path not in selected
        ):
            selected.append(chunk.image_path)
            if len(selected) >= cap:
                return selected

    if include_non_figure_crops and not selected:
        for item in chunks:
            chunk = unwrap_chunk(item)
            if chunk.image_path and chunk.image_path not in selected:
                selected.append(chunk.image_path)
                if len(selected) >= cap:
                    return selected

    return selected


async def retrieve_node(
    state: VisionGraphState,
    retriever: RetrieverProtocol,
) -> VisionGraphState:
    """Execute hybrid retrieval + cross-encoder reranking for the active query."""
    active_query = (state.get("active_query") or state.get("query") or "").strip()
    top_k = int(state.get("top_k", 5))
    modality_filter = state.get("modality_filter")

    result = await retriever.retrieve(
        active_query,
        top_k=top_k,
        modality_filter=modality_filter,
    )

    prev_retrieval_ms = float(state.get("retrieval_ms", 0.0))
    prev_rerank_ms = float(state.get("rerank_ms", 0.0))
    retrieval_ms = round(prev_retrieval_ms + result.retrieval_ms, 2)
    rerank_ms = round(prev_rerank_ms + result.rerank_ms, 2)

    logger.info(
        "Executed retrieve_node",
        extra={
            "active_query": active_query,
            "retrieved_count": len(result.chunks),
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
        },
    )

    return VisionGraphState(
        active_query=active_query,
        retrieved_chunks=list(result.chunks),
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
    )


async def crag_evaluate_node(
    state: VisionGraphState,
    *,
    threshold: float = CRAG_CONFIDENCE_THRESHOLD,
) -> VisionGraphState:
    """Evaluate top reranker confidence score against the CRAG threshold (0.35)."""
    chunks = state.get("retrieved_chunks", [])
    max_score = max((_extract_chunk_score(c) for c in chunks), default=0.0)
    max_score = round(max(0.0, min(1.0, max_score)), 6)

    crag_status = (
        CRAGStatus.CONFIDENT if max_score >= threshold else CRAGStatus.LOW_CONFIDENCE
    )

    logger.info(
        "Evaluated CRAG retrieval confidence",
        extra={
            "max_rerank_score": max_score,
            "threshold": threshold,
            "crag_status": crag_status.value,
            "rewrite_count": state.get("rewrite_count", 0),
        },
    )

    return VisionGraphState(
        max_rerank_score=max_score,
        crag_status=crag_status,
    )


async def rewrite_query_node(
    state: VisionGraphState,
    vlm_service: VLMServiceProtocol,
) -> VisionGraphState:
    """Rewrite an ambiguous or low-confidence query using VLMService."""
    base_query = (state.get("query") or state.get("active_query") or "").strip()
    rewritten_raw = await vlm_service.rewrite_query(base_query)
    rewritten = rewritten_raw.strip() if rewritten_raw and rewritten_raw.strip() else base_query
    next_rewrite_count = int(state.get("rewrite_count", 0)) + 1

    logger.info(
        "Rewrote query in CRAG self-correction loop",
        extra={
            "original_query": base_query,
            "rewritten_query": rewritten,
            "rewrite_count": next_rewrite_count,
        },
    )

    return VisionGraphState(
        rewritten_query=rewritten,
        active_query=rewritten,
        rewrite_count=next_rewrite_count,
    )


def adaptive_router_edge(state: VisionGraphState) -> RoutingPath:
    """Select VISUAL_VLM_PATH if any chunk is a figure or forced, else FAST_TEXT_TABLE_PATH."""
    if state.get("force_visual_route", False):
        return RoutingPath.VISUAL_VLM_PATH

    chunks = state.get("retrieved_chunks", [])
    has_figure = any(unwrap_chunk(item).modality is ChunkModality.FIGURE for item in chunks)
    if has_figure:
        return RoutingPath.VISUAL_VLM_PATH

    return RoutingPath.FAST_TEXT_TABLE_PATH


def crag_conditional_edge(
    state: VisionGraphState,
    *,
    threshold: float = CRAG_CONFIDENCE_THRESHOLD,
    max_rewrites: int = MAX_REWRITE_COUNT,
) -> CRAGRouteTarget:
    """Route to rewrite_query_node if confidence < threshold and rewrite_count < 1, else route."""
    chunks = state.get("retrieved_chunks", [])
    max_score = float(
        state.get(
            "max_rerank_score",
            max((_extract_chunk_score(c) for c in chunks), default=0.0),
        )
    )
    rewrite_count = int(state.get("rewrite_count", 0))

    if max_score < threshold and rewrite_count < max_rewrites:
        return "rewrite_query_node"

    return adaptive_router_edge(state)


async def fast_text_table_generation_node(
    state: VisionGraphState,
    vlm_service: VLMServiceProtocol,
) -> VisionGraphState:
    """Generate a fast text/table answer without encoding or inspecting Base64 images."""
    query = (state.get("query") or state.get("active_query") or "").strip()
    chunks = state.get("retrieved_chunks", [])

    t0 = time.perf_counter()
    answer = await vlm_service.generate_fast_text(query, chunks)
    generation_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    retrieval_ms = float(state.get("retrieval_ms", 0.0))
    rerank_ms = float(state.get("rerank_ms", 0.0))
    total_ms = round(retrieval_ms + rerank_ms + generation_ms, 2)
    citations = build_citations(chunks)

    logger.info(
        "Completed fast text/table generation",
        extra={
            "routing_path": RoutingPath.FAST_TEXT_TABLE_PATH.value,
            "vlm_images_count": 0,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
    )

    return VisionGraphState(
        answer=answer,
        routing_path=RoutingPath.FAST_TEXT_TABLE_PATH,
        vlm_images_inspected=0,
        citations=citations,
        generation_ms=generation_ms,
        total_ms=total_ms,
    )


async def visual_vlm_generation_node(
    state: VisionGraphState,
    vlm_service: VLMServiceProtocol,
    *,
    max_images: int = MAX_VISUAL_IMAGES,
) -> VisionGraphState:
    """Generate a pixel-aware multimodal answer attaching up to 2 cropped figures."""
    query = (state.get("query") or state.get("active_query") or "").strip()
    chunks = state.get("retrieved_chunks", [])
    force_visual = bool(state.get("force_visual_route", False))

    image_paths = select_visual_image_paths(
        chunks,
        max_images=max_images,
        include_non_figure_crops=force_visual,
    )

    t0 = time.perf_counter()
    answer = await vlm_service.generate_visual_reasoning(
        query,
        chunks,
        image_paths=image_paths,
    )
    generation_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    retrieval_ms = float(state.get("retrieval_ms", 0.0))
    rerank_ms = float(state.get("rerank_ms", 0.0))
    total_ms = round(retrieval_ms + rerank_ms + generation_ms, 2)
    citations = build_citations(chunks)
    vlm_images_inspected = min(MAX_VISUAL_IMAGES, len(image_paths))

    logger.info(
        "Completed visual VLM generation",
        extra={
            "routing_path": RoutingPath.VISUAL_VLM_PATH.value,
            "vlm_images_count": vlm_images_inspected,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
    )

    return VisionGraphState(
        answer=answer,
        routing_path=RoutingPath.VISUAL_VLM_PATH,
        vlm_images_inspected=vlm_images_inspected,
        citations=citations,
        generation_ms=generation_ms,
        total_ms=total_ms,
    )
