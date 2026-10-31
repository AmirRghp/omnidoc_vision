"""State definitions and schema converters for the LangGraph CRAG and adaptive routing workflow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from src.retrieval.dense import ScoredChunk, normalize_modality_filter
from src.schemas.api import (
    CRAGStatus,
    LatencyBreakdown,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    RoutingPath,
    VisualCitation,
)
from src.schemas.document import ChunkModality, MultimodalChunk
from src.services.vlm_service import unwrap_chunk

CRAG_CONFIDENCE_THRESHOLD: Final[float] = 0.35
MAX_REWRITE_COUNT: Final[int] = 1
CROP_URL_PREFIX: Final[str] = "/api/v1/crops/"
CONTENT_PREVIEW_MAX_CHARS: Final[int] = 240


class VisionGraphState(TypedDict, total=False):
    """Typed state dictionary flowing through the LangGraph multimodal workflow."""

    query: str
    active_query: str
    rewritten_query: str | None
    top_k: int
    modality_filter: ChunkModality | None
    force_visual_route: bool
    rewrite_count: int
    retrieved_chunks: list[ScoredChunk]
    max_rerank_score: float
    crag_status: CRAGStatus
    routing_path: RoutingPath
    answer: str
    vlm_images_inspected: int
    citations: list[VisualCitation]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


def build_initial_state(
    request: MultimodalQueryRequest | str,
    *,
    top_k: int = 5,
    modality_filter: ChunkModality | str | None = None,
    force_visual_route: bool = False,
) -> VisionGraphState:
    """Create a normalized initial VisionGraphState from a request model or raw query string."""
    if isinstance(request, MultimodalQueryRequest):
        clean_query = request.query.strip()
        resolved_top_k = request.top_k
        resolved_modality = request.modality_filter
        resolved_force_visual = request.force_visual_route
    else:
        clean_query = request.strip()
        resolved_top_k = top_k
        resolved_modality = normalize_modality_filter(modality_filter)
        resolved_force_visual = force_visual_route

    return VisionGraphState(
        query=clean_query,
        active_query=clean_query,
        rewritten_query=None,
        top_k=resolved_top_k,
        modality_filter=resolved_modality,
        force_visual_route=resolved_force_visual,
        rewrite_count=0,
        retrieved_chunks=[],
        max_rerank_score=0.0,
        crag_status=CRAGStatus.LOW_CONFIDENCE,
        routing_path=RoutingPath.FAST_TEXT_TABLE_PATH,
        answer="",
        vlm_images_inspected=0,
        citations=[],
        retrieval_ms=0.0,
        rerank_ms=0.0,
        generation_ms=0.0,
        total_ms=0.0,
    )


def _build_content_preview(content: str, max_chars: int = CONTENT_PREVIEW_MAX_CHARS) -> str:
    cleaned = " ".join(content.split()).strip()
    if not cleaned:
        return "[EMPTY CHUNK]"
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def chunk_to_visual_citation(item: ScoredChunk | MultimodalChunk) -> VisualCitation:
    """Convert a ScoredChunk or MultimodalChunk into a validated VisualCitation."""
    chunk = unwrap_chunk(item)
    if isinstance(item, ScoredChunk):
        raw_score = item.rerank_score if item.rerank_score is not None else item.score
    else:
        raw_score = 0.0

    clamped_score = round(max(0.0, min(1.0, float(raw_score))), 6)

    image_url: str | None = None
    if chunk.image_path:
        crop_filename = Path(chunk.image_path).name
        image_url = f"{CROP_URL_PREFIX}{crop_filename}"

    return VisualCitation(
        chunk_id=chunk.chunk_id,
        doc_name=chunk.doc_name,
        page_number=chunk.page_number,
        modality=chunk.modality,
        section_title=chunk.section_title,
        rerank_score=clamped_score,
        bounding_box=chunk.bounding_box,
        image_url=image_url,
        content_preview=_build_content_preview(chunk.content),
    )


def build_citations(chunks: Sequence[ScoredChunk | MultimodalChunk]) -> list[VisualCitation]:
    """Convert a sequence of retrieved chunks into VisualCitation objects."""
    return [chunk_to_visual_citation(item) for item in chunks]


def state_to_query_response(state: VisionGraphState) -> MultimodalQueryResponse:
    """Compile the final VisionGraphState into a validated MultimodalQueryResponse."""
    retrieval_ms = round(max(0.0, float(state.get("retrieval_ms", 0.0))), 2)
    rerank_ms = round(max(0.0, float(state.get("rerank_ms", 0.0))), 2)
    generation_ms = round(max(0.0, float(state.get("generation_ms", 0.0))), 2)
    raw_total = float(state.get("total_ms", 0.0))
    computed_total = round(retrieval_ms + rerank_ms + generation_ms, 2)
    total_ms = round(max(raw_total, computed_total), 2)

    answer = (state.get("answer") or "").strip()
    if not answer:
        answer = "No grounded answer could be generated for the given query."

    citations = state.get("citations")
    if citations is None:
        citations = build_citations(state.get("retrieved_chunks", []))

    rewritten_raw = state.get("rewritten_query")
    rewritten_query = rewritten_raw.strip() if rewritten_raw and rewritten_raw.strip() else None

    vlm_images = max(0, min(2, int(state.get("vlm_images_inspected", 0))))

    return MultimodalQueryResponse(
        answer=answer,
        routing_path=state.get("routing_path", RoutingPath.FAST_TEXT_TABLE_PATH),
        crag_status=state.get("crag_status", CRAGStatus.LOW_CONFIDENCE),
        rewritten_query=rewritten_query,
        vlm_images_inspected=vlm_images,
        latency_ms=LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        ),
        citations=list(citations),
    )
