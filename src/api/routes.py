"""FastAPI route endpoints for ingestion, multimodal querying, crop assets, and health."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from src.common.config import PROJECT_ROOT, Settings, get_settings
from src.common.exceptions import OmniDocError
from src.common.logger import get_logger
from src.graph.workflow import AdaptiveRAGWorkflow
from src.ingestion.docling_parser import DoclingParser
from src.ingestion.indexer import HybridIndexer
from src.ingestion.visual_captioner import VisualCaptioner
from src.schemas.api import (
    ChunkCounts,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    ServiceStatus,
)
from src.schemas.document import ChunkModality
from src.services.cache_service import RedisCacheService, build_query_cache_key

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OmniDoc-Vision API"])

_PARSER_CACHE: dict[int, DoclingParser] = {}
_CAPTIONER_CACHE: dict[int, VisualCaptioner] = {}
_INDEXER_CACHE: dict[int, HybridIndexer] = {}
_WORKFLOW_CACHE: dict[int, AdaptiveRAGWorkflow] = {}
_CACHE_SERVICE_CACHE: dict[int, RedisCacheService] = {}


async def close_cached_services() -> None:
    """Close all singleton client resources held by route dependencies."""
    for indexer in list(_INDEXER_CACHE.values()):
        await indexer.close()
    _INDEXER_CACHE.clear()

    for workflow in list(_WORKFLOW_CACHE.values()):
        await workflow.close()
    _WORKFLOW_CACHE.clear()

    for cache_service in list(_CACHE_SERVICE_CACHE.values()):
        await cache_service.close()
    _CACHE_SERVICE_CACHE.clear()

    _PARSER_CACHE.clear()
    _CAPTIONER_CACHE.clear()


def get_settings_dep(request: Request) -> Settings:
    """Return process settings instance or app.state.settings if configured."""
    active_settings: Settings | None = getattr(request.app.state, "settings", None)
    return active_settings if active_settings is not None else get_settings()


def get_parser_dep(settings: Annotated[Settings, Depends(get_settings_dep)]) -> DoclingParser:
    """Return singleton DoclingParser instance for active settings."""
    key = id(settings)
    if key not in _PARSER_CACHE:
        _PARSER_CACHE[key] = DoclingParser(settings=settings, crops_dir=settings.crops_dir)
    return _PARSER_CACHE[key]


def get_captioner_dep(settings: Annotated[Settings, Depends(get_settings_dep)]) -> VisualCaptioner:
    """Return singleton VisualCaptioner instance for active settings."""
    key = id(settings)
    if key not in _CAPTIONER_CACHE:
        _CAPTIONER_CACHE[key] = VisualCaptioner(settings=settings)
    return _CAPTIONER_CACHE[key]


def get_indexer_dep(settings: Annotated[Settings, Depends(get_settings_dep)]) -> HybridIndexer:
    """Return singleton HybridIndexer instance for active settings."""
    key = id(settings)
    if key not in _INDEXER_CACHE:
        _INDEXER_CACHE[key] = HybridIndexer(settings=settings)
    return _INDEXER_CACHE[key]


def get_workflow_dep(
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> AdaptiveRAGWorkflow:
    """Return singleton AdaptiveRAGWorkflow instance for active settings."""
    key = id(settings)
    if key not in _WORKFLOW_CACHE:
        _WORKFLOW_CACHE[key] = AdaptiveRAGWorkflow(settings=settings)
    return _WORKFLOW_CACHE[key]


def get_cache_dep(
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> RedisCacheService:
    """Return singleton RedisCacheService instance for active settings."""
    key = id(settings)
    if key not in _CACHE_SERVICE_CACHE:
        _CACHE_SERVICE_CACHE[key] = RedisCacheService(settings=settings)
    return _CACHE_SERVICE_CACHE[key]


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(
    request: IngestRequest,
    settings: Annotated[Settings, Depends(get_settings_dep)],
    parser: Annotated[DoclingParser, Depends(get_parser_dep)],
    captioner: Annotated[VisualCaptioner, Depends(get_captioner_dep)],
    indexer: Annotated[HybridIndexer, Depends(get_indexer_dep)],
    cache: Annotated[RedisCacheService, Depends(get_cache_dep)],
) -> IngestResponse:
    """Ingest a PDF document, extract multimodal chunks, generate VLM captions, and index."""
    raw_dir = settings.raw_pdfs_dir.resolve()
    project_root = PROJECT_ROOT.resolve()

    target_subpath = Path(request.pdf_path)
    candidate_paths = [
        (raw_dir / target_subpath).resolve(),
        (project_root / target_subpath).resolve(),
    ]

    resolved_pdf: Path | None = None
    for cand in candidate_paths:
        if cand.is_file() and (cand.is_relative_to(raw_dir) or cand.is_relative_to(project_root)):
            resolved_pdf = cand
            break

    if resolved_pdf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file '{request.pdf_path}' not found",
        )

    if resolved_pdf.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for ingestion",
        )

    file_size_mb = resolved_pdf.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.max_pdf_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"PDF size ({file_size_mb:.1f} MB) exceeds maximum allowed "
                f"{settings.max_pdf_size_mb} MB"
            ),
        )

    start_time = time.perf_counter()
    logger.info(
        "Starting ingestion for %s (generate_vlm_captions=%s)",
        resolved_pdf.name,
        request.generate_vlm_captions,
    )

    try:
        chunks = await parser.aparse_pdf(resolved_pdf)
        if request.generate_vlm_captions:
            chunks = await captioner.caption_chunks(chunks)
        await indexer.index_chunks(chunks)
        await cache.invalidate_all()
    except OmniDocError:
        raise
    except Exception as exc:
        logger.error("Failed to ingest %s: %s", resolved_pdf.name, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion processing failed: {exc}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    text_count = sum(1 for c in chunks if c.modality == ChunkModality.TEXT)
    table_count = sum(1 for c in chunks if c.modality == ChunkModality.TABLE)
    figure_count = sum(1 for c in chunks if c.modality == ChunkModality.FIGURE)
    total_count = len(chunks)

    pages_processed = max((c.page_number for c in chunks), default=0)
    doc_id = chunks[0].doc_id if chunks else f"doc_{resolved_pdf.stem}"
    doc_name = resolved_pdf.name

    logger.info(
        "Ingestion completed for %s: %d chunks in %.1f ms",
        doc_name,
        total_count,
        elapsed_ms,
    )

    return IngestResponse(
        status="indexed",
        doc_id=doc_id,
        doc_name=doc_name,
        pages_processed=pages_processed,
        chunks_created=ChunkCounts(
            text=text_count,
            table=table_count,
            figure=figure_count,
            total=total_count,
        ),
        elapsed_ms=elapsed_ms,
    )


@router.post("/query", response_model=MultimodalQueryResponse, status_code=status.HTTP_200_OK)
async def query_multimodal(
    request: MultimodalQueryRequest,
    workflow: Annotated[AdaptiveRAGWorkflow, Depends(get_workflow_dep)],
    cache: Annotated[RedisCacheService, Depends(get_cache_dep)],
) -> MultimodalQueryResponse:
    """Execute hybrid retrieval, CRAG evaluation, adaptive routing, and multimodal generation."""
    logger.info("Executing multimodal query: '%s' (top_k=%d)", request.query, request.top_k)
    cache_key = build_query_cache_key(
        request.query,
        request.modality_filter,
        force_visual_route=request.force_visual_route,
        top_k=request.top_k,
    )
    cached_response = await cache.get(cache_key)
    if cached_response is not None:
        logger.debug("Cache hit for query key %s", cache_key)
        return cached_response

    try:
        response = await workflow.run_query(request)
    except OmniDocError:
        raise
    except Exception as exc:
        logger.error("Query execution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {exc}",
        ) from exc

    await cache.set(cache_key, response)
    return response


@router.get("/crops/{crop_filename}")
async def get_crop_asset(
    crop_filename: str,
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> FileResponse:
    """Serve high-resolution cropped diagram or table with strict path traversal validation."""
    if "/" in crop_filename or "\\" in crop_filename or ".." in crop_filename:
        logger.warning("Path traversal attempt rejected for crop_filename: '%s'", crop_filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid crop filename: path separators and parent references are forbidden",
        )

    sanitized_name = Path(crop_filename).name
    if sanitized_name != crop_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid crop filename structure",
        )

    crops_dir = settings.crops_dir.resolve()
    resolved_path = (crops_dir / sanitized_name).resolve()

    if not resolved_path.is_relative_to(crops_dir):
        logger.warning("Crop path outside crops_dir rejected: '%s'", resolved_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal violation detected",
        )

    if not resolved_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crop asset '{crop_filename}' not found",
        )

    media_type = "image/png"
    if resolved_path.suffix.lower() in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"

    return FileResponse(path=resolved_path, media_type=media_type)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    indexer: Annotated[HybridIndexer, Depends(get_indexer_dep)],
) -> HealthResponse:
    """Return readiness status of Qdrant, BM25 index, Redis cache, and vLLM server."""
    services: dict[str, str] = {}

    # 1. Qdrant check
    try:
        await asyncio.wait_for(indexer.qdrant_client.get_collections(), timeout=1.0)
        services["qdrant"] = ServiceStatus.OK.value
    except Exception:
        services["qdrant"] = ServiceStatus.UNREACHABLE.value

    # 2. BM25 check
    if settings.bm25_index_dir.exists():
        services["bm25"] = ServiceStatus.OK.value
    else:
        services["bm25"] = ServiceStatus.MISSING.value

    # 3. Redis check
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(settings.redis_url, socket_timeout=1.0)
        await asyncio.wait_for(redis_client.ping(), timeout=1.0)
        await redis_client.aclose()
        services["redis"] = ServiceStatus.OK.value
    except Exception:
        services["redis"] = ServiceStatus.UNAVAILABLE.value

    # 4. vLLM check
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{settings.vlm_base_url}/models")
            is_ok = resp.status_code < 400
            services["vlm"] = ServiceStatus.OK.value if is_ok else ServiceStatus.UNREACHABLE.value
    except Exception:
        services["vlm"] = ServiceStatus.UNREACHABLE.value

    # Determine overall status
    is_core_healthy = services["qdrant"] == ServiceStatus.OK.value
    is_all_healthy = all(status_val == ServiceStatus.OK.value for status_val in services.values())
    overall_status: Literal["healthy", "degraded"] = (
        "healthy" if (is_all_healthy or is_core_healthy) else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        services=services,
    )
