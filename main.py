"""OmniDoc-Vision FastAPI Application Entrypoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    close_cached_services,
    get_indexer_dep,
    get_settings_dep,
    health_check,
)
from src.api.routes import router as api_v1_router
from src.common.config import Settings, get_settings
from src.common.exceptions import (
    AssetAccessError,
    DocumentProcessingError,
    GenerationError,
    IndexingError,
    OmniDocError,
    PdfValidationError,
    RetrievalError,
)
from src.common.logger import get_logger
from src.ingestion.indexer import HybridIndexer
from src.schemas.api import HealthResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles."""
    settings: Settings = getattr(app.state, "settings", get_settings())
    logger.info("Initializing OmniDoc-Vision backend (env=%s)...", settings.environment)
    settings.crops_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        logger.info("Shutting down OmniDoc-Vision backend...")
        await close_cached_services()


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI application factory."""
    active_settings = settings or get_settings()

    app = FastAPI(
        title="OmniDoc-Vision API",
        version="0.1.0",
        description=(
            "Multimodal Document Intelligence & Visual RAG for Complex Engineering PDFs. "
            "IBM Docling + Approach A Visual Captioning + Qdrant/BM25 Hybrid Retrieval + "
            "LangGraph CRAG & Adaptive Routing + Qwen2.5-VL"
        ),
        lifespan=lifespan,
    )

    app.state.settings = active_settings

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(api_v1_router)

    # Top-level /health alias
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        status_code=status.HTTP_200_OK,
    )
    async def root_health(
        resolved_settings: Annotated[Settings, Depends(get_settings_dep)],
        indexer: Annotated[HybridIndexer, Depends(get_indexer_dep)],
    ) -> HealthResponse:
        return await health_check(settings=resolved_settings, indexer=indexer)

    # Custom exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request payload or parameters",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(PdfValidationError)
    async def pdf_validation_handler(request: Request, exc: PdfValidationError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"code": "PDF_VALIDATION_ERROR", "message": str(exc)}},
        )

    @app.exception_handler(AssetAccessError)
    async def asset_access_handler(request: Request, exc: AssetAccessError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"code": "ASSET_ACCESS_ERROR", "message": str(exc)}},
        )

    @app.exception_handler(IndexingError)
    async def indexing_error_handler(request: Request, exc: IndexingError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INDEXING_ERROR", "message": str(exc)}},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "RETRIEVAL_ERROR", "message": str(exc)}},
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "GENERATION_ERROR", "message": str(exc)}},
        )

    @app.exception_handler(DocumentProcessingError)
    async def processing_error_handler(
        request: Request,
        exc: DocumentProcessingError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "PROCESSING_ERROR", "message": str(exc)}},
        )

    @app.exception_handler(OmniDocError)
    async def general_omnidoc_handler(request: Request, exc: OmniDocError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
        )

    return app


app = create_app()

if __name__ == "__main__":
    current_settings = get_settings()
    uvicorn.run(
        "main:app",
        host=current_settings.host,
        port=current_settings.port,
        reload=True,
    )
