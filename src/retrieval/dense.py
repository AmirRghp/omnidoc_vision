"""Dense vector search over Qdrant using BAAI/bge-m3 embeddings."""

from __future__ import annotations

import importlib.util
from typing import Any, Final

from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, models

from src.common.config import Settings, get_settings
from src.common.exceptions import RetrievalError
from src.common.logger import get_logger
from src.ingestion.indexer import (
    DENSE_VECTOR_DIM,
    BaseEmbedder,
    BGEM3Embedder,
    MockEmbedder,
)
from src.schemas.document import ChunkModality, MultimodalChunk

logger = get_logger(__name__)

DENSE_TOP_K_DEFAULT: Final[int] = 20


class ScoredChunk(BaseModel):
    """A MultimodalChunk paired with stage-specific retrieval and reranking scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: MultimodalChunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def normalize_modality_filter(
    modality_filter: ChunkModality | str | None,
) -> ChunkModality | None:
    """Validate and normalize an optional modality filter."""
    if modality_filter is None:
        return None
    if isinstance(modality_filter, ChunkModality):
        return modality_filter
    try:
        return ChunkModality(str(modality_filter).strip().lower())
    except ValueError as exc:
        raise RetrievalError(f"Unsupported modality_filter: {modality_filter}") from exc


class DenseRetriever:
    """Asynchronous Qdrant vector retriever using BAAI/bge-m3 (with MockEmbedder fallback)."""

    def __init__(
        self,
        settings: Settings | None = None,
        qdrant_client: AsyncQdrantClient | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._collection_name = self._settings.qdrant_collection

        if embedder is not None:
            self._embedder = embedder
        elif (
            self._settings.environment == "testing"
            or importlib.util.find_spec("FlagEmbedding") is None
        ):
            self._embedder = MockEmbedder(dimension=DENSE_VECTOR_DIM)
        else:
            self._embedder = BGEM3Embedder(
                model_name=self._settings.embedding_model_name,
                device=self._settings.embedding_device,
            )

        if qdrant_client is not None:
            self._qdrant_client = qdrant_client
        else:
            self._qdrant_client = AsyncQdrantClient(
                host=self._settings.qdrant_host,
                port=self._settings.qdrant_port,
            )

    @property
    def embedder(self) -> BaseEmbedder:
        """The active query embedder."""
        return self._embedder

    @property
    def qdrant_client(self) -> AsyncQdrantClient:
        """The active Qdrant client."""
        return self._qdrant_client

    async def _embed_query(self, query: str) -> list[float]:
        try:
            vectors = await self._embedder.embed_texts([query])
            return vectors[0]
        except Exception as exc:
            if isinstance(self._embedder, BGEM3Embedder):
                logger.warning(
                    "BGEM3Embedder failed during query embedding; falling back to MockEmbedder",
                    extra={"error": str(exc)},
                )
                self._embedder = MockEmbedder(dimension=DENSE_VECTOR_DIM)
                vectors = await self._embedder.embed_texts([query])
                return vectors[0]
            raise RetrievalError(f"Failed to embed query for dense retrieval: {exc}") from exc

    @staticmethod
    def _build_qdrant_filter(
        modality: ChunkModality | None,
        doc_name: str | None,
    ) -> models.Filter | None:
        conditions: list[Any] = []
        if modality is not None:
            conditions.append(
                models.FieldCondition(
                    key="modality",
                    match=models.MatchValue(value=modality.value),
                )
            )
        if doc_name is not None and doc_name.strip():
            conditions.append(
                models.FieldCondition(
                    key="doc_name",
                    match=models.MatchValue(value=doc_name.strip()),
                )
            )
        return models.Filter(must=conditions) if conditions else None

    async def search(
        self,
        query: str,
        *,
        top_k: int = DENSE_TOP_K_DEFAULT,
        modality_filter: ChunkModality | str | None = None,
        doc_name: str | None = None,
    ) -> list[ScoredChunk]:
        """Search Qdrant for the top_k most similar chunks."""
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []

        modality = normalize_modality_filter(modality_filter)
        query_vector = await self._embed_query(cleaned_query)
        qdrant_filter = self._build_qdrant_filter(modality, doc_name)

        try:
            if not await self._qdrant_client.collection_exists(self._collection_name):
                return []

            if hasattr(self._qdrant_client, "query_points"):
                response = await self._qdrant_client.query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                )
                raw_points = list(getattr(response, "points", response))
            else:
                raw_points = await self._qdrant_client.search(  # type: ignore[attr-defined]
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError(f"Qdrant dense vector search failed: {exc}") from exc

        results: list[ScoredChunk] = []
        for point in raw_points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue
            chunk = MultimodalChunk.model_validate(payload)
            score = float(getattr(point, "score", 0.0))
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                )
            )

        return results

    async def close(self) -> None:
        """Close the underlying Qdrant client."""
        await self._qdrant_client.close()
