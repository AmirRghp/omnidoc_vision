"""Unified hybrid indexer combining Qdrant dense vector search (BGE-M3) and BM25s sparse index.

Indexes text, reconstructed tables, and technical figure captions into a synchronized
multimodal retrieval store preserving all layout metadata and bounding boxes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import bm25s
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient, models

from src.common.config import Settings, get_settings
from src.common.exceptions import IndexingError
from src.common.logger import get_logger
from src.schemas.document import MultimodalChunk

logger = get_logger(__name__)

DENSE_VECTOR_DIM: Final[int] = 1024


class IndexingStats(BaseModel):
    """Statistics returned after indexing a batch of multimodal chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_chunks: int = Field(ge=0)
    indexed_dense: int = Field(ge=0)
    indexed_sparse: int = Field(ge=0)
    doc_ids: list[str] = Field(default_factory=list)


class BaseEmbedder(ABC):
    """Abstract base class for dense vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute normalized dense embeddings for a sequence of texts."""
        ...


class MockEmbedder(BaseEmbedder):
    """Deterministic, zero-dependency unit-length embedder for CPU testing and CI."""

    def __init__(self, dimension: int = DENSE_VECTOR_DIM) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _hash_to_vector(text: str, dim: int) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._hash_to_vector(t, self._dim) for t in texts]


class BGEM3Embedder(BaseEmbedder):
    """Production dense embedder using BAAI/bge-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    @property
    def dimension(self) -> int:
        return DENSE_VECTOR_DIM

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel

            use_fp16 = self._device.lower() not in ("cpu", "mps")
            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=use_fp16,
                device=self._device,
            )
            return self._model
        except ImportError as exc:
            raise IndexingError(
                "FlagEmbedding is required for BGEM3Embedder. Install it or use MockEmbedder."
            ) from exc

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        outputs = model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vecs = outputs["dense_vecs"]
        return [[float(x) for x in v] for v in dense_vecs]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts))


class HybridIndexer:
    """Unified hybrid indexer managing Qdrant dense vector store and BM25s sparse index."""

    def __init__(
        self,
        settings: Settings | None = None,
        qdrant_client: AsyncQdrantClient | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._collection_name = self._settings.qdrant_collection
        self._bm25_dir = Path(self._settings.bm25_index_dir)
        self._bm25_metadata_file = self._bm25_dir / "corpus_metadata.json"

        if embedder is not None:
            self._embedder = embedder
        elif self._settings.environment == "testing":
            self._embedder = MockEmbedder(dimension=DENSE_VECTOR_DIM)
        else:
            try:
                self._embedder = BGEM3Embedder(
                    model_name=self._settings.embedding_model_name,
                    device=self._settings.embedding_device,
                )
            except Exception:
                logger.warning("Falling back to MockEmbedder as FlagEmbedding is unavailable")
                self._embedder = MockEmbedder(dimension=DENSE_VECTOR_DIM)

        if qdrant_client is not None:
            self._qdrant_client = qdrant_client
        else:
            self._qdrant_client = AsyncQdrantClient(
                host=self._settings.qdrant_host,
                port=self._settings.qdrant_port,
            )

    @property
    def embedder(self) -> BaseEmbedder:
        """The active embedder instance."""
        return self._embedder

    @property
    def qdrant_client(self) -> AsyncQdrantClient:
        """The active Qdrant client."""
        return self._qdrant_client

    async def ensure_collection(self) -> None:
        """Ensure the target Qdrant collection and payload indexes exist."""
        try:
            exists = await self._qdrant_client.collection_exists(self._collection_name)
            if not exists:
                logger.info(
                    "Creating Qdrant collection",
                    extra={"collection": self._collection_name},
                )
                await self._qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedder.dimension,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    ),
                )

            payload_indexes = [
                ("modality", models.PayloadSchemaType.KEYWORD),
                ("doc_name", models.PayloadSchemaType.KEYWORD),
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("page_number", models.PayloadSchemaType.INTEGER),
            ]
            for field_name, field_schema in payload_indexes:
                with contextlib.suppress(Exception):
                    await self._qdrant_client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
        except Exception as exc:
            if isinstance(exc, IndexingError):
                raise
            err_msg = f"Failed to ensure Qdrant collection '{self._collection_name}': {exc}"
            raise IndexingError(err_msg) from exc

    def _load_bm25_records_sync(self) -> list[dict[str, Any]]:
        """Load persistent BM25 chunk metadata synchronously."""
        if not self._bm25_metadata_file.exists():
            return []
        try:
            content = self._bm25_metadata_file.read_text(encoding="utf-8")
            data = json.loads(content)
            return list(data) if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(
                "Could not read existing BM25 metadata, starting fresh",
                extra={"error": str(exc)},
            )
            return []

    def _save_bm25_index_sync(self, records: list[dict[str, Any]]) -> None:
        """Tokenize, index, and save the BM25s sparse index synchronously."""
        self._bm25_dir.mkdir(parents=True, exist_ok=True)
        if not records:
            if self._bm25_metadata_file.exists():
                self._bm25_metadata_file.unlink()
            return

        corpus_texts = [r["content"] for r in records]
        tokens = bm25s.tokenize(corpus_texts, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokens)
        retriever.save(str(self._bm25_dir), corpus=corpus_texts)

        self._bm25_metadata_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def index_chunks(self, chunks: Sequence[MultimodalChunk]) -> IndexingStats:
        """Index a batch of MultimodalChunks into Qdrant and BM25s."""
        if not chunks:
            return IndexingStats(total_chunks=0, indexed_dense=0, indexed_sparse=0, doc_ids=[])

        await self.ensure_collection()

        texts = [c.content for c in chunks]
        dense_vectors = await self._embedder.embed_texts(texts)

        points: list[models.PointStruct] = []
        new_records: list[dict[str, Any]] = []
        unique_doc_ids: set[str] = set()

        for chunk, vector in zip(chunks, dense_vectors, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "modality": chunk.modality.value,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "table_html": chunk.table_html,
                "image_path": chunk.image_path,
                "bounding_box": chunk.bounding_box.model_dump(),
                "ingestion_timestamp": chunk.ingestion_timestamp.isoformat(),
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
            new_records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_name": chunk.doc_name,
                    "page_number": chunk.page_number,
                    "modality": chunk.modality.value,
                    "content": chunk.content,
                }
            )
            unique_doc_ids.add(chunk.doc_id)

        try:
            await self._qdrant_client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise IndexingError(f"Failed to upsert points into Qdrant: {exc}") from exc

        existing_records = await asyncio.to_thread(self._load_bm25_records_sync)
        record_map = {r["chunk_id"]: r for r in existing_records}
        for rec in new_records:
            record_map[rec["chunk_id"]] = rec

        merged_records = list(record_map.values())
        await asyncio.to_thread(self._save_bm25_index_sync, merged_records)

        stats = IndexingStats(
            total_chunks=len(chunks),
            indexed_dense=len(points),
            indexed_sparse=len(merged_records),
            doc_ids=sorted(unique_doc_ids),
        )
        logger.info(
            "Indexed multimodal chunks",
            extra={
                "total_chunks": stats.total_chunks,
                "indexed_dense": stats.indexed_dense,
                "indexed_sparse": stats.indexed_sparse,
            },
        )
        return stats

    async def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a given doc_id from both Qdrant and BM25s."""
        await self.ensure_collection()

        delete_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        await self._qdrant_client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(filter=delete_filter),
            wait=True,
        )

        existing_records = await asyncio.to_thread(self._load_bm25_records_sync)
        remaining_records = [r for r in existing_records if r.get("doc_id") != doc_id]
        deleted_count = len(existing_records) - len(remaining_records)
        await asyncio.to_thread(self._save_bm25_index_sync, remaining_records)

        logger.info(
            "Deleted document chunks from hybrid index",
            extra={"doc_id": doc_id, "deleted_count": deleted_count},
        )
        return deleted_count

    async def close(self) -> None:
        """Close client connections."""
        await self._qdrant_client.close()
