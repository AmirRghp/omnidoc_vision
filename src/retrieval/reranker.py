"""Cross-Encoder reranking (BAAI/bge-reranker-v2-m3) and parallel hybrid retrieval orchestrator."""

from __future__ import annotations

import asyncio
import importlib.util
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from src.common.config import Settings, get_settings
from src.common.exceptions import RetrievalError
from src.common.logger import get_logger
from src.ingestion.indexer import normalize_bilingual_text
from src.retrieval.dense import (
    DENSE_TOP_K_DEFAULT,
    DenseRetriever,
    ScoredChunk,
    normalize_modality_filter,
)
from src.retrieval.fusion import RRF_K_DEFAULT, RRF_TOP_N_DEFAULT, reciprocal_rank_fusion
from src.retrieval.sparse import SPARSE_TOP_K_DEFAULT, SparseRetriever
from src.schemas.document import ChunkModality

logger = get_logger(__name__)

RERANK_TOP_K_DEFAULT: Final[int] = 5
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"(?u)\b\w+\b")


def _sigmoid(value: float) -> float:
    clamped = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clamped))


class BaseReranker(ABC):
    """Abstract interface for cross-encoder candidate rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int = RERANK_TOP_K_DEFAULT,
    ) -> list[ScoredChunk]:
        """Re-score and sort candidates descending by relevance score in [0.0, 1.0]."""
        ...


class MockReranker(BaseReranker):
    """Deterministic bilingual lexical + RRF reranker for offline CPU testing and CI."""

    @staticmethod
    def _score_pair(query_tokens: set[str], candidate: ScoredChunk) -> float:
        doc_text = normalize_bilingual_text(
            f"{candidate.chunk.section_title or ''} {candidate.chunk.content}"
        )
        doc_tokens = set(_WORD_RE.findall(doc_text))
        if not query_tokens or not doc_tokens:
            overlap_ratio = 0.0
        else:
            overlap_ratio = len(query_tokens & doc_tokens) / float(len(query_tokens))

        rrf_bonus = min(0.25, (candidate.rrf_score or candidate.score or 0.0) * 6.0)
        raw_score = (0.75 * overlap_ratio) + rrf_bonus
        return round(max(0.0, min(1.0, raw_score)), 6)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int = RERANK_TOP_K_DEFAULT,
    ) -> list[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []

        norm_query = normalize_bilingual_text(query)
        query_tokens = set(_WORD_RE.findall(norm_query))

        reranked: list[ScoredChunk] = []
        for candidate in candidates:
            score = self._score_pair(query_tokens, candidate)
            reranked.append(
                candidate.model_copy(
                    update={
                        "score": score,
                        "rerank_score": score,
                    }
                )
            )

        reranked.sort(
            key=lambda item: (
                -item.score,
                -(item.rrf_score if item.rrf_score is not None else 0.0),
                item.chunk.chunk_id,
            )
        )
        return reranked[:top_k]


class BGEReranker(BaseReranker):
    """Production Cross-Encoder reranker using BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker

            use_fp16 = self._device.lower() not in ("cpu", "mps")
            self._model = FlagReranker(
                self._model_name,
                use_fp16=use_fp16,
                device=self._device,
            )
            return self._model
        except Exception as exc:
            raise RetrievalError(
                f"FlagReranker model '{self._model_name}' could not be loaded: {exc}"
            ) from exc

    def _rerank_sync(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        model = self._load_model()
        pairs = [[query, c.chunk.content] for c in candidates]
        try:
            raw_scores = model.compute_score(pairs, normalize=True)
        except TypeError:
            raw_scores = model.compute_score(pairs)

        if isinstance(raw_scores, (int, float)):
            score_list = [float(raw_scores)]
        else:
            score_list = [float(x) for x in raw_scores]

        reranked: list[ScoredChunk] = []
        for candidate, raw_val in zip(candidates, score_list, strict=True):
            norm_score = (
                raw_val if 0.0 <= raw_val <= 1.0 else _sigmoid(raw_val)
            )
            clamped_score = round(max(0.0, min(1.0, norm_score)), 6)
            reranked.append(
                candidate.model_copy(
                    update={
                        "score": clamped_score,
                        "rerank_score": clamped_score,
                    }
                )
            )

        reranked.sort(
            key=lambda item: (
                -item.score,
                -(item.rrf_score if item.rrf_score is not None else 0.0),
                item.chunk.chunk_id,
            )
        )
        return reranked[:top_k]

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int = RERANK_TOP_K_DEFAULT,
    ) -> list[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []
        return await asyncio.to_thread(
            self._rerank_sync,
            query,
            list(candidates),
            top_k,
        )


class HybridRetrievalResult(BaseModel):
    """Output of the hybrid retrieval + RRF + cross-encoder reranking pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunks: list[ScoredChunk] = Field(default_factory=list)
    retrieval_ms: float = Field(ge=0.0)
    rerank_ms: float = Field(ge=0.0)


class HybridRetriever:
    """Executes parallel Qdrant + BM25s search, RRF fusion (k=60), and BGE reranking."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._dense = dense_retriever or DenseRetriever(settings=self._settings)
        self._sparse = sparse_retriever or SparseRetriever(settings=self._settings)

        if reranker is not None:
            self._reranker = reranker
        elif (
            self._settings.environment == "testing"
            or importlib.util.find_spec("FlagEmbedding") is None
        ):
            self._reranker = MockReranker()
        else:
            self._reranker = BGEReranker(
                model_name=self._settings.reranker_model_name,
                device=self._settings.reranker_device,
            )

    @property
    def reranker(self) -> BaseReranker:
        """The active cross-encoder reranker."""
        return self._reranker

    async def _rerank_with_fallback(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int,
    ) -> list[ScoredChunk]:
        try:
            return await self._reranker.rerank(query, candidates, top_k=top_k)
        except Exception as exc:
            if isinstance(self._reranker, BGEReranker):
                logger.warning(
                    "BGEReranker failed at runtime; falling back to MockReranker",
                    extra={"error": str(exc)},
                )
                self._reranker = MockReranker()
                return await self._reranker.rerank(query, candidates, top_k=top_k)
            raise RetrievalError(f"Cross-encoder reranking failed: {exc}") from exc

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = RERANK_TOP_K_DEFAULT,
        dense_top_k: int = DENSE_TOP_K_DEFAULT,
        sparse_top_k: int = SPARSE_TOP_K_DEFAULT,
        rrf_k: int = RRF_K_DEFAULT,
        rrf_top_n: int = RRF_TOP_N_DEFAULT,
        modality_filter: ChunkModality | str | None = None,
        doc_name: str | None = None,
    ) -> HybridRetrievalResult:
        """Run concurrent dense + sparse search, RRF fusion, and Top-K reranking."""
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return HybridRetrievalResult(chunks=[], retrieval_ms=0.0, rerank_ms=0.0)

        modality = normalize_modality_filter(modality_filter)

        t0 = time.perf_counter()
        dense_hits, sparse_hits = await asyncio.gather(
            self._dense.search(
                cleaned_query,
                top_k=dense_top_k,
                modality_filter=modality,
                doc_name=doc_name,
            ),
            self._sparse.search(
                cleaned_query,
                top_k=sparse_top_k,
                modality_filter=modality,
                doc_name=doc_name,
            ),
        )
        candidates = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=rrf_k,
            top_n=max(rrf_top_n, top_k),
        )
        retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        t1 = time.perf_counter()
        reranked = await self._rerank_with_fallback(
            cleaned_query,
            candidates,
            top_k=top_k,
        )
        rerank_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        logger.info(
            "Completed hybrid retrieval and reranking",
            extra={
                "dense_hits": len(dense_hits),
                "sparse_hits": len(sparse_hits),
                "rrf_candidates": len(candidates),
                "returned_chunks": len(reranked),
                "modality_filter": modality.value if modality else None,
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
            },
        )
        return HybridRetrievalResult(
            chunks=reranked,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int = RERANK_TOP_K_DEFAULT,
        modality_filter: ChunkModality | str | None = None,
        doc_name: str | None = None,
    ) -> list[ScoredChunk]:
        """Convenience method returning the reranked list of ScoredChunk objects."""
        result = await self.retrieve(
            query,
            top_k=top_k,
            modality_filter=modality_filter,
            doc_name=doc_name,
        )
        return result.chunks

    async def close(self) -> None:
        """Close underlying retriever connections."""
        await self._dense.close()
