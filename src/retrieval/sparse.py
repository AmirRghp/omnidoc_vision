"""Sparse keyword and metric search using BM25s with bilingual tokenization."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Final

import bm25s

from src.common.config import Settings, get_settings
from src.common.exceptions import RetrievalError
from src.common.logger import get_logger
from src.ingestion.indexer import tokenize_bilingual
from src.retrieval.dense import ScoredChunk, normalize_modality_filter
from src.schemas.document import ChunkModality, MultimodalChunk

logger = get_logger(__name__)

SPARSE_TOP_K_DEFAULT: Final[int] = 20


class SparseRetriever:
    """Asynchronous BM25s sparse retriever over local bilingual index artifacts."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        bm25_dir: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._bm25_dir = bm25_dir if bm25_dir is not None else Path(self._settings.bm25_index_dir)
        self._metadata_file = self._bm25_dir / "corpus_metadata.json"

    def _load_records_sync(self) -> list[dict[str, Any]]:
        if not self._metadata_file.exists():
            return []
        try:
            raw = self._metadata_file.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            return list(parsed) if isinstance(parsed, list) else []
        except Exception as exc:
            raise RetrievalError(
                f"Failed to read BM25 metadata from {self._metadata_file}: {exc}"
            ) from exc

    def _search_sync(
        self,
        query: str,
        *,
        top_k: int,
        modality: ChunkModality | None,
        doc_name: str | None,
    ) -> list[ScoredChunk]:
        records = self._load_records_sync()
        if not records:
            return []

        filtered_records: list[dict[str, Any]] = []
        for record in records:
            if modality is not None and record.get("modality") != modality.value:
                continue
            if doc_name is not None and doc_name.strip():
                if record.get("doc_name") != doc_name.strip():
                    continue
            filtered_records.append(record)

        if not filtered_records:
            return []

        try:
            corpus_texts = [str(r["content"]) for r in filtered_records]
            corpus_tokens = tokenize_bilingual(corpus_texts)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)

            query_tokens = tokenize_bilingual([query])
            k_limit = min(top_k, len(filtered_records))
            doc_indices, scores = retriever.retrieve(query_tokens, k=k_limit)
        except Exception as exc:
            raise RetrievalError(f"BM25s sparse retrieval failed: {exc}") from exc

        results: list[ScoredChunk] = []
        row_indices = doc_indices[0] if len(doc_indices) > 0 else []
        row_scores = scores[0] if len(scores) > 0 else []

        for idx_val, score_val in zip(row_indices, row_scores, strict=False):
            idx = int(idx_val)
            score = float(score_val)
            if idx < 0 or idx >= len(filtered_records):
                continue
            record = filtered_records[idx]
            chunk = MultimodalChunk.model_validate(record)
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    sparse_score=score,
                )
            )

        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:top_k]

    async def search(
        self,
        query: str,
        *,
        top_k: int = SPARSE_TOP_K_DEFAULT,
        modality_filter: ChunkModality | str | None = None,
        doc_name: str | None = None,
    ) -> list[ScoredChunk]:
        """Asynchronously search the BM25s index for the top_k matching chunks."""
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []

        modality = normalize_modality_filter(modality_filter)
        return await asyncio.to_thread(
            self._search_sync,
            cleaned_query,
            top_k=top_k,
            modality=modality,
            doc_name=doc_name,
        )
