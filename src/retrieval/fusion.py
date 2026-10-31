"""Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from src.retrieval.dense import ScoredChunk
from src.schemas.document import MultimodalChunk

RRF_K_DEFAULT: Final[int] = 60
RRF_TOP_N_DEFAULT: Final[int] = 15


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = RRF_K_DEFAULT,
    top_n: int = RRF_TOP_N_DEFAULT,
) -> list[ScoredChunk]:
    """Fuse multiple ranked ScoredChunk lists using Reciprocal Rank Fusion: sum(1 / (k + rank))."""
    if k <= 0:
        raise ValueError("RRF parameter k must be positive")
    if top_n <= 0 or not ranked_lists:
        return []

    chunks_by_id: dict[str, MultimodalChunk] = {}
    rrf_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        rank = 0
        for item in ranked_list:
            chunk_id = item.chunk.chunk_id
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            rank += 1

            chunks_by_id[chunk_id] = item.chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / float(k + rank))

            if item.dense_score is not None:
                prev_dense = dense_scores.get(chunk_id)
                dense_scores[chunk_id] = (
                    max(prev_dense, item.dense_score)
                    if prev_dense is not None
                    else item.dense_score
                )
            if item.sparse_score is not None:
                prev_sparse = sparse_scores.get(chunk_id)
                sparse_scores[chunk_id] = (
                    max(prev_sparse, item.sparse_score)
                    if prev_sparse is not None
                    else item.sparse_score
                )

    fused: list[ScoredChunk] = []
    for chunk_id, chunk in chunks_by_id.items():
        score = rrf_scores[chunk_id]
        fused.append(
            ScoredChunk(
                chunk=chunk,
                score=score,
                dense_score=dense_scores.get(chunk_id),
                sparse_score=sparse_scores.get(chunk_id),
                rrf_score=score,
            )
        )

    fused.sort(
        key=lambda item: (
            -item.score,
            -(item.dense_score if item.dense_score is not None else float("-inf")),
            -(item.sparse_score if item.sparse_score is not None else float("-inf")),
            item.chunk.chunk_id,
        )
    )
    return fused[:top_n]
