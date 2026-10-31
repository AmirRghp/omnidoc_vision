"""Redis and in-memory fallback caching service for multimodal query responses."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from typing import Any, Final

from src.common.config import Settings, get_settings
from src.common.logger import get_logger
from src.ingestion.indexer import normalize_bilingual_text
from src.schemas.api import MultimodalQueryResponse
from src.schemas.document import ChunkModality

logger = get_logger(__name__)

CACHE_KEY_PREFIX: Final[str] = "omnidoc_vision:v1:query:"
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600


def build_query_cache_key(
    query: str,
    modality_filter: ChunkModality | str | None = None,
    *,
    force_visual_route: bool = False,
    top_k: int = 5,
) -> str:
    """Build deterministic SHA-256 Redis key for a normalized multimodal query."""
    normalized_query = normalize_bilingual_text(query)
    if isinstance(modality_filter, ChunkModality):
        modality_str = modality_filter.value
    elif isinstance(modality_filter, str) and modality_filter.strip():
        modality_str = modality_filter.strip().lower()
    else:
        modality_str = "all"

    has_custom_opts = force_visual_route or top_k != 5
    suffix = f"|force={int(force_visual_route)}|k={top_k}" if has_custom_opts else ""
    raw_material = f"{normalized_query}:{modality_str}{suffix}"
    digest = hashlib.sha256(raw_material.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


class RedisCacheService:
    """Asynchronous semantic/exact response cache backed by Redis with safe fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        redis_client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ttl_seconds = ttl_seconds
        self._redis_client: Any | None = redis_client
        self._memory_cache: dict[str, tuple[float, str]] = {}

    def _get_or_init_redis(self) -> Any | None:
        if self._redis_client is not None:
            return self._redis_client
        if self._settings.environment == "testing":
            return None
        try:
            import redis.asyncio as aioredis  # type: ignore[import-not-found]

            self._redis_client = aioredis.from_url(
                self._settings.redis_url,
                socket_timeout=0.5,
                decode_responses=True,
            )
            return self._redis_client
        except Exception:
            return None

    async def get(self, key: str) -> MultimodalQueryResponse | None:
        """Retrieve cached MultimodalQueryResponse by key or return None on miss."""
        client = self._get_or_init_redis()
        if client is not None:
            try:
                raw_payload = await asyncio.wait_for(client.get(key), timeout=0.5)
                if isinstance(raw_payload, (str, bytes)):
                    text = (
                        raw_payload.decode("utf-8")
                        if isinstance(raw_payload, bytes)
                        else raw_payload
                    )
                    return MultimodalQueryResponse.model_validate_json(text)
            except Exception as exc:
                logger.debug("Redis cache get fallback for key %s: %s", key, exc)

        entry = self._memory_cache.get(key)
        if entry is None:
            return None
        expires_at, serialized = entry
        if time.monotonic() >= expires_at:
            self._memory_cache.pop(key, None)
            return None
        try:
            return MultimodalQueryResponse.model_validate_json(serialized)
        except Exception:
            self._memory_cache.pop(key, None)
            return None

    async def set(
        self,
        key: str,
        response: MultimodalQueryResponse,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store MultimodalQueryResponse in Redis and fallback memory store with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        serialized = response.model_dump_json()
        self._memory_cache[key] = (time.monotonic() + ttl, serialized)

        client = self._get_or_init_redis()
        if client is not None:
            try:
                await asyncio.wait_for(client.set(key, serialized, ex=ttl), timeout=0.5)
            except Exception as exc:
                logger.debug("Redis cache set fallback for key %s: %s", key, exc)

    async def invalidate_all(self) -> None:
        """Invalidate all cached query responses after document ingestion."""
        self._memory_cache.clear()
        client = self._get_or_init_redis()
        if client is not None:
            try:
                keys = await asyncio.wait_for(
                    client.keys(f"{CACHE_KEY_PREFIX}*"),
                    timeout=0.5,
                )
                if keys:
                    await asyncio.wait_for(client.delete(*keys), timeout=0.5)
            except Exception as exc:
                logger.debug("Redis cache invalidation fallback: %s", exc)

    async def close(self) -> None:
        """Close underlying Redis connection pool if initialized."""
        self._memory_cache.clear()
        if self._redis_client is not None:
            with contextlib.suppress(Exception):
                await self._redis_client.aclose()
            self._redis_client = None
