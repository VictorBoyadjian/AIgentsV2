"""
Redis-based cache for short-term memory and session state.

Provides fast key-value caching for agent context, conversation
history, and intermediate computation results.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)


class MemoryCache:
    """
    Redis-backed cache for agent short-term memory.

    Stores:
    - Agent conversation history per session
    - Intermediate computation results
    - Project context snapshots
    - Task results awaiting collection
    """

    def __init__(self, redis_url: str | None = None) -> None:
        settings = get_settings()
        self._redis_url = redis_url or settings.redis.redis_url
        self._redis: aioredis.Redis | None = None
        self._default_ttl = settings.budget.cache_ttl_seconds

    async def initialize(self) -> None:
        """Connect to Redis."""
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        logger.info("memory_cache.initialized")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    def _get_redis(self) -> aioredis.Redis:
        """Get Redis client, raising if not initialized."""
        if not self._redis:
            raise RuntimeError("MemoryCache not initialized. Call initialize() first.")
        return self._redis

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    async def get(self, key: str) -> str | None:
        """Get a value from cache."""
        return await self._get_redis().get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Set a value in cache with optional TTL."""
        effective_ttl = ttl or self._default_ttl
        await self._get_redis().setex(key, effective_ttl, value)

    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        await self._get_redis().delete(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return await self._get_redis().exists(key) > 0

    # ------------------------------------------------------------------
    # JSON operations
    # ------------------------------------------------------------------
    async def get_json(self, key: str) -> Any | None:
        """Get and deserialize a JSON value from cache."""
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Serialize and set a JSON value in cache."""
        await self.set(key, json.dumps(value, default=str), ttl)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------
    async def get_conversation(self, session_id: str) -> list[dict[str, str]]:
        """Get conversation history for a session."""
        key = f"conv:{session_id}"
        data = await self.get(key)
        if data:
            return json.loads(data)
        return []

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        max_messages: int = 50,
    ) -> None:
        """Append a message to conversation history."""
        key = f"conv:{session_id}"
        history = await self.get_conversation(session_id)
        history.append({"role": role, "content": content})

        # Trim to max_messages (keep system prompt + latest messages)
        if len(history) > max_messages:
            history = history[:1] + history[-(max_messages - 1):]

        await self.set(key, json.dumps(history), ttl=7200)  # 2h TTL

    async def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        await self.delete(f"conv:{session_id}")

    # ------------------------------------------------------------------
    # Project context
    # ------------------------------------------------------------------
    async def store_project_context(
        self,
        project_id: str,
        context_type: str,
        content: str,
    ) -> None:
        """Store project context (architecture, PRD, etc.)."""
        key = f"project:{project_id}:{context_type}"
        await self.set(key, content, ttl=86400)  # 24h TTL

    async def get_project_context(
        self, project_id: str, context_type: str
    ) -> str | None:
        """Retrieve project context."""
        key = f"project:{project_id}:{context_type}"
        return await self.get(key)

    async def get_all_project_context(self, project_id: str) -> dict[str, str]:
        """Get all context types for a project."""
        redis = self._get_redis()
        pattern = f"project:{project_id}:*"
        context: dict[str, str] = {}

        async for key in redis.scan_iter(match=pattern):
            context_type = key.split(":")[-1]
            value = await redis.get(key)
            if value:
                context[context_type] = value

        return context

    # ------------------------------------------------------------------
    # Task results
    # ------------------------------------------------------------------
    async def store_task_result(
        self, task_id: str, result: dict[str, Any]
    ) -> None:
        """Store a task result for later collection."""
        key = f"task_result:{task_id}"
        await self.set_json(key, result, ttl=86400)

    async def get_task_result(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a stored task result."""
        return await self.get_json(f"task_result:{task_id}")
