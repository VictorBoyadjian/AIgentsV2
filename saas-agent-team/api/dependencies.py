"""
FastAPI dependency injection providers.

Provides singleton instances of core infrastructure components via
``lru_cache`` so that the same objects are reused across the application
lifetime.  Each getter can be overridden in tests via
``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

import structlog

from core.config import Settings, get_settings
from core.cost_tracker import CostTracker
from memory.cache import MemoryCache
from memory.database import Database
from orchestration.crew_manager import CrewManager
from orchestration.human_in_loop import HumanInTheLoop

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def get_app_settings() -> Settings:
    """Return the cached application-wide settings singleton.

    Delegates to ``core.config.get_settings`` which is itself ``@lru_cache``
    decorated.
    """
    return get_settings()


# ---------------------------------------------------------------------------
# CrewManager  (the "god object" that owns all agents + infra)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_crew_manager() -> CrewManager:
    """Return a singleton ``CrewManager``.

    The manager must be explicitly initialised during the application lifespan
    startup (``await manager.initialize()``).  After that every request handler
    receives the same ready-to-use instance.
    """
    return CrewManager()


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_cost_tracker() -> CostTracker:
    """Return a singleton ``CostTracker``.

    Shares the same budget config and database engine across all callers.
    Must be initialised during application startup.
    """
    return CostTracker()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_database() -> Database:
    """Return a singleton async ``Database`` manager.

    Must be initialised during application startup (``await db.initialize()``).
    """
    return Database()


# ---------------------------------------------------------------------------
# MemoryCache (Redis)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_cache() -> MemoryCache:
    """Return a singleton ``MemoryCache`` (Redis-backed).

    Must be initialised during application startup.
    """
    return MemoryCache()


# ---------------------------------------------------------------------------
# HumanInTheLoop
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_human_in_loop() -> HumanInTheLoop:
    """Return a singleton ``HumanInTheLoop`` checkpoint manager."""
    return HumanInTheLoop()
