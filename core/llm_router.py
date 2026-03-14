"""
Intelligent LLM router with per-agent model selection, complexity-based
overrides, automatic fallback, and rate limiting.

The routing table maps each agent role to the optimal model for its
task profile, balancing quality against cost. Complexity overrides
automatically downgrade to cheaper models for simple tasks and upgrade
to the most capable model for critical decisions.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import litellm
import structlog

from core.config import AgentRole, TaskComplexity, get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Routing table (model, fallback, justification)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteEntry:
    """A routing table entry for an agent role."""

    primary: str
    fallback: str
    justification: str


ROUTING_TABLE: dict[str, RouteEntry] = {
    "architect": RouteEntry("claude-opus-4", "claude-sonnet-4-5", "raisonnement max requis"),
    "pm": RouteEntry("claude-sonnet-4-5", "gpt-4o", "rédaction + specs"),
    "dev": RouteEntry("claude-sonnet-4-5", "gpt-4o", "meilleur code/prix"),
    "qa": RouteEntry("claude-haiku-4-5", "gpt-4o-mini", "validation simple"),
    "security": RouteEntry("claude-sonnet-4-5", "gpt-4o", "analyse code critique"),
    "devops": RouteEntry("claude-haiku-4-5", "gpt-4o-mini", "templates + scripts"),
    "research": RouteEntry("gpt-4o", "claude-sonnet-4-5", "web search natif OpenAI"),
    "fallback": RouteEntry("gpt-4o-mini", "claude-haiku-4-5", "ultra-cheap pour retry"),
}

TASK_COMPLEXITY_OVERRIDE: dict[TaskComplexity, str | None] = {
    TaskComplexity.SIMPLE: "claude-haiku-4-5",
    TaskComplexity.MEDIUM: None,  # use agent default
    TaskComplexity.COMPLEX: None,  # use agent default
    TaskComplexity.CRITICAL: "claude-opus-4",  # force upgrade
}

# Quality scores for model selection (higher = better, 0-1 scale)
MODEL_QUALITY_SCORES: dict[str, float] = {
    "claude-opus-4": 0.98,
    "claude-sonnet-4-5": 0.90,
    "gpt-4o": 0.88,
    "claude-haiku-4-5": 0.75,
    "gpt-4o-mini": 0.65,
}

# Approximate cost per 1K tokens (input+output averaged) for ranking
MODEL_COST_PER_1K: dict[str, float] = {
    "claude-opus-4": 0.045,
    "claude-sonnet-4-5": 0.009,
    "gpt-4o": 0.00625,
    "claude-haiku-4-5": 0.003,
    "gpt-4o-mini": 0.000375,
}

# Rate limits: max concurrent requests per model
MODEL_CONCURRENCY_LIMITS: dict[str, int] = {
    "claude-opus-4": 5,
    "claude-sonnet-4-5": 20,
    "claude-haiku-4-5": 50,
    "gpt-4o": 30,
    "gpt-4o-mini": 100,
}


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------
class LLMRouter:
    """
    Intelligent LLM router that selects the optimal model per agent and task.

    Features:
    - Per-agent model routing via ROUTING_TABLE
    - Complexity-based override (simple→cheap, critical→best)
    - Automatic fallback on 429/500/timeout errors
    - Rate limiting per model with asyncio.Semaphore
    - Cost estimation utilities
    """

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {
            model: asyncio.Semaphore(limit)
            for model, limit in MODEL_CONCURRENCY_LIMITS.items()
        }
        self._failure_counts: dict[str, int] = {}
        self._last_failure_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def resolve_model(
        self,
        agent_role: str,
        task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
    ) -> str:
        """
        Resolve the best model for a given agent and task complexity.

        Priority:
        1. Complexity override (SIMPLE → haiku, CRITICAL → opus)
        2. Agent routing table entry (primary model)
        3. Global fallback (gpt-4o-mini)
        """
        # Check complexity override first
        override = TASK_COMPLEXITY_OVERRIDE.get(task_complexity)
        if override is not None:
            return override

        # Agent routing table
        route = ROUTING_TABLE.get(agent_role, ROUTING_TABLE["fallback"])
        return route.primary

    def get_fallback_model(self, agent_role: str) -> str:
        """Get the fallback model for an agent role."""
        route = ROUTING_TABLE.get(agent_role, ROUTING_TABLE["fallback"])
        return route.fallback

    def get_cheapest_capable_model(self, min_quality_score: float = 0.7) -> str:
        """Find the cheapest model that meets a minimum quality threshold."""
        eligible = [
            (model, cost)
            for model, cost in MODEL_COST_PER_1K.items()
            if MODEL_QUALITY_SCORES.get(model, 0) >= min_quality_score
        ]
        if not eligible:
            return "gpt-4o-mini"
        eligible.sort(key=lambda x: x[1])
        return eligible[0][0]

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost in USD for a given token count and model."""
        from core.cost_tracker import PRICING_TABLE

        pricing = PRICING_TABLE.get(model)
        if not pricing:
            return 0.0
        return (
            prompt_tokens * pricing.input_per_million / 1_000_000
            + completion_tokens * pricing.output_per_million / 1_000_000
        )

    # ------------------------------------------------------------------
    # LLM call with routing, fallback, and rate limiting
    # ------------------------------------------------------------------
    async def call_llm(
        self,
        agent_role: str,
        messages: list[dict[str, Any]],
        task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        force_model: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """
        Call an LLM with intelligent routing, rate limiting, and fallback.

        Returns:
            Tuple of (response_text, usage_info) where usage_info contains
            token counts including cache metrics.
        """
        model = force_model or self.resolve_model(agent_role, task_complexity)

        # Try primary model, then fallback
        models_to_try = [model]
        fallback = self.get_fallback_model(agent_role)
        if fallback != model:
            models_to_try.append(fallback)
        # Ultimate fallback
        if "gpt-4o-mini" not in models_to_try:
            models_to_try.append("gpt-4o-mini")

        last_error: Exception | None = None

        for attempt_model in models_to_try:
            try:
                return await self._call_with_rate_limit(
                    model=attempt_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
                self._record_failure(attempt_model)
                logger.warning(
                    "llm_router.model_failed",
                    model=attempt_model,
                    agent_role=agent_role,
                    error=str(exc),
                    fallback=models_to_try[models_to_try.index(attempt_model) + 1]
                    if models_to_try.index(attempt_model) + 1 < len(models_to_try)
                    else "none",
                )

        raise RuntimeError(
            f"All models failed for agent '{agent_role}': {last_error}"
        ) from last_error

    async def _call_with_rate_limit(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Call LLM with rate limiting via semaphore."""
        semaphore = self._semaphores.get(model)
        if semaphore is None:
            semaphore = asyncio.Semaphore(10)
            self._semaphores[model] = semaphore

        async with semaphore:
            settings = get_settings()

            # Set API keys via litellm
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=(
                    settings.llm.anthropic_api_key
                    if model.startswith("claude")
                    else settings.llm.openai_api_key
                ),
                **kwargs,
            )

            # Extract response text
            content = response.choices[0].message.content or ""

            # Extract usage info including cache metrics
            usage = response.usage
            usage_info: dict[str, Any] = {
                "model": model,
                "input_tokens": getattr(usage, "prompt_tokens", 0),
                "output_tokens": getattr(usage, "completion_tokens", 0),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

            logger.debug(
                "llm_router.call_success",
                model=model,
                input_tokens=usage_info["input_tokens"],
                output_tokens=usage_info["output_tokens"],
            )

            return content, usage_info

    # ------------------------------------------------------------------
    # Failure tracking for circuit breaker
    # ------------------------------------------------------------------
    def _record_failure(self, model: str) -> None:
        """Record a model failure for circuit breaker logic."""
        now = time.time()
        self._failure_counts[model] = self._failure_counts.get(model, 0) + 1
        self._last_failure_time[model] = now

    def is_model_healthy(self, model: str, cooldown_seconds: float = 60.0) -> bool:
        """Check if a model is healthy (no recent failures / cooldown expired)."""
        count = self._failure_counts.get(model, 0)
        if count == 0:
            return True
        last_time = self._last_failure_time.get(model, 0.0)
        if time.time() - last_time > cooldown_seconds:
            # Reset after cooldown
            self._failure_counts[model] = 0
            return True
        return count < 3  # Allow up to 2 failures before considering unhealthy

    def reset_failures(self, model: str | None = None) -> None:
        """Reset failure counts, optionally for a specific model."""
        if model:
            self._failure_counts.pop(model, None)
            self._last_failure_time.pop(model, None)
        else:
            self._failure_counts.clear()
            self._last_failure_time.clear()

    # ------------------------------------------------------------------
    # Emergency mode
    # ------------------------------------------------------------------
    def get_emergency_model(self) -> str:
        """Return the emergency (cheapest) model configured in budget settings."""
        settings = get_settings()
        return settings.budget.emergency_model_override
