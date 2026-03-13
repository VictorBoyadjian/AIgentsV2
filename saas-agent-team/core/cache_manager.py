"""
Anthropic Prompt Caching manager for reducing LLM input costs by up to 50%.

Manages cache_control markers on stable message content (system prompts,
project context, code files) so that repeated tokens are served from
Anthropic's server-side cache at heavily discounted rates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog

from core.config import AgentRole, get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompts per agent role (stable content → always cached)
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    AgentRole.ARCHITECT: (
        "Tu es un Architecte Logiciel Senior, expert SaaS, maîtrisant les patterns "
        "(CQRS, Event Sourcing, DDD, microservices vs monolithe modulaire). "
        "Tu produis des documents d'architecture exhaustifs en JSON structuré. "
        "Tu justifies chaque choix technique par un critère de scalabilité ou de coût."
    ),
    AgentRole.PM: (
        "Tu es un Product Manager Senior expert SaaS B2B, maîtrisant RICE, "
        "Jobs-to-be-Done et Shape Up. Tu génères des PRD actionnables et des "
        "user stories avec critères d'acceptance Gherkin."
    ),
    AgentRole.DEV: (
        "Tu es un Senior Full-Stack Engineer (Python/FastAPI + TypeScript/Next.js). "
        "Tu écris du code production-ready du premier coup, propre, typé, testé. "
        "Tu suis TDD : tests d'abord, implémentation ensuite. "
        "Tu n'utilises jamais de placeholder ni de code incomplet."
    ),
    AgentRole.QA: (
        "Tu es un QA Engineer Senior spécialisé en test automation. "
        "Tu écris des plans de test exhaustifs, des tests unitaires, d'intégration et E2E. "
        "Tu maîtrises pytest, Playwright, et les stratégies de couverture de code."
    ),
    AgentRole.SECURITY: (
        "Tu es un Security Engineer Senior spécialisé en AppSec. "
        "Tu réalises des audits de code OWASP Top 10, analyses de dépendances, "
        "reviews d'authentification/autorisation, et génères des checklists de sécurité."
    ),
    AgentRole.DEVOPS: (
        "Tu es un DevOps/SRE Senior expert en CI/CD, Docker, Kubernetes, Terraform, "
        "et plateformes cloud (AWS, GCP, Railway). Tu génères des configurations "
        "d'infrastructure production-ready avec monitoring intégré."
    ),
    AgentRole.RESEARCH: (
        "Tu es un Research Analyst Senior. Tu effectues des recherches web approfondies, "
        "analyses concurrentielles, benchmarks technologiques, et synthèses documentaires. "
        "Tu cites toujours tes sources et quantifies tes recommandations."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    """Prompt caching statistics."""

    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_regular_input_tokens: int = 0
    estimated_savings_usd: float = 0.0
    cache_hit_ratio: float = 0.0
    calls_with_cache: int = 0
    total_calls: int = 0


@dataclass
class CacheEntry:
    """Metadata about a cached content block."""

    content_hash: str
    agent_role: str
    content_type: str  # "system_prompt", "project_context", "code_file"
    token_count_estimate: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------
class CacheManager:
    """
    Manages Anthropic prompt caching to reduce input token costs by ~50%.

    Caching strategy:
    - ALWAYS cache: agent system prompt (role, rules, backstory)
    - ALWAYS cache: project context (architecture doc, PRD, tech stack)
    - ALWAYS cache: existing code files (for review/modification)
    - NEVER cache: current task/question (variable part)

    The Anthropic API caches content blocks marked with
    ``cache_control: {"type": "ephemeral"}``. Cached content persists for
    ~5 minutes on Anthropic's servers and is read at 90% discount.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        settings = get_settings()
        self._redis_url = redis_url or settings.redis.redis_url
        self._redis: aioredis.Redis | None = None
        self._stats = CacheStats()
        self._cache_ttl = settings.budget.cache_ttl_seconds

    async def initialize(self) -> None:
        """Connect to Redis for local cache metadata tracking."""
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        logger.info("cache_manager.initialized")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    # ------------------------------------------------------------------
    # Core: build cached messages
    # ------------------------------------------------------------------
    def build_cached_messages(
        self,
        system_prompt: str,
        project_context: str | None = None,
        existing_code: str | None = None,
        current_task: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Build messages with Anthropic cache_control markers.

        Returns:
            A tuple of (system_messages, user_messages) formatted for the
            Anthropic Messages API with cache_control on stable content.

        The system parameter uses content blocks with cache_control.
        The user messages combine cached context with the uncached task.
        """
        # --- System message (always cached) ---
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # --- User message content blocks ---
        user_content_blocks: list[dict[str, Any]] = []

        # Project context (cached if present)
        if project_context:
            user_content_blocks.append({
                "type": "text",
                "text": f"<project_context>\n{project_context}\n</project_context>",
                "cache_control": {"type": "ephemeral"},
            })

        # Existing code (cached if present)
        if existing_code:
            user_content_blocks.append({
                "type": "text",
                "text": f"<existing_code>\n{existing_code}\n</existing_code>",
                "cache_control": {"type": "ephemeral"},
            })

        # Current task (NOT cached — variable part)
        if current_task:
            user_content_blocks.append({
                "type": "text",
                "text": current_task,
            })

        user_messages: list[dict[str, Any]] = []
        if user_content_blocks:
            user_messages.append({
                "role": "user",
                "content": user_content_blocks,
            })

        return system_blocks, user_messages

    def build_cached_messages_litellm(
        self,
        system_prompt: str,
        project_context: str | None = None,
        existing_code: str | None = None,
        current_task: str = "",
    ) -> list[dict[str, Any]]:
        """
        Build messages for LiteLLM with Anthropic cache control.

        LiteLLM passes cache_control through to Anthropic when the model
        is an Anthropic model. Returns a flat list of messages suitable
        for ``litellm.acompletion(messages=...)``.
        """
        messages: list[dict[str, Any]] = []

        # System message with cache control
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        })

        # User message with cached context + uncached task
        user_content: list[dict[str, Any]] = []

        if project_context:
            user_content.append({
                "type": "text",
                "text": f"<project_context>\n{project_context}\n</project_context>",
                "cache_control": {"type": "ephemeral"},
            })

        if existing_code:
            user_content.append({
                "type": "text",
                "text": f"<existing_code>\n{existing_code}\n</existing_code>",
                "cache_control": {"type": "ephemeral"},
            })

        if current_task:
            user_content.append({
                "type": "text",
                "text": current_task,
            })

        if user_content:
            messages.append({
                "role": "user",
                "content": user_content,
            })

        return messages

    # ------------------------------------------------------------------
    # System prompt retrieval
    # ------------------------------------------------------------------
    def get_system_prompt(self, agent_role: str | AgentRole) -> str:
        """Get the system prompt for an agent role."""
        role_key = agent_role.value if isinstance(agent_role, AgentRole) else agent_role
        return AGENT_SYSTEM_PROMPTS.get(role_key, AGENT_SYSTEM_PROMPTS.get(AgentRole.DEV, ""))

    # ------------------------------------------------------------------
    # Cache warmup
    # ------------------------------------------------------------------
    async def warm_cache(self, agent_role: str, project_context: str) -> None:
        """
        Pre-warm Anthropic's cache by sending a minimal request with
        the system prompt and project context marked for caching.

        This invests a small cost upfront (cache write) so that all
        subsequent calls benefit from the ~90% discount on cached tokens.
        """
        import litellm

        system_prompt = self.get_system_prompt(agent_role)
        messages = self.build_cached_messages_litellm(
            system_prompt=system_prompt,
            project_context=project_context,
            current_task="Acknowledge receipt of the project context. Reply with 'OK' only.",
        )

        settings = get_settings()
        model = settings.llm.get_model_for_role(AgentRole(agent_role))

        # Only warm cache for Anthropic models
        if not model.startswith("claude"):
            logger.info("cache_manager.skip_warmup_non_anthropic", model=model)
            return

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
            usage = response.usage
            logger.info(
                "cache_manager.cache_warmed",
                agent_role=agent_role,
                model=model,
                cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0),
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            )
        except Exception as exc:
            logger.warning(
                "cache_manager.warmup_failed",
                agent_role=agent_role,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------
    def update_stats(
        self,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        regular_input_tokens: int,
        model: str,
    ) -> None:
        """Update running cache statistics after an API call."""
        self._stats.total_calls += 1
        self._stats.total_cache_write_tokens += cache_creation_tokens
        self._stats.total_cache_read_tokens += cache_read_tokens
        self._stats.total_regular_input_tokens += regular_input_tokens

        if cache_read_tokens > 0:
            self._stats.calls_with_cache += 1

        total_input = (
            self._stats.total_cache_write_tokens
            + self._stats.total_cache_read_tokens
            + self._stats.total_regular_input_tokens
        )
        if total_input > 0:
            self._stats.cache_hit_ratio = self._stats.total_cache_read_tokens / total_input

        # Estimate savings (cache_read at 10% of normal price = 90% savings)
        from core.cost_tracker import PRICING_TABLE

        pricing = PRICING_TABLE.get(model)
        if pricing:
            normal_cost = cache_read_tokens * pricing.input_per_million / 1_000_000
            actual_cost = cache_read_tokens * pricing.cache_read_per_million / 1_000_000
            self._stats.estimated_savings_usd += normal_cost - actual_cost

    def get_cache_stats(self) -> CacheStats:
        """Return current cache statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Redis-backed content hash tracking
    # ------------------------------------------------------------------
    async def _store_cache_entry(self, entry: CacheEntry) -> None:
        """Store cache entry metadata in Redis."""
        if not self._redis:
            return
        key = f"cache_entry:{entry.content_hash}"
        await self._redis.setex(
            key,
            self._cache_ttl,
            json.dumps({
                "agent_role": entry.agent_role,
                "content_type": entry.content_type,
                "token_count_estimate": entry.token_count_estimate,
                "created_at": entry.created_at.isoformat(),
            }),
        )

    async def is_content_cached(self, content: str) -> bool:
        """Check if content is likely still in Anthropic's cache based on local tracking."""
        if not self._redis:
            return False
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return await self._redis.exists(f"cache_entry:{content_hash}") > 0

    async def mark_as_cached(
        self,
        content: str,
        agent_role: str,
        content_type: str,
    ) -> None:
        """Mark content as having been sent with cache_control (locally tracked)."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        entry = CacheEntry(
            content_hash=content_hash,
            agent_role=agent_role,
            content_type=content_type,
            token_count_estimate=len(content) // 4,  # rough estimate
        )
        await self._store_cache_entry(entry)
