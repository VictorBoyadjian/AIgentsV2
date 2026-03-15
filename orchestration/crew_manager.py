"""
Crew manager for orchestrating the agent team.

Provides a high-level interface for managing agent lifecycle,
project initialization, and workflow execution.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.architect_agent import ArchitectAgent
from agents.dev_agent import DevAgent
from agents.devops_agent import DevOpsAgent
from agents.pm_agent import PMAgent
from agents.qa_agent import QAAgent
from agents.research_agent import ResearchAgent
from agents.security_agent import SecurityAgent
from core.batch_processor import BatchProcessor
from core.cache_manager import CacheManager
from core.config import get_settings
from core.cost_tracker import CostTracker
from core.llm_router import LLMRouter
from memory.cache import MemoryCache
from memory.database import Database
from memory.vector_store import VectorStore
from orchestration.workflow_graph import WorkflowGraph, WorkflowState

logger = structlog.get_logger(__name__)


class CrewManager:
    """
    Manages the full agent team lifecycle.

    Responsibilities:
    - Initialize and configure all agents
    - Set up shared infrastructure (DB, cache, vector store)
    - Provide project creation and management
    - Execute development workflows
    - Handle budget and cost management
    """

    def __init__(self) -> None:
        self._settings = get_settings()

        # Shared infrastructure
        self._llm_router = LLMRouter()
        self._cost_tracker = CostTracker()
        self._cache_manager = CacheManager()
        self._batch_processor = BatchProcessor()
        self._database = Database()
        self._memory_cache = MemoryCache()
        self._vector_store = VectorStore()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all infrastructure components."""
        await self._cost_tracker.initialize()
        await self._cache_manager.initialize()
        await self._batch_processor.initialize()
        await self._database.initialize()
        await self._memory_cache.initialize()
        await self._vector_store.initialize()
        self._initialized = True
        logger.info("crew_manager.initialized")

    async def shutdown(self) -> None:
        """Gracefully shut down all components."""
        await self._cache_manager.close()
        await self._memory_cache.close()
        await self._vector_store.close()
        await self._database.close()
        logger.info("crew_manager.shut_down")

    def _create_agents(self, project_id: str) -> dict[str, Any]:
        """Create all agents with shared infrastructure."""
        kwargs: dict[str, Any] = {
            "llm_router": self._llm_router,
            "cache_manager": self._cache_manager,
            "cost_tracker": self._cost_tracker,
            "batch_processor": self._batch_processor,
        }
        return {
            "architect": ArchitectAgent(project_id=project_id, **kwargs),
            "pm": PMAgent(project_id=project_id, **kwargs),
            "dev": DevAgent(project_id=project_id, **kwargs),
            "qa": QAAgent(project_id=project_id, **kwargs),
            "security": SecurityAgent(project_id=project_id, **kwargs),
            "devops": DevOpsAgent(project_id=project_id, **kwargs),
            "research": ResearchAgent(project_id=project_id, **kwargs),
        }

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------
    async def create_project(
        self,
        name: str,
        description: str = "",
        budget: float = 200.0,
    ) -> str:
        """Create a new project and return its ID."""
        project = await self._database.create_project(
            name=name,
            description=description,
            budget=budget,
        )
        logger.info("crew_manager.project_created", id=project.id, name=name)
        return project.id

    async def run_workflow(
        self,
        project_id: str,
        product_idea: str,
        target_market: str = "B2B SaaS",
        human_callback: Any = None,
        github_repo: str | None = None,
    ) -> WorkflowState:
        """Run the full development workflow for a project."""
        if not self._initialized:
            await self.initialize()

        workflow = WorkflowGraph(project_id=project_id)
        if human_callback:
            workflow.set_human_callback(human_callback)

        state = WorkflowState(
            project_id=project_id,
            product_idea=product_idea,
            target_market=target_market,
        )

        # Use an existing GitHub repo if provided
        if github_repo:
            state.github_repo_name = github_repo
            state.github_repo_url = f"https://github.com/{github_repo}"

        result = await workflow.run(state)

        # Store result in cache
        await self._memory_cache.store_project_context(
            project_id=project_id,
            context_type="workflow_result",
            content=result.cost_report,
        )

        return result

    # ------------------------------------------------------------------
    # Cost management
    # ------------------------------------------------------------------
    async def get_project_costs(self, project_id: str) -> dict[str, Any]:
        """Get cost summary for a project."""
        daily = await self._cost_tracker.get_daily_cost(project_id)
        projection = await self._cost_tracker.get_monthly_projection(project_id)
        tips = await self._cost_tracker.get_optimization_suggestions(project_id)

        return {
            "daily": {
                "total_cost_usd": daily.total_cost_usd,
                "total_calls": daily.total_calls,
                "cost_by_agent": daily.cost_by_agent,
                "cache_savings_usd": daily.cache_savings_usd,
                "batch_savings_usd": daily.batch_savings_usd,
            },
            "projection": {
                "current_month_cost": projection.current_month_cost,
                "projected_month_cost": projection.projected_month_cost,
                "daily_average": projection.daily_average,
                "budget_usd": projection.budget_usd,
                "over_budget": projection.projected_over_budget,
            },
            "optimization_tips": [
                {
                    "category": tip.category,
                    "description": tip.description,
                    "estimated_savings": tip.estimated_monthly_savings_usd,
                }
                for tip in tips
            ],
        }
