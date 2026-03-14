"""
Shared pytest fixtures for the SaaS Agent Team test suite.

Provides mock instances of all core services and agents to enable
unit testing without external dependencies (no DB, no Redis, no LLM APIs).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import (
    AgentRole,
    BatchPriority,
    BudgetConfig,
    LLMConfig,
    Settings,
    TaskComplexity,
)
from core.cost_tracker import CallCost, CostTracker, PRICING_TABLE, ModelPricing
from core.cache_manager import CacheManager
from core.batch_processor import BatchProcessor
from core.llm_router import LLMRouter


# ---------------------------------------------------------------------------
# Event loop fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def budget_config() -> BudgetConfig:
    """Test budget configuration with low limits for easy threshold testing."""
    return BudgetConfig(
        monthly_budget_usd=100.0,
        daily_budget_usd=5.0,
        warning_threshold=0.70,
        critical_threshold=0.90,
        emergency_model_override="gpt-4o-mini",
        enable_batch_by_default=True,
        cache_ttl_seconds=60,
    )


@pytest.fixture
def llm_config() -> LLMConfig:
    """Test LLM configuration with fake API keys."""
    return LLMConfig(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        architect_model="claude-opus-4",
        pm_model="claude-sonnet-4-5",
        dev_model="claude-sonnet-4-5",
        qa_model="claude-haiku-4-5",
        security_model="claude-sonnet-4-5",
        devops_model="claude-haiku-4-5",
        research_model="gpt-4o",
        fallback_model="gpt-4o-mini",
        enable_prompt_caching=True,
        enable_batch_api=True,
        enable_cost_tracking=True,
    )


@pytest.fixture
def mock_settings(budget_config: BudgetConfig, llm_config: LLMConfig) -> Settings:
    """Complete test settings."""
    return Settings(budget=budget_config, llm=llm_config)


# ---------------------------------------------------------------------------
# Core service mocks
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_cost_tracker() -> AsyncMock:
    """Mock CostTracker that records calls without DB access."""
    tracker = AsyncMock(spec=CostTracker)
    tracker.record_call.return_value = CallCost(
        model="claude-sonnet-4-5",
        agent_role="dev",
        project_id="test-project",
        input_tokens=100,
        output_tokens=50,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        is_batch=False,
        real_cost_usd=0.001,
        theoretical_cost_usd=0.002,
        cache_savings_usd=0.0,
        batch_savings_usd=0.0,
    )
    tracker.check_budget_alerts.return_value = []
    tracker.has_emergency_alert.return_value = False
    return tracker


@pytest.fixture
def mock_cache_manager() -> MagicMock:
    """Mock CacheManager."""
    manager = MagicMock(spec=CacheManager)
    manager.get_system_prompt.return_value = "You are a test agent."
    manager.build_cached_messages_litellm.return_value = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Test task."},
    ]
    manager.get_cache_stats.return_value = MagicMock(
        total_cache_write_tokens=0,
        total_cache_read_tokens=0,
        cache_hit_ratio=0.0,
        estimated_savings_usd=0.0,
    )
    manager.update_stats = MagicMock()
    return manager


@pytest.fixture
def mock_batch_processor() -> AsyncMock:
    """Mock BatchProcessor."""
    processor = AsyncMock(spec=BatchProcessor)
    processor.should_use_batch.return_value = False
    return processor


@pytest.fixture
def llm_router() -> LLMRouter:
    """Real LLMRouter instance (no external calls, just routing logic)."""
    return LLMRouter()


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_project_id() -> str:
    """Standard test project ID."""
    return "test-project-001"


@pytest.fixture
def sample_task_dict() -> dict:
    """Sample task dictionary for API tests."""
    return {
        "id": "test-task-001",
        "type": "code_generation",
        "description": "Implement user authentication module",
        "complexity": "medium",
        "project_id": "test-project-001",
        "context": {},
    }
