"""Tests for the LLM Router — model selection, complexity overrides, fallback."""

from __future__ import annotations

import pytest

from core.config import AgentRole, TaskComplexity
from core.llm_router import (
    LLMRouter,
    ROUTING_TABLE,
    TASK_COMPLEXITY_OVERRIDE,
    MODEL_QUALITY_SCORES,
    MODEL_COST_PER_1K,
)


class TestModelResolution:
    """Test model selection based on agent role and task complexity."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        return LLMRouter()

    @pytest.mark.parametrize(
        "role,expected_model",
        [
            ("architect", "claude-opus-4"),
            ("pm", "claude-sonnet-4-5"),
            ("dev", "claude-sonnet-4-5"),
            ("qa", "claude-haiku-4-5"),
            ("security", "claude-sonnet-4-5"),
            ("devops", "claude-haiku-4-5"),
            ("research", "gpt-4o"),
            ("fallback", "gpt-4o-mini"),
        ],
    )
    def test_default_model_per_role(
        self, router: LLMRouter, role: str, expected_model: str
    ) -> None:
        """Each role should map to its configured primary model."""
        model = router.resolve_model(role, TaskComplexity.MEDIUM)
        assert model == expected_model

    def test_simple_task_downgrades_to_haiku(self, router: LLMRouter) -> None:
        """SIMPLE complexity should override to claude-haiku-4-5."""
        for role in ["architect", "pm", "dev", "security"]:
            model = router.resolve_model(role, TaskComplexity.SIMPLE)
            assert model == "claude-haiku-4-5", (
                f"Role '{role}' should downgrade to haiku for SIMPLE tasks"
            )

    def test_critical_task_upgrades_to_opus(self, router: LLMRouter) -> None:
        """CRITICAL complexity should override to claude-opus-4."""
        for role in ["qa", "devops", "fallback", "dev"]:
            model = router.resolve_model(role, TaskComplexity.CRITICAL)
            assert model == "claude-opus-4", (
                f"Role '{role}' should upgrade to opus for CRITICAL tasks"
            )

    def test_medium_complexity_uses_default(self, router: LLMRouter) -> None:
        """MEDIUM complexity should use the default model for the role."""
        assert router.resolve_model("dev", TaskComplexity.MEDIUM) == "claude-sonnet-4-5"
        assert router.resolve_model("qa", TaskComplexity.MEDIUM) == "claude-haiku-4-5"
        assert router.resolve_model("research", TaskComplexity.MEDIUM) == "gpt-4o"

    def test_complex_complexity_uses_default(self, router: LLMRouter) -> None:
        """COMPLEX complexity should use the default model (no override)."""
        assert router.resolve_model("architect", TaskComplexity.COMPLEX) == "claude-opus-4"
        assert router.resolve_model("dev", TaskComplexity.COMPLEX) == "claude-sonnet-4-5"

    def test_unknown_role_falls_back(self, router: LLMRouter) -> None:
        """Unknown agent role should use fallback model."""
        model = router.resolve_model("nonexistent_role", TaskComplexity.MEDIUM)
        assert model == "gpt-4o-mini"


class TestFallbackModel:
    """Test fallback model resolution."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        return LLMRouter()

    @pytest.mark.parametrize(
        "role,expected_fallback",
        [
            ("architect", "claude-sonnet-4-5"),
            ("pm", "gpt-4o"),
            ("dev", "gpt-4o"),
            ("qa", "gpt-4o-mini"),
            ("research", "claude-sonnet-4-5"),
            ("fallback", "claude-haiku-4-5"),
        ],
    )
    def test_fallback_model_per_role(
        self, router: LLMRouter, role: str, expected_fallback: str
    ) -> None:
        """Each role should have its configured fallback model."""
        model = router.get_fallback_model(role)
        assert model == expected_fallback


class TestCheapestCapableModel:
    """Test finding the cheapest model above a quality threshold."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        return LLMRouter()

    def test_low_quality_threshold_returns_cheapest(self, router: LLMRouter) -> None:
        """Low quality threshold should return gpt-4o-mini (cheapest)."""
        model = router.get_cheapest_capable_model(min_quality_score=0.5)
        assert model == "gpt-4o-mini"

    def test_high_quality_threshold_returns_opus(self, router: LLMRouter) -> None:
        """Very high quality threshold should return opus (only option)."""
        model = router.get_cheapest_capable_model(min_quality_score=0.95)
        assert model == "claude-opus-4"

    def test_medium_threshold_returns_haiku(self, router: LLMRouter) -> None:
        """Medium threshold (0.7) should return claude-haiku-4-5."""
        model = router.get_cheapest_capable_model(min_quality_score=0.7)
        assert model == "claude-haiku-4-5"

    def test_impossible_threshold_returns_mini(self, router: LLMRouter) -> None:
        """Impossible threshold (>1.0) should return fallback."""
        model = router.get_cheapest_capable_model(min_quality_score=1.5)
        assert model == "gpt-4o-mini"


class TestCostEstimation:
    """Test cost estimation utility."""

    def test_estimate_cost_sonnet(self) -> None:
        """Verify cost estimation for claude-sonnet-4-5."""
        cost = LLMRouter.estimate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="claude-sonnet-4-5",
        )
        # $3/M input + $15/M output
        expected = 1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000
        assert abs(cost - expected) < 1e-8

    def test_estimate_cost_unknown_model(self) -> None:
        """Unknown model should return 0."""
        cost = LLMRouter.estimate_cost(1000, 500, "unknown-model")
        assert cost == 0.0


class TestFailureTracking:
    """Test model health and circuit breaker logic."""

    @pytest.fixture
    def router(self) -> LLMRouter:
        return LLMRouter()

    def test_initially_healthy(self, router: LLMRouter) -> None:
        """All models should be healthy initially."""
        for model in MODEL_QUALITY_SCORES:
            assert router.is_model_healthy(model) is True

    def test_unhealthy_after_failures(self, router: LLMRouter) -> None:
        """Model should be unhealthy after 3+ failures."""
        router._record_failure("gpt-4o")
        router._record_failure("gpt-4o")
        router._record_failure("gpt-4o")
        assert router.is_model_healthy("gpt-4o") is False

    def test_reset_failures(self, router: LLMRouter) -> None:
        """Resetting failures should restore health."""
        router._record_failure("gpt-4o")
        router._record_failure("gpt-4o")
        router._record_failure("gpt-4o")
        router.reset_failures("gpt-4o")
        assert router.is_model_healthy("gpt-4o") is True

    def test_reset_all_failures(self, router: LLMRouter) -> None:
        """Reset all should clear everything."""
        router._record_failure("gpt-4o")
        router._record_failure("claude-opus-4")
        router.reset_failures()
        assert router.is_model_healthy("gpt-4o") is True
        assert router.is_model_healthy("claude-opus-4") is True


class TestRoutingTable:
    """Validate routing table consistency."""

    def test_all_agent_roles_have_routes(self) -> None:
        """Every AgentRole should have a routing entry."""
        for role in AgentRole:
            assert role.value in ROUTING_TABLE, (
                f"Agent role '{role.value}' missing from ROUTING_TABLE"
            )

    def test_primary_models_exist_in_pricing(self) -> None:
        """All primary models should be in the pricing table."""
        from core.cost_tracker import PRICING_TABLE as pricing

        for role, entry in ROUTING_TABLE.items():
            assert entry.primary in pricing, (
                f"Primary model '{entry.primary}' for role '{role}' not in pricing table"
            )
            assert entry.fallback in pricing, (
                f"Fallback model '{entry.fallback}' for role '{role}' not in pricing table"
            )

    def test_complexity_overrides_valid(self) -> None:
        """All complexity overrides should point to valid models."""
        from core.cost_tracker import PRICING_TABLE as pricing

        for complexity, model in TASK_COMPLEXITY_OVERRIDE.items():
            if model is not None:
                assert model in pricing, (
                    f"Override model '{model}' for {complexity} not in pricing table"
                )
