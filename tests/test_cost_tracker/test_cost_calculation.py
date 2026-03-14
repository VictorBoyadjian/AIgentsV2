"""Tests for the CostTracker module — cost calculation, recording, and alerts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.cost_tracker import (
    CallCost,
    CostTracker,
    PRICING_TABLE,
    ModelPricing,
)
from core.config import BudgetAlertLevel


class TestCostCalculation:
    """Test the static cost calculation logic."""

    def test_basic_cost_calculation(self) -> None:
        """Verify basic input/output token cost calculation."""
        real, theoretical, cache_sav, batch_sav = CostTracker.calculate_cost(
            model="claude-sonnet-4-5",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            is_batch=False,
        )

        # claude-sonnet-4-5: $3/M input, $15/M output
        expected_input = 1000 * 3.0 / 1_000_000  # 0.003
        expected_output = 500 * 15.0 / 1_000_000  # 0.0075
        expected_total = expected_input + expected_output

        assert abs(real - expected_total) < 1e-8
        assert abs(theoretical - expected_total) < 1e-8
        assert cache_sav == 0.0
        assert batch_sav == 0.0

    def test_cost_with_cache_tokens(self) -> None:
        """Verify cache read tokens are discounted vs standard input price."""
        real, theoretical, cache_sav, batch_sav = CostTracker.calculate_cost(
            model="claude-sonnet-4-5",
            input_tokens=500,
            output_tokens=200,
            cache_creation_tokens=0,
            cache_read_tokens=500,
            is_batch=False,
        )

        pricing = PRICING_TABLE["claude-sonnet-4-5"]

        # Theoretical: all 1000 input tokens at standard rate
        theo_input = 1000 * pricing.input_per_million / 1_000_000
        theo_output = 200 * pricing.output_per_million / 1_000_000
        expected_theoretical = theo_input + theo_output

        # Real: 500 regular + 500 cache_read (at discounted rate)
        real_input = 500 * pricing.input_per_million / 1_000_000
        real_cache = 500 * pricing.cache_read_per_million / 1_000_000
        real_output = 200 * pricing.output_per_million / 1_000_000
        expected_real = real_input + real_cache + real_output

        # Cache savings: what cache tokens would have cost at full price minus actual
        expected_savings = (500 * pricing.input_per_million / 1_000_000) - real_cache

        assert abs(real - expected_real) < 1e-8
        assert abs(theoretical - expected_theoretical) < 1e-8
        assert abs(cache_sav - expected_savings) < 1e-8
        assert cache_sav > 0  # cache should save money

    def test_cost_with_batch_discount(self) -> None:
        """Verify batch mode applies 50% discount."""
        real_normal, _, _, _ = CostTracker.calculate_cost(
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            is_batch=False,
        )

        real_batch, _, _, batch_sav = CostTracker.calculate_cost(
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            is_batch=True,
        )

        assert abs(real_batch - real_normal * 0.5) < 1e-8
        assert abs(batch_sav - real_normal * 0.5) < 1e-8

    def test_cost_with_cache_write(self) -> None:
        """Verify cache write tokens use the higher write rate."""
        real, _, _, _ = CostTracker.calculate_cost(
            model="claude-opus-4",
            input_tokens=100,
            output_tokens=100,
            cache_creation_tokens=500,
            cache_read_tokens=0,
            is_batch=False,
        )

        pricing = PRICING_TABLE["claude-opus-4"]
        expected = (
            100 * pricing.input_per_million / 1_000_000
            + 500 * pricing.cache_write_per_million / 1_000_000
            + 100 * pricing.output_per_million / 1_000_000
        )

        assert abs(real - expected) < 1e-8

    def test_unknown_model_returns_zero(self) -> None:
        """Unknown model should return zero costs."""
        real, theoretical, cache_sav, batch_sav = CostTracker.calculate_cost(
            model="unknown-model-xyz",
            input_tokens=1000,
            output_tokens=500,
        )
        assert real == 0.0
        assert theoretical == 0.0
        assert cache_sav == 0.0
        assert batch_sav == 0.0

    def test_combined_cache_and_batch(self) -> None:
        """Verify combined cache + batch discounts stack correctly."""
        real, theoretical, cache_sav, batch_sav = CostTracker.calculate_cost(
            model="claude-sonnet-4-5",
            input_tokens=200,
            output_tokens=100,
            cache_creation_tokens=0,
            cache_read_tokens=800,
            is_batch=True,
        )

        pricing = PRICING_TABLE["claude-sonnet-4-5"]

        # Pre-batch cost
        pre_batch = (
            200 * pricing.input_per_million / 1_000_000
            + 800 * pricing.cache_read_per_million / 1_000_000
            + 100 * pricing.output_per_million / 1_000_000
        )

        # After batch discount
        expected_real = pre_batch * 0.5

        assert abs(real - expected_real) < 1e-8
        assert cache_sav > 0
        assert batch_sav > 0
        assert real < theoretical  # optimized is always cheaper


class TestPricingTable:
    """Validate the pricing table configuration."""

    def test_all_models_present(self) -> None:
        """Verify all expected models are in the pricing table."""
        expected_models = {
            "claude-opus-4",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "gpt-4o",
            "gpt-4o-mini",
        }
        assert set(PRICING_TABLE.keys()) == expected_models

    def test_cache_read_cheaper_than_input(self) -> None:
        """Cache read must always be cheaper than standard input."""
        for model, pricing in PRICING_TABLE.items():
            assert pricing.cache_read_per_million < pricing.input_per_million, (
                f"{model}: cache_read ({pricing.cache_read_per_million}) "
                f"should be < input ({pricing.input_per_million})"
            )

    def test_batch_discount_is_half(self) -> None:
        """All models should have 50% batch discount."""
        for model, pricing in PRICING_TABLE.items():
            assert pricing.batch_discount == 0.5, (
                f"{model}: batch_discount should be 0.5, got {pricing.batch_discount}"
            )

    def test_opus_most_expensive(self) -> None:
        """Opus should be the most expensive model."""
        opus = PRICING_TABLE["claude-opus-4"]
        for model, pricing in PRICING_TABLE.items():
            if model != "claude-opus-4":
                assert pricing.input_per_million <= opus.input_per_million
                assert pricing.output_per_million <= opus.output_per_million
