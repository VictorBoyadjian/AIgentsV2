"""
Real-time LLM cost tracking with budget alerts and optimization suggestions.

Tracks every LLM call with full cost breakdown including cache and batch discounts.
Persists data to PostgreSQL and exposes Prometheus metrics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from core.config import (
    BudgetAlertLevel,
    BudgetConfig,
    get_settings,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
LLM_CALL_COST = Histogram(
    "llm_call_cost_usd",
    "Cost of individual LLM calls in USD",
    ["agent_role", "model", "project_id"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)
LLM_TOTAL_COST = Counter(
    "llm_total_cost_usd",
    "Cumulative LLM cost in USD",
    ["agent_role", "model", "project_id"],
)
LLM_TOKENS_USED = Counter(
    "llm_tokens_used_total",
    "Total tokens used",
    ["agent_role", "model", "token_type"],
)
LLM_CACHE_SAVINGS = Counter(
    "llm_cache_savings_usd",
    "Money saved via prompt caching in USD",
    ["project_id"],
)
LLM_BATCH_SAVINGS = Counter(
    "llm_batch_savings_usd",
    "Money saved via batch API in USD",
    ["project_id"],
)
BUDGET_USAGE_RATIO = Gauge(
    "budget_usage_ratio",
    "Current budget usage as ratio (0.0-1.0+)",
    ["project_id", "period"],
)


# ---------------------------------------------------------------------------
# Pricing table (March 2026 rates)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPricing:
    """Pricing per million tokens for a specific model."""

    input_per_million: float
    output_per_million: float
    cache_write_per_million: float
    cache_read_per_million: float
    batch_discount: float  # multiplier, e.g. 0.50 = 50% off


PRICING_TABLE: dict[str, ModelPricing] = {
    "claude-opus-4": ModelPricing(
        input_per_million=15.00,
        output_per_million=75.00,
        cache_write_per_million=18.75,
        cache_read_per_million=1.50,
        batch_discount=0.50,
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_per_million=3.00,
        output_per_million=15.00,
        cache_write_per_million=3.75,
        cache_read_per_million=0.30,
        batch_discount=0.50,
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_million=1.00,
        output_per_million=5.00,
        cache_write_per_million=1.25,
        cache_read_per_million=0.10,
        batch_discount=0.50,
    ),
    "gpt-4o": ModelPricing(
        input_per_million=2.50,
        output_per_million=10.00,
        cache_write_per_million=2.50,
        cache_read_per_million=1.25,
        batch_discount=0.50,
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_million=0.15,
        output_per_million=0.60,
        cache_write_per_million=0.15,
        cache_read_per_million=0.075,
        batch_discount=0.50,
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CallCost:
    """Cost breakdown for a single LLM call."""

    model: str
    agent_role: str
    project_id: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    is_batch: bool
    real_cost_usd: float
    theoretical_cost_usd: float  # cost without any optimizations
    cache_savings_usd: float
    batch_savings_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DailyCostReport:
    """Daily cost report for a project."""

    project_id: str
    date: str
    total_cost_usd: float
    total_calls: int
    cost_by_agent: dict[str, float]
    cache_savings_usd: float
    batch_savings_usd: float
    total_savings_usd: float


@dataclass
class MonthlyProjection:
    """Monthly cost projection based on recent usage."""

    project_id: str
    current_month_cost: float
    projected_month_cost: float
    daily_average: float
    days_remaining: int
    budget_usd: float
    projected_over_budget: bool


@dataclass
class BudgetAlert:
    """Budget alert triggered by threshold crossing."""

    level: BudgetAlertLevel
    project_id: str
    message: str
    current_spend: float
    budget_limit: float
    usage_ratio: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptimizationTip:
    """Automated cost optimization suggestion."""

    category: str
    description: str
    estimated_monthly_savings_usd: float
    agent_role: str | None = None


# ---------------------------------------------------------------------------
# SQLAlchemy model
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class LLMCallRecord(Base):
    """Persistent record of every LLM API call with cost details."""

    __tablename__ = "llm_call_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_role = Column(String(50), nullable=False, index=True)
    project_id = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cache_creation_tokens = Column(Integer, nullable=False, default=0)
    cache_read_tokens = Column(Integer, nullable=False, default=0)
    is_batch = Column(Boolean, nullable=False, default=False)
    real_cost_usd = Column(Float, nullable=False)
    theoretical_cost_usd = Column(Float, nullable=False)
    cache_savings_usd = Column(Float, nullable=False, default=0.0)
    batch_savings_usd = Column(Float, nullable=False, default=0.0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------
class CostTracker:
    """
    Real-time LLM cost tracker with budget alerts.

    Tracks every LLM call, persists to PostgreSQL, exposes Prometheus metrics,
    and triggers alerts when budget thresholds are crossed.
    """

    def __init__(
        self,
        budget_config: BudgetConfig | None = None,
        database_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._budget = budget_config or settings.budget
        db_url = database_url or settings.database.database_url
        self._engine = create_async_engine(db_url, pool_size=5, max_overflow=5)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("cost_tracker.initialized")

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        is_batch: bool = False,
    ) -> tuple[float, float, float, float]:
        """
        Calculate real and theoretical costs for an LLM call.

        Returns:
            (real_cost, theoretical_cost, cache_savings, batch_savings)
        """
        pricing = PRICING_TABLE.get(model)
        if pricing is None:
            logger.warning("cost_tracker.unknown_model", model=model)
            return 0.0, 0.0, 0.0, 0.0

        # Theoretical cost: no caching, no batch — all tokens at standard input rate
        total_input = input_tokens + cache_creation_tokens + cache_read_tokens
        theoretical = (
            total_input * pricing.input_per_million / 1_000_000
            + output_tokens * pricing.output_per_million / 1_000_000
        )

        # Real cost: regular input + cache write + cache read
        regular_input_cost = input_tokens * pricing.input_per_million / 1_000_000
        cache_write_cost = cache_creation_tokens * pricing.cache_write_per_million / 1_000_000
        cache_read_cost = cache_read_tokens * pricing.cache_read_per_million / 1_000_000
        output_cost = output_tokens * pricing.output_per_million / 1_000_000

        real_cost = regular_input_cost + cache_write_cost + cache_read_cost + output_cost

        # Cache savings: what cache_read_tokens would have cost at full price
        cache_savings = (
            cache_read_tokens * pricing.input_per_million / 1_000_000 - cache_read_cost
        )

        # Apply batch discount
        batch_savings = 0.0
        if is_batch:
            pre_batch = real_cost
            real_cost *= pricing.batch_discount
            batch_savings = pre_batch - real_cost

        return real_cost, theoretical, max(cache_savings, 0.0), max(batch_savings, 0.0)

    # ------------------------------------------------------------------
    # Record a call
    # ------------------------------------------------------------------
    async def record_call(
        self,
        agent_role: str,
        project_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        is_batch: bool = False,
    ) -> CallCost:
        """Record the cost of an LLM call and persist to database."""
        real_cost, theoretical, cache_savings, batch_savings = self.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            is_batch=is_batch,
        )

        call_cost = CallCost(
            model=model,
            agent_role=agent_role,
            project_id=project_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            is_batch=is_batch,
            real_cost_usd=real_cost,
            theoretical_cost_usd=theoretical,
            cache_savings_usd=cache_savings,
            batch_savings_usd=batch_savings,
        )

        # Persist to DB
        async with self._session_factory() as session:
            record = LLMCallRecord(
                agent_role=agent_role,
                project_id=project_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                is_batch=is_batch,
                real_cost_usd=real_cost,
                theoretical_cost_usd=theoretical,
                cache_savings_usd=cache_savings,
                batch_savings_usd=batch_savings,
            )
            session.add(record)
            await session.commit()

        # Update Prometheus metrics
        LLM_CALL_COST.labels(
            agent_role=agent_role, model=model, project_id=project_id
        ).observe(real_cost)
        LLM_TOTAL_COST.labels(
            agent_role=agent_role, model=model, project_id=project_id
        ).inc(real_cost)
        LLM_TOKENS_USED.labels(
            agent_role=agent_role, model=model, token_type="input"
        ).inc(input_tokens)
        LLM_TOKENS_USED.labels(
            agent_role=agent_role, model=model, token_type="output"
        ).inc(output_tokens)
        if cache_savings > 0:
            LLM_CACHE_SAVINGS.labels(project_id=project_id).inc(cache_savings)
        if batch_savings > 0:
            LLM_BATCH_SAVINGS.labels(project_id=project_id).inc(batch_savings)

        logger.info(
            "cost_tracker.call_recorded",
            agent_role=agent_role,
            model=model,
            real_cost=round(real_cost, 6),
            cache_savings=round(cache_savings, 6),
            batch_savings=round(batch_savings, 6),
        )

        return call_cost

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    async def get_daily_cost(self, project_id: str, date: datetime | None = None) -> DailyCostReport:
        """Get cost report for a specific day."""
        target_date = date or datetime.now(timezone.utc)
        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        async with self._session_factory() as session:
            # Total cost and counts
            result = await session.execute(
                select(
                    func.coalesce(func.sum(LLMCallRecord.real_cost_usd), 0.0),
                    func.coalesce(func.count(LLMCallRecord.id), 0),
                    func.coalesce(func.sum(LLMCallRecord.cache_savings_usd), 0.0),
                    func.coalesce(func.sum(LLMCallRecord.batch_savings_usd), 0.0),
                ).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= day_start,
                    LLMCallRecord.created_at < day_end,
                )
            )
            row = result.one()
            total_cost = float(row[0])
            total_calls = int(row[1])
            cache_savings = float(row[2])
            batch_savings = float(row[3])

            # Cost by agent
            agent_result = await session.execute(
                select(
                    LLMCallRecord.agent_role,
                    func.sum(LLMCallRecord.real_cost_usd),
                ).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= day_start,
                    LLMCallRecord.created_at < day_end,
                ).group_by(LLMCallRecord.agent_role)
            )
            cost_by_agent = {row[0]: float(row[1]) for row in agent_result.all()}

        return DailyCostReport(
            project_id=project_id,
            date=day_start.strftime("%Y-%m-%d"),
            total_cost_usd=total_cost,
            total_calls=total_calls,
            cost_by_agent=cost_by_agent,
            cache_savings_usd=cache_savings,
            batch_savings_usd=batch_savings,
            total_savings_usd=cache_savings + batch_savings,
        )

    async def get_monthly_projection(self, project_id: str) -> MonthlyProjection:
        """Project monthly cost based on the last 7 days of usage."""
        now = datetime.now(timezone.utc)
        seven_days_ago = now - timedelta(days=7)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        async with self._session_factory() as session:
            # Last 7 days cost
            result = await session.execute(
                select(func.coalesce(func.sum(LLMCallRecord.real_cost_usd), 0.0)).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= seven_days_ago,
                )
            )
            last_7_days_cost = float(result.scalar_one())

            # Current month cost
            result = await session.execute(
                select(func.coalesce(func.sum(LLMCallRecord.real_cost_usd), 0.0)).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= month_start,
                )
            )
            current_month_cost = float(result.scalar_one())

        daily_average = last_7_days_cost / 7.0 if last_7_days_cost > 0 else 0.0
        days_in_month = (month_start.replace(month=month_start.month % 12 + 1, day=1) - month_start).days
        days_elapsed = (now - month_start).days + 1
        days_remaining = max(days_in_month - days_elapsed, 0)
        projected = current_month_cost + (daily_average * days_remaining)

        return MonthlyProjection(
            project_id=project_id,
            current_month_cost=current_month_cost,
            projected_month_cost=projected,
            daily_average=daily_average,
            days_remaining=days_remaining,
            budget_usd=self._budget.monthly_budget_usd,
            projected_over_budget=projected > self._budget.monthly_budget_usd,
        )

    # ------------------------------------------------------------------
    # Budget alerts
    # ------------------------------------------------------------------
    async def check_budget_alerts(self, project_id: str) -> list[BudgetAlert]:
        """Check current spend against budget thresholds and return any triggered alerts."""
        alerts: list[BudgetAlert] = []
        now = datetime.now(timezone.utc)

        # Daily check
        daily_report = await self.get_daily_cost(project_id)
        daily_ratio = daily_report.total_cost_usd / self._budget.daily_budget_usd if self._budget.daily_budget_usd > 0 else 0.0

        BUDGET_USAGE_RATIO.labels(project_id=project_id, period="daily").set(daily_ratio)

        if daily_ratio >= 1.0:
            alerts.append(BudgetAlert(
                level=BudgetAlertLevel.EMERGENCY,
                project_id=project_id,
                message=f"Daily budget EXCEEDED: ${daily_report.total_cost_usd:.2f} / ${self._budget.daily_budget_usd:.2f}. Auto-downgrading all agents to {self._budget.emergency_model_override}.",
                current_spend=daily_report.total_cost_usd,
                budget_limit=self._budget.daily_budget_usd,
                usage_ratio=daily_ratio,
            ))
        elif daily_ratio >= self._budget.critical_threshold:
            alerts.append(BudgetAlert(
                level=BudgetAlertLevel.CRITICAL,
                project_id=project_id,
                message=f"Daily budget at {daily_ratio:.0%}: ${daily_report.total_cost_usd:.2f} / ${self._budget.daily_budget_usd:.2f}",
                current_spend=daily_report.total_cost_usd,
                budget_limit=self._budget.daily_budget_usd,
                usage_ratio=daily_ratio,
            ))
        elif daily_ratio >= self._budget.warning_threshold:
            alerts.append(BudgetAlert(
                level=BudgetAlertLevel.WARNING,
                project_id=project_id,
                message=f"Daily budget at {daily_ratio:.0%}: ${daily_report.total_cost_usd:.2f} / ${self._budget.daily_budget_usd:.2f}",
                current_spend=daily_report.total_cost_usd,
                budget_limit=self._budget.daily_budget_usd,
                usage_ratio=daily_ratio,
            ))

        # Monthly check
        projection = await self.get_monthly_projection(project_id)
        monthly_ratio = projection.current_month_cost / self._budget.monthly_budget_usd if self._budget.monthly_budget_usd > 0 else 0.0

        BUDGET_USAGE_RATIO.labels(project_id=project_id, period="monthly").set(monthly_ratio)

        if monthly_ratio >= 1.0:
            alerts.append(BudgetAlert(
                level=BudgetAlertLevel.EMERGENCY,
                project_id=project_id,
                message=f"Monthly budget EXCEEDED: ${projection.current_month_cost:.2f} / ${self._budget.monthly_budget_usd:.2f}. Auto-downgrading all agents.",
                current_spend=projection.current_month_cost,
                budget_limit=self._budget.monthly_budget_usd,
                usage_ratio=monthly_ratio,
            ))
        elif monthly_ratio >= self._budget.critical_threshold:
            alerts.append(BudgetAlert(
                level=BudgetAlertLevel.CRITICAL,
                project_id=project_id,
                message=f"Monthly budget at {monthly_ratio:.0%}: ${projection.current_month_cost:.2f} / ${self._budget.monthly_budget_usd:.2f}",
                current_spend=projection.current_month_cost,
                budget_limit=self._budget.monthly_budget_usd,
                usage_ratio=monthly_ratio,
            ))
        elif monthly_ratio >= self._budget.warning_threshold:
            alerts.append(BudgetAlert(
                level=BudgetAlertLevel.WARNING,
                project_id=project_id,
                message=f"Monthly budget at {monthly_ratio:.0%}: ${projection.current_month_cost:.2f} / ${self._budget.monthly_budget_usd:.2f}",
                current_spend=projection.current_month_cost,
                budget_limit=self._budget.monthly_budget_usd,
                usage_ratio=monthly_ratio,
            ))

        for alert in alerts:
            logger.warning(
                "cost_tracker.budget_alert",
                level=alert.level.value,
                project_id=project_id,
                usage_ratio=alert.usage_ratio,
                message=alert.message,
            )

        return alerts

    def has_emergency_alert(self, alerts: list[BudgetAlert]) -> bool:
        """Check if any alert is at EMERGENCY level."""
        return any(a.level == BudgetAlertLevel.EMERGENCY for a in alerts)

    # ------------------------------------------------------------------
    # Optimization suggestions
    # ------------------------------------------------------------------
    async def get_optimization_suggestions(self, project_id: str) -> list[OptimizationTip]:
        """Analyze usage patterns and suggest cost optimizations."""
        tips: list[OptimizationTip] = []
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        async with self._session_factory() as session:
            # Check for agents using expensive models on simple tasks
            agent_costs = await session.execute(
                select(
                    LLMCallRecord.agent_role,
                    LLMCallRecord.model,
                    func.count(LLMCallRecord.id).label("call_count"),
                    func.sum(LLMCallRecord.real_cost_usd).label("total_cost"),
                    func.avg(LLMCallRecord.input_tokens).label("avg_input"),
                    func.avg(LLMCallRecord.output_tokens).label("avg_output"),
                ).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= thirty_days_ago,
                ).group_by(LLMCallRecord.agent_role, LLMCallRecord.model)
            )

            for row in agent_costs.all():
                role = row[0]
                model = row[1]
                call_count = int(row[2])
                total_cost = float(row[3])
                avg_input = float(row[4])
                avg_output = float(row[5])

                # Suggest downgrade if average tokens are low (simple tasks)
                if model in ("claude-sonnet-4-5", "claude-opus-4") and avg_input < 500 and avg_output < 200:
                    cheaper = "claude-haiku-4-5"
                    cheaper_pricing = PRICING_TABLE[cheaper]
                    current_pricing = PRICING_TABLE.get(model)
                    if current_pricing:
                        current_per_call = (avg_input * current_pricing.input_per_million + avg_output * current_pricing.output_per_million) / 1_000_000
                        cheaper_per_call = (avg_input * cheaper_pricing.input_per_million + avg_output * cheaper_pricing.output_per_million) / 1_000_000
                        monthly_savings = (current_per_call - cheaper_per_call) * call_count
                        tips.append(OptimizationTip(
                            category="model_downgrade",
                            description=f"Agent '{role}' uses {model} but avg token counts are low ({avg_input:.0f} in / {avg_output:.0f} out). Switch to {cheaper} for ~${monthly_savings:.2f}/month savings.",
                            estimated_monthly_savings_usd=monthly_savings,
                            agent_role=role,
                        ))

            # Check for low cache utilization
            cache_stats = await session.execute(
                select(
                    func.sum(LLMCallRecord.cache_read_tokens).label("total_cache_read"),
                    func.sum(LLMCallRecord.input_tokens).label("total_input"),
                    func.sum(LLMCallRecord.cache_savings_usd).label("total_savings"),
                ).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= thirty_days_ago,
                )
            )
            cache_row = cache_stats.one()
            total_cache_read = float(cache_row[0] or 0)
            total_input = float(cache_row[1] or 1)

            cache_ratio = total_cache_read / total_input if total_input > 0 else 0
            if cache_ratio < 0.2 and total_input > 10000:
                tips.append(OptimizationTip(
                    category="cache_utilization",
                    description=f"Only {cache_ratio:.0%} of input tokens are cached. Ensure system prompts and project context use cache_control for up to 50% savings on repeated tokens.",
                    estimated_monthly_savings_usd=total_input * 0.003 / 1_000_000 * 0.5 * 30,
                ))

            # Check for batch-eligible tasks running in real-time
            non_batch_count = await session.execute(
                select(func.count(LLMCallRecord.id)).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.is_batch == False,  # noqa: E712
                    LLMCallRecord.created_at >= thirty_days_ago,
                    LLMCallRecord.agent_role.in_(["qa", "devops"]),
                )
            )
            nb_count = int(non_batch_count.scalar_one())
            if nb_count > 20:
                tips.append(OptimizationTip(
                    category="batch_eligible",
                    description=f"{nb_count} calls from QA/DevOps agents ran in real-time mode. Route non-blocking tasks through Batch API for 50% savings.",
                    estimated_monthly_savings_usd=nb_count * 0.01 * 0.5,
                ))

        return tips

    # ------------------------------------------------------------------
    # Period cost query
    # ------------------------------------------------------------------
    async def get_period_cost(
        self,
        project_id: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """Get aggregated cost data for an arbitrary time period."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.coalesce(func.sum(LLMCallRecord.real_cost_usd), 0.0),
                    func.coalesce(func.sum(LLMCallRecord.theoretical_cost_usd), 0.0),
                    func.coalesce(func.sum(LLMCallRecord.cache_savings_usd), 0.0),
                    func.coalesce(func.sum(LLMCallRecord.batch_savings_usd), 0.0),
                    func.coalesce(func.count(LLMCallRecord.id), 0),
                ).where(
                    LLMCallRecord.project_id == project_id,
                    LLMCallRecord.created_at >= start,
                    LLMCallRecord.created_at < end,
                )
            )
            row = result.one()
            return {
                "project_id": project_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "real_cost_usd": float(row[0]),
                "theoretical_cost_usd": float(row[1]),
                "cache_savings_usd": float(row[2]),
                "batch_savings_usd": float(row[3]),
                "total_savings_usd": float(row[2]) + float(row[3]),
                "total_calls": int(row[4]),
            }
