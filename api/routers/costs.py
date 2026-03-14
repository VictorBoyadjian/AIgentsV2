"""
Cost monitoring and budget management endpoints.

Provides routes for cost summaries, breakdowns by agent, savings analysis,
monthly projections, budget management, optimization tips, and a WebSocket
for real-time cost streaming.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from api.dependencies import get_cost_tracker, get_database
from api.schemas import (
    BudgetSetRequest,
    BudgetSetResponse,
    CostBreakdownResponse,
    CostProjectionResponse,
    CostSavingsResponse,
    CostSummaryResponse,
    OptimizationTipResponse,
    SavingsDetail,
)
from core.cost_tracker import CostTracker
from memory.database import Database

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _period_bounds(period: str) -> tuple[datetime, datetime]:
    """Return (start, end) datetimes for a named period.

    Supported periods: ``day``, ``week``, ``month``.
    """
    now = datetime.now(timezone.utc)
    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == "week":
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(weeks=1)
    elif period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # First day of next month
        if now.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period '{period}'. Must be 'day', 'week', or 'month'.",
        )
    return start, end


# ---------------------------------------------------------------------------
# GET /costs/summary  -- cost totals per period
# ---------------------------------------------------------------------------
@router.get(
    "/summary",
    response_model=list[CostSummaryResponse],
    summary="Cost summary per period for a project",
)
async def cost_summary(
    project_id: str = Query(..., description="Project ID to query costs for."),
    periods: str = Query(
        default="day,week,month",
        description="Comma-separated periods: day, week, month.",
    ),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> list[CostSummaryResponse]:
    """Return aggregated cost summaries for the requested periods.

    Each period includes real cost, theoretical cost, savings from cache and
    batch, and the total number of LLM calls.
    """
    logger.info("api.cost_summary", project_id=project_id, periods=periods)

    results: list[CostSummaryResponse] = []
    for period in periods.split(","):
        period = period.strip()
        start, end = _period_bounds(period)
        data = await cost_tracker.get_period_cost(project_id, start, end)
        results.append(
            CostSummaryResponse(
                project_id=project_id,
                period=period,
                start=start.isoformat(),
                end=end.isoformat(),
                real_cost_usd=data["real_cost_usd"],
                theoretical_cost_usd=data["theoretical_cost_usd"],
                cache_savings_usd=data["cache_savings_usd"],
                batch_savings_usd=data["batch_savings_usd"],
                total_savings_usd=data["total_savings_usd"],
                total_calls=data["total_calls"],
            )
        )

    return results


# ---------------------------------------------------------------------------
# GET /costs/{project_id}/breakdown  -- detail by agent
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}/breakdown",
    response_model=CostBreakdownResponse,
    summary="Cost breakdown by agent for today",
)
async def cost_breakdown(
    project_id: str,
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> CostBreakdownResponse:
    """Return today's cost broken down by agent role.

    Includes total cost, per-agent breakdown, and savings from cache and
    batch optimizations.
    """
    logger.info("api.cost_breakdown", project_id=project_id)

    report = await cost_tracker.get_daily_cost(project_id)

    return CostBreakdownResponse(
        project_id=report.project_id,
        date=report.date,
        total_cost_usd=report.total_cost_usd,
        total_calls=report.total_calls,
        cost_by_agent=report.cost_by_agent,
        cache_savings_usd=report.cache_savings_usd,
        batch_savings_usd=report.batch_savings_usd,
        total_savings_usd=report.total_savings_usd,
    )


# ---------------------------------------------------------------------------
# GET /costs/{project_id}/savings  -- savings achieved
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}/savings",
    response_model=CostSavingsResponse,
    summary="Savings from cache and batch optimizations",
)
async def cost_savings(
    project_id: str,
    period: str = Query(
        default="month",
        description="Period to calculate savings for: day, week, month.",
    ),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> CostSavingsResponse:
    """Return dollar and percentage savings from cache and batch APIs.

    Compares real cost against theoretical cost (what would have been paid
    without any optimizations).
    """
    logger.info("api.cost_savings", project_id=project_id, period=period)

    start, end = _period_bounds(period)
    data = await cost_tracker.get_period_cost(project_id, start, end)

    real = data["real_cost_usd"]
    theoretical = data["theoretical_cost_usd"]
    cache_saved = data["cache_savings_usd"]
    batch_saved = data["batch_savings_usd"]
    total_saved = cache_saved + batch_saved

    total_pct = (total_saved / theoretical * 100.0) if theoretical > 0 else 0.0
    cache_pct = (cache_saved / theoretical * 100.0) if theoretical > 0 else 0.0
    batch_pct = (batch_saved / theoretical * 100.0) if theoretical > 0 else 0.0

    return CostSavingsResponse(
        project_id=project_id,
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        real_cost_usd=real,
        theoretical_cost_usd=theoretical,
        savings=[
            SavingsDetail(source="cache", saved_usd=cache_saved, percentage=cache_pct),
            SavingsDetail(source="batch", saved_usd=batch_saved, percentage=batch_pct),
        ],
        total_saved_usd=total_saved,
        total_saved_percentage=total_pct,
    )


# ---------------------------------------------------------------------------
# GET /costs/{project_id}/projection  -- monthly projection
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}/projection",
    response_model=CostProjectionResponse,
    summary="Monthly cost projection with trend",
)
async def cost_projection(
    project_id: str,
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> CostProjectionResponse:
    """Project the monthly cost based on the last 7 days of usage.

    Returns the current month-to-date cost, projected end-of-month cost,
    daily average, remaining days, and whether the projection exceeds
    the configured budget.
    """
    logger.info("api.cost_projection", project_id=project_id)

    projection = await cost_tracker.get_monthly_projection(project_id)

    budget_ratio = (
        projection.current_month_cost / projection.budget_usd
        if projection.budget_usd > 0
        else 0.0
    )

    return CostProjectionResponse(
        project_id=projection.project_id,
        current_month_cost=projection.current_month_cost,
        projected_month_cost=projection.projected_month_cost,
        daily_average=projection.daily_average,
        days_remaining=projection.days_remaining,
        budget_usd=projection.budget_usd,
        projected_over_budget=projection.projected_over_budget,
        budget_usage_ratio=round(budget_ratio, 4),
    )


# ---------------------------------------------------------------------------
# POST /costs/{project_id}/budget  -- set monthly budget
# ---------------------------------------------------------------------------
@router.post(
    "/{project_id}/budget",
    response_model=BudgetSetResponse,
    summary="Set monthly budget for a project",
)
async def set_budget(
    project_id: str,
    body: BudgetSetRequest,
    db: Database = Depends(get_database),
) -> BudgetSetResponse:
    """Set or update the monthly LLM budget for a project.

    The budget is stored on the project record in the database. Budget
    alerts and projections use this value going forward.
    """
    logger.info(
        "api.set_budget",
        project_id=project_id,
        budget=body.monthly_budget_usd,
    )

    project = await db.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    # Update the budget on the project record
    async with db._session_factory() as session:
        from memory.database import ProjectRecord
        from sqlalchemy import select

        result = await session.execute(
            select(ProjectRecord).where(ProjectRecord.id == project_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project '{project_id}' not found.",
            )
        record.monthly_budget_usd = body.monthly_budget_usd
        await session.commit()

    logger.info(
        "api.budget_updated",
        project_id=project_id,
        new_budget=body.monthly_budget_usd,
    )

    return BudgetSetResponse(
        project_id=project_id,
        monthly_budget_usd=body.monthly_budget_usd,
    )


# ---------------------------------------------------------------------------
# GET /costs/optimization-tips  -- automatic suggestions
# ---------------------------------------------------------------------------
@router.get(
    "/optimization-tips",
    response_model=list[OptimizationTipResponse],
    summary="Get cost optimization suggestions",
)
async def optimization_tips(
    project_id: str = Query(..., description="Project ID to analyse."),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> list[OptimizationTipResponse]:
    """Analyse usage patterns and return actionable optimization tips.

    Tips may include model downgrade suggestions, cache utilisation
    improvements, and batch API adoption recommendations.
    """
    logger.info("api.optimization_tips", project_id=project_id)

    tips = await cost_tracker.get_optimization_suggestions(project_id)

    return [
        OptimizationTipResponse(
            category=tip.category,
            description=tip.description,
            estimated_monthly_savings_usd=tip.estimated_monthly_savings_usd,
            agent_role=tip.agent_role,
        )
        for tip in tips
    ]


# ---------------------------------------------------------------------------
# WebSocket /ws/costs/live  -- real-time cost streaming
# ---------------------------------------------------------------------------
@router.websocket("/ws/costs/live")
async def live_cost_stream(
    websocket: WebSocket,
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> None:
    """Stream real-time cost updates to connected WebSocket clients.

    After connecting, the client must send a JSON message with:

    .. code-block:: json

        {"project_id": "xxx", "interval_seconds": 5}

    The server then pushes a cost summary every *interval_seconds* until
    the client disconnects.
    """
    await websocket.accept()
    logger.info("ws.costs.connected")

    try:
        # Wait for initial configuration message from client
        config_raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        config = json.loads(config_raw)
        project_id: str = config.get("project_id", "")
        interval: int = max(int(config.get("interval_seconds", 5)), 1)

        if not project_id:
            await websocket.send_json({"error": "project_id is required"})
            await websocket.close(code=1008)
            return

        logger.info(
            "ws.costs.streaming",
            project_id=project_id,
            interval=interval,
        )

        while True:
            try:
                # Fetch current daily cost
                report = await cost_tracker.get_daily_cost(project_id)
                projection = await cost_tracker.get_monthly_projection(project_id)

                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "project_id": project_id,
                    "daily": {
                        "total_cost_usd": report.total_cost_usd,
                        "total_calls": report.total_calls,
                        "cost_by_agent": report.cost_by_agent,
                        "cache_savings_usd": report.cache_savings_usd,
                        "batch_savings_usd": report.batch_savings_usd,
                    },
                    "projection": {
                        "current_month_cost": projection.current_month_cost,
                        "projected_month_cost": projection.projected_month_cost,
                        "daily_average": projection.daily_average,
                        "budget_usd": projection.budget_usd,
                        "over_budget": projection.projected_over_budget,
                    },
                }
                await websocket.send_json(payload)
            except Exception as exc:
                logger.warning(
                    "ws.costs.query_error",
                    project_id=project_id,
                    error=str(exc),
                )
                await websocket.send_json(
                    {"error": f"Failed to fetch costs: {exc}"}
                )

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info("ws.costs.disconnected")
    except asyncio.TimeoutError:
        logger.warning("ws.costs.config_timeout")
        await websocket.close(code=1008)
    except json.JSONDecodeError:
        logger.warning("ws.costs.invalid_config")
        await websocket.send_json({"error": "Invalid JSON configuration"})
        await websocket.close(code=1008)
    except Exception as exc:
        logger.error("ws.costs.error", error=str(exc))
        await websocket.close(code=1011)
