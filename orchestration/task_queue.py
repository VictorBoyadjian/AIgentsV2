"""
Task queue backed by Celery for async task processing.

Handles batch job polling, periodic cost reporting, and
background agent task execution.
"""

from __future__ import annotations

import asyncio

from celery import Celery
from celery.schedules import crontab

from core.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
celery_app = Celery(
    "saas_agent_team",
    broker=settings.redis.celery_broker_url,
    backend=settings.redis.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

# ---------------------------------------------------------------------------
# Periodic tasks (Celery Beat)
# ---------------------------------------------------------------------------
celery_app.conf.beat_schedule = {
    "poll-batch-results": {
        "task": "orchestration.task_queue.poll_all_pending_batches",
        "schedule": 1800.0,  # Every 30 minutes
    },
    "daily-cost-report": {
        "task": "orchestration.task_queue.generate_daily_cost_report",
        "schedule": crontab(hour=23, minute=55),
    },
    "check-budget-alerts": {
        "task": "orchestration.task_queue.check_all_budget_alerts",
        "schedule": 900.0,  # Every 15 minutes
    },
}


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------
@celery_app.task(name="orchestration.task_queue.poll_all_pending_batches")
def poll_all_pending_batches() -> dict[str, int]:
    """Poll all pending batch jobs for results."""

    async def _poll() -> dict[str, int]:
        from core.batch_processor import BatchProcessor

        processor = BatchProcessor()
        await processor.initialize()
        pending = await processor.get_pending_jobs()
        results_count = 0

        for job in pending:
            results = await processor.poll_batch_results(job.job_id)
            results_count += len(results)

        return {"polled_jobs": len(pending), "results_collected": results_count}

    return asyncio.run(_poll())


@celery_app.task(name="orchestration.task_queue.generate_daily_cost_report")
def generate_daily_cost_report() -> dict[str, float]:
    """Generate daily cost reports for all active projects."""

    async def _report() -> dict[str, float]:
        from core.cost_tracker import CostTracker
        from memory.database import Database

        db = Database()
        await db.initialize()
        tracker = CostTracker()
        await tracker.initialize()

        try:
            projects = await db.list_projects(status="active")
            total_cost = 0.0

            for project in projects:
                daily = await tracker.get_daily_cost(project.id)
                total_cost += daily.total_cost_usd

            return {"total_daily_cost_usd": total_cost, "projects_count": len(projects)}
        finally:
            await db.close()

    return asyncio.run(_report())


@celery_app.task(name="orchestration.task_queue.check_all_budget_alerts")
def check_all_budget_alerts() -> dict[str, int]:
    """Check budget alerts for all active projects."""

    async def _check() -> dict[str, int]:
        from core.cost_tracker import CostTracker
        from memory.database import Database

        db = Database()
        await db.initialize()
        tracker = CostTracker()
        await tracker.initialize()

        try:
            projects = await db.list_projects(status="active")
            total_alerts = 0

            for project in projects:
                alerts = await tracker.check_budget_alerts(project.id)
                total_alerts += len(alerts)

            return {"projects_checked": len(projects), "alerts_triggered": total_alerts}
        finally:
            await db.close()

    return asyncio.run(_check())


@celery_app.task(name="orchestration.task_queue.execute_agent_task")
def execute_agent_task(
    agent_role: str,
    project_id: str,
    task_data: dict,
) -> dict:
    """Execute an agent task asynchronously via Celery."""

    async def _execute() -> dict:
        from agents.base_agent import Task, TaskType
        from core.config import AgentRole, TaskComplexity
        from orchestration.crew_manager import CrewManager

        manager = CrewManager()
        await manager.initialize()

        try:
            agents = manager._create_agents(project_id)

            agent = agents.get(agent_role)
            if not agent:
                return {"error": f"Unknown agent role: {agent_role}"}

            task = Task(
                id=task_data.get("id", f"celery_{agent_role}"),
                type=TaskType(task_data.get("type", "code_generation")),
                description=task_data.get("description", ""),
                complexity=TaskComplexity(task_data.get("complexity", "medium")),
                project_id=project_id,
                context=task_data.get("context", {}),
            )

            output = await agent.execute(task)
            return {
                "task_id": output.task_id,
                "content": output.content[:5000],
                "cost_usd": output.cost.real_cost_usd if output.cost else 0.0,
            }
        finally:
            await manager.shutdown()

    return asyncio.run(_execute())
