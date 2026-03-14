"""
Batch API processor for Anthropic and OpenAI.

Routes non-urgent tasks through batch endpoints for 50% cost savings.
Results are polled periodically and stored in PostgreSQL.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import httpx
import structlog
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, select, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from core.config import BatchPriority, get_settings
from core.cost_tracker import Base

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Batch-eligible task categories
# ---------------------------------------------------------------------------
BATCH_ELIGIBLE_TASKS: set[str] = {
    "generate_tests",
    "generate_documentation",
    "security_review_non_blocking",
    "generate_readme",
    "generate_changelog",
    "analyze_dependencies",
    "generate_cicd_pipeline",
    "generate_iac",
    "generate_test_plan",
    "write_e2e_tests",
    "generate_security_checklist",
    "check_dependencies",
    "analyze_competitors",
    "prioritize_backlog",
    "generate_roadmap",
    "provision_infrastructure",
}

REALTIME_ONLY_TASKS: set[str] = {
    "generate_code_feature",
    "human_response",
    "debug_urgent",
    "deploy",
    "run_tests",
    "fix_bug",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class BatchStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass
class BatchTask:
    """A single task to include in a batch submission."""

    custom_id: str
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 4096
    temperature: float = 0.7
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a single task within a completed batch."""

    custom_id: str
    status: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class BatchJob:
    """Represents a submitted batch job."""

    job_id: str
    provider: str
    batch_api_id: str | None  # ID from Anthropic/OpenAI
    status: BatchStatus
    task_count: int
    submitted_at: datetime
    completed_at: datetime | None = None
    results: list[BatchResult] = field(default_factory=list)
    estimated_savings_usd: float = 0.0


@dataclass
class BatchSavingsReport:
    """Report of cost savings achieved via batch processing."""

    period_days: int
    total_batch_calls: int
    total_realtime_equivalent_cost: float
    total_batch_cost: float
    total_savings_usd: float
    savings_percentage: float


# ---------------------------------------------------------------------------
# SQLAlchemy model for batch jobs
# ---------------------------------------------------------------------------
class BatchJobRecord(Base):
    """Persistent record of batch jobs."""

    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    provider = Column(String(20), nullable=False)
    batch_api_id = Column(String(200), nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    task_count = Column(Integer, nullable=False)
    project_id = Column(String(100), nullable=False, index=True)
    priority = Column(String(20), nullable=False, default="normal")
    results_json = Column(JSON, nullable=True)
    estimated_savings_usd = Column(Float, nullable=False, default=0.0)
    submitted_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------
class BatchProcessor:
    """
    Processes non-urgent LLM tasks via Batch API for 50% cost savings.

    Eligible tasks (non-blocking):
    - Test generation, documentation, security checklists
    - README/CHANGELOG, dependency analysis
    - DevOps scripts, IaC templates, CI/CD pipelines

    Non-eligible tasks (real-time required):
    - Feature code generation, human responses, urgent debugging
    - Deployment, running tests, critical bug fixes
    """

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        db_url = database_url or settings.database.database_url
        self._engine = create_async_engine(db_url, pool_size=5, max_overflow=5)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._anthropic_key = settings.llm.anthropic_api_key
        self._openai_key = settings.llm.openai_api_key
        self._polling_interval = 1800  # 30 minutes

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("batch_processor.initialized")

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    def should_use_batch(self, task_type: str, is_blocking: bool = False) -> bool:
        """
        Decide whether a task should use batch or real-time processing.

        Rule: if the task is not on the critical path of the workflow → batch.
        """
        if is_blocking:
            return False
        if task_type in REALTIME_ONLY_TASKS:
            return False
        if task_type in BATCH_ELIGIBLE_TASKS:
            return True
        # Default: real-time for unknown tasks
        return False

    # ------------------------------------------------------------------
    # Submit batch
    # ------------------------------------------------------------------
    async def submit_batch(
        self,
        tasks: list[BatchTask],
        provider: Literal["anthropic", "openai"],
        project_id: str,
        priority: BatchPriority = BatchPriority.NORMAL,
    ) -> BatchJob:
        """
        Submit a batch of tasks to Anthropic or OpenAI Batch API.

        Anthropic: processes within 24h, 50% cheaper.
        OpenAI: processes within 24h, 50% cheaper.
        """
        job_id = f"batch_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        if provider == "anthropic":
            batch_api_id = await self._submit_anthropic_batch(tasks)
        else:
            batch_api_id = await self._submit_openai_batch(tasks)

        job = BatchJob(
            job_id=job_id,
            provider=provider,
            batch_api_id=batch_api_id,
            status=BatchStatus.PENDING,
            task_count=len(tasks),
            submitted_at=now,
        )

        # Persist to DB
        async with self._session_factory() as session:
            record = BatchJobRecord(
                job_id=job_id,
                provider=provider,
                batch_api_id=batch_api_id,
                status=BatchStatus.PENDING.value,
                task_count=len(tasks),
                project_id=project_id,
                priority=priority.value,
                submitted_at=now,
            )
            session.add(record)
            await session.commit()

        logger.info(
            "batch_processor.submitted",
            job_id=job_id,
            provider=provider,
            task_count=len(tasks),
            batch_api_id=batch_api_id,
        )

        return job

    async def _submit_anthropic_batch(self, tasks: list[BatchTask]) -> str | None:
        """Submit batch to Anthropic Message Batches API."""
        requests = []
        for task in tasks:
            request_body: dict[str, Any] = {
                "model": task.model,
                "max_tokens": task.max_tokens,
                "temperature": task.temperature,
                "messages": task.messages,
            }
            if task.system:
                request_body["system"] = task.system

            requests.append({
                "custom_id": task.custom_id,
                "params": request_body,
            })

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages/batches",
                json={"requests": requests},
                headers={
                    "x-api-key": self._anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "anthropic-beta": "message-batches-2024-09-24",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("id")  # type: ignore[no-any-return]

    async def _submit_openai_batch(self, tasks: list[BatchTask]) -> str | None:
        """Submit batch to OpenAI Batch API."""
        # OpenAI batch requires JSONL upload
        jsonl_lines = []
        for task in tasks:
            body: dict[str, Any] = {
                "model": task.model,
                "messages": task.messages,
                "max_tokens": task.max_tokens,
                "temperature": task.temperature,
            }
            jsonl_lines.append(json.dumps({
                "custom_id": task.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))

        jsonl_content = "\n".join(jsonl_lines)

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Upload file
            upload_response = await client.post(
                "https://api.openai.com/v1/files",
                headers={"Authorization": f"Bearer {self._openai_key}"},
                files={"file": ("batch_input.jsonl", jsonl_content.encode(), "application/jsonl")},
                data={"purpose": "batch"},
            )
            upload_response.raise_for_status()
            file_id = upload_response.json()["id"]

            # Create batch
            batch_response = await client.post(
                "https://api.openai.com/v1/batches",
                headers={
                    "Authorization": f"Bearer {self._openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            batch_response.raise_for_status()
            return batch_response.json().get("id")  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Poll results
    # ------------------------------------------------------------------
    async def poll_batch_results(self, job_id: str) -> list[BatchResult]:
        """
        Poll batch job status and retrieve results when complete.

        This is called periodically (every 30 minutes) by the Celery beat scheduler.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(BatchJobRecord).where(BatchJobRecord.job_id == job_id)
            )
            record = result.scalar_one_or_none()
            if record is None:
                logger.error("batch_processor.job_not_found", job_id=job_id)
                return []

            if record.status in (BatchStatus.COMPLETED.value, BatchStatus.FAILED.value):
                # Already processed
                if record.results_json:
                    return [BatchResult(**r) for r in record.results_json]
                return []

            if record.provider == "anthropic":
                results, status = await self._poll_anthropic(record.batch_api_id or "")
            else:
                results, status = await self._poll_openai(record.batch_api_id or "")

            record.status = status.value
            if status == BatchStatus.COMPLETED:
                record.completed_at = datetime.now(timezone.utc)
                record.results_json = [
                    {
                        "custom_id": r.custom_id,
                        "status": r.status,
                        "content": r.content,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "error": r.error,
                    }
                    for r in results
                ]

            await session.commit()

            logger.info(
                "batch_processor.polled",
                job_id=job_id,
                status=status.value,
                results_count=len(results),
            )

            return results

    async def _poll_anthropic(self, batch_api_id: str) -> tuple[list[BatchResult], BatchStatus]:
        """Poll Anthropic batch status."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"https://api.anthropic.com/v1/messages/batches/{batch_api_id}",
                headers={
                    "x-api-key": self._anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "message-batches-2024-09-24",
                },
            )
            response.raise_for_status()
            data = response.json()

            processing_status = data.get("processing_status", "in_progress")

            if processing_status != "ended":
                return [], BatchStatus.IN_PROGRESS

            # Fetch results
            results_response = await client.get(
                f"https://api.anthropic.com/v1/messages/batches/{batch_api_id}/results",
                headers={
                    "x-api-key": self._anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "message-batches-2024-09-24",
                },
            )
            results_response.raise_for_status()

            results: list[BatchResult] = []
            for line in results_response.text.strip().split("\n"):
                if not line.strip():
                    continue
                item = json.loads(line)
                result_data = item.get("result", {})
                message = result_data.get("message", {})
                content_blocks = message.get("content", [])
                content_text = " ".join(
                    b.get("text", "") for b in content_blocks if b.get("type") == "text"
                )
                usage = message.get("usage", {})

                results.append(BatchResult(
                    custom_id=item.get("custom_id", ""),
                    status=result_data.get("type", "unknown"),
                    content=content_text,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    error=result_data.get("error", {}).get("message") if result_data.get("type") == "errored" else None,
                ))

            return results, BatchStatus.COMPLETED

    async def _poll_openai(self, batch_api_id: str) -> tuple[list[BatchResult], BatchStatus]:
        """Poll OpenAI batch status."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"https://api.openai.com/v1/batches/{batch_api_id}",
                headers={"Authorization": f"Bearer {self._openai_key}"},
            )
            response.raise_for_status()
            data = response.json()

            status_str = data.get("status", "in_progress")
            if status_str in ("validating", "in_progress", "finalizing"):
                return [], BatchStatus.IN_PROGRESS
            if status_str in ("failed", "expired"):
                return [], BatchStatus(status_str)
            if status_str == "cancelled":
                return [], BatchStatus.CANCELLED

            # Completed — download results
            output_file_id = data.get("output_file_id")
            if not output_file_id:
                return [], BatchStatus.FAILED

            file_response = await client.get(
                f"https://api.openai.com/v1/files/{output_file_id}/content",
                headers={"Authorization": f"Bearer {self._openai_key}"},
            )
            file_response.raise_for_status()

            results: list[BatchResult] = []
            for line in file_response.text.strip().split("\n"):
                if not line.strip():
                    continue
                item = json.loads(line)
                resp_body = item.get("response", {}).get("body", {})
                choices = resp_body.get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                usage = resp_body.get("usage", {})

                results.append(BatchResult(
                    custom_id=item.get("custom_id", ""),
                    status="succeeded" if item.get("response", {}).get("status_code") == 200 else "failed",
                    content=content,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    error=item.get("error", {}).get("message") if item.get("error") else None,
                ))

            return results, BatchStatus.COMPLETED

    # ------------------------------------------------------------------
    # Pending jobs listing
    # ------------------------------------------------------------------
    async def get_pending_jobs(self, project_id: str | None = None) -> list[BatchJob]:
        """List all pending/in-progress batch jobs, optionally filtered by project."""
        async with self._session_factory() as session:
            query = select(BatchJobRecord).where(
                BatchJobRecord.status.in_([BatchStatus.PENDING.value, BatchStatus.IN_PROGRESS.value])
            )
            if project_id:
                query = query.where(BatchJobRecord.project_id == project_id)

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                BatchJob(
                    job_id=r.job_id,
                    provider=r.provider,
                    batch_api_id=r.batch_api_id,
                    status=BatchStatus(r.status),
                    task_count=r.task_count,
                    submitted_at=r.submitted_at,
                    completed_at=r.completed_at,
                )
                for r in records
            ]

    # ------------------------------------------------------------------
    # Savings report
    # ------------------------------------------------------------------
    async def get_batch_savings(self, project_id: str, period_days: int = 30) -> BatchSavingsReport:
        """Calculate cost savings achieved through batch processing over a period."""
        from core.cost_tracker import PRICING_TABLE

        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=period_days)

        async with self._session_factory() as session:
            result = await session.execute(
                select(BatchJobRecord).where(
                    BatchJobRecord.project_id == project_id,
                    BatchJobRecord.status == BatchStatus.COMPLETED.value,
                    BatchJobRecord.submitted_at >= cutoff,
                )
            )
            records = result.scalars().all()

            total_batch_calls = 0
            total_input_tokens = 0
            total_output_tokens = 0
            models_used: dict[str, tuple[int, int]] = {}

            for record in records:
                if not record.results_json:
                    continue
                for r in record.results_json:
                    total_batch_calls += 1
                    total_input_tokens += r.get("input_tokens", 0)
                    total_output_tokens += r.get("output_tokens", 0)

            # Estimate costs (using average model pricing)
            avg_input_price = 3.0  # middle-of-road estimate per million
            avg_output_price = 15.0

            realtime_cost = (
                total_input_tokens * avg_input_price / 1_000_000
                + total_output_tokens * avg_output_price / 1_000_000
            )
            batch_cost = realtime_cost * 0.5  # 50% discount
            savings = realtime_cost - batch_cost

            return BatchSavingsReport(
                period_days=period_days,
                total_batch_calls=total_batch_calls,
                total_realtime_equivalent_cost=realtime_cost,
                total_batch_cost=batch_cost,
                total_savings_usd=savings,
                savings_percentage=50.0 if realtime_cost > 0 else 0.0,
            )
