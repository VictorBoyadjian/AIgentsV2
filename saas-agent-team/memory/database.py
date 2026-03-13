"""
PostgreSQL database layer for persistent storage.

Manages project state, task history, and cost records via SQLAlchemy async.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, JSON, Boolean, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from core.config import get_settings
from core.cost_tracker import Base

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SQLAlchemy models
# ---------------------------------------------------------------------------
class ProjectRecord(Base):
    """Persistent project state."""

    __tablename__ = "projects"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="active")
    config_json = Column(JSON, nullable=True)
    monthly_budget_usd = Column(Float, nullable=False, default=200.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class TaskRecord(Base):
    """Persistent task history."""

    __tablename__ = "tasks"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(100), nullable=False, index=True)
    agent_role = Column(String(50), nullable=False)
    task_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="pending")
    complexity = Column(String(20), nullable=True)
    result_content = Column(Text, nullable=True)
    cost_usd = Column(Float, nullable=True)
    is_batch = Column(Boolean, nullable=False, default=False)
    batch_job_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------
class Database:
    """
    Async PostgreSQL database manager.

    Handles CRUD operations for projects, tasks, and related entities.
    """

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        db_url = database_url or settings.database.database_url
        self._engine = create_async_engine(
            db_url,
            pool_size=settings.database.db_pool_size,
            max_overflow=settings.database.db_max_overflow,
            pool_recycle=settings.database.db_pool_recycle,
        )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        """Create all tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database.initialized")

    async def close(self) -> None:
        """Dispose of the engine."""
        await self._engine.dispose()

    def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self._session_factory()

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------
    async def create_project(
        self,
        name: str,
        description: str = "",
        config: dict[str, Any] | None = None,
        budget: float = 200.0,
    ) -> ProjectRecord:
        """Create a new project."""
        async with self._session_factory() as session:
            project = ProjectRecord(
                name=name,
                description=description,
                config_json=config,
                monthly_budget_usd=budget,
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)
            logger.info("database.project_created", id=project.id, name=name)
            return project

    async def get_project(self, project_id: str) -> ProjectRecord | None:
        """Get a project by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ProjectRecord).where(ProjectRecord.id == project_id)
            )
            return result.scalar_one_or_none()

    async def list_projects(self, status: str = "active") -> list[ProjectRecord]:
        """List projects by status."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ProjectRecord).where(ProjectRecord.status == status)
            )
            return list(result.scalars().all())

    async def update_project_status(self, project_id: str, status: str) -> None:
        """Update project status."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ProjectRecord).where(ProjectRecord.id == project_id)
            )
            project = result.scalar_one_or_none()
            if project:
                project.status = status
                project.updated_at = datetime.now(timezone.utc)
                await session.commit()

    # ------------------------------------------------------------------
    # Task operations
    # ------------------------------------------------------------------
    async def create_task(
        self,
        project_id: str,
        agent_role: str,
        task_type: str,
        description: str = "",
        complexity: str = "medium",
    ) -> TaskRecord:
        """Create a task record."""
        async with self._session_factory() as session:
            task = TaskRecord(
                project_id=project_id,
                agent_role=agent_role,
                task_type=task_type,
                description=description,
                complexity=complexity,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task

    async def complete_task(
        self,
        task_id: str,
        result_content: str,
        cost_usd: float = 0.0,
    ) -> None:
        """Mark a task as completed."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(TaskRecord).where(TaskRecord.id == task_id)
            )
            task = result.scalar_one_or_none()
            if task:
                task.status = "completed"
                task.result_content = result_content
                task.cost_usd = cost_usd
                task.completed_at = datetime.now(timezone.utc)
                await session.commit()

    async def get_project_tasks(
        self,
        project_id: str,
        status: str | None = None,
    ) -> list[TaskRecord]:
        """Get tasks for a project."""
        async with self._session_factory() as session:
            query = select(TaskRecord).where(TaskRecord.project_id == project_id)
            if status:
                query = query.where(TaskRecord.status == status)
            result = await session.execute(query.order_by(TaskRecord.created_at.desc()))
            return list(result.scalars().all())
