"""
Task management endpoints.

Provides routes for listing and retrieving tasks within a project.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_database
from api.schemas import TaskResponse
from memory.database import Database

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /tasks/{project_id}  -- list tasks for a project
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}",
    response_model=list[TaskResponse],
    summary="List tasks for a project",
)
async def list_project_tasks(
    project_id: str,
    task_status: str | None = Query(
        default=None,
        alias="status",
        description="Filter by task status (pending, in_progress, completed, failed).",
    ),
    db: Database = Depends(get_database),
) -> list[TaskResponse]:
    """Return all tasks belonging to a project, optionally filtered by status.

    Results are ordered by creation time, newest first.
    """
    logger.info(
        "api.list_tasks",
        project_id=project_id,
        status=task_status,
    )

    # Verify project exists
    project = await db.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    records = await db.get_project_tasks(
        project_id=project_id,
        status=task_status,
    )

    return [TaskResponse.model_validate(r) for r in records]


# ---------------------------------------------------------------------------
# GET /tasks/{project_id}/{task_id}  -- get a specific task
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}/{task_id}",
    response_model=TaskResponse,
    summary="Get a specific task",
)
async def get_task(
    project_id: str,
    task_id: str,
    db: Database = Depends(get_database),
) -> TaskResponse:
    """Retrieve a single task by project ID and task ID."""
    logger.info(
        "api.get_task",
        project_id=project_id,
        task_id=task_id,
    )

    # Verify project exists
    project = await db.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    # Fetch all tasks and find the matching one
    records = await db.get_project_tasks(project_id=project_id)
    task_record = next((r for r in records if r.id == task_id), None)

    if task_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found in project '{project_id}'.",
        )

    return TaskResponse.model_validate(task_record)
