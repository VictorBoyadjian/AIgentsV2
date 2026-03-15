"""
Project CRUD and workflow endpoints.

Provides routes for creating, listing, and retrieving projects, as well as
starting and monitoring development workflows.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.dependencies import get_cache, get_crew_manager, get_database, get_human_in_loop
from api.schemas import (
    CheckpointResolveRequest,
    CheckpointResponse,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    WorkflowStartRequest,
    WorkflowStatusResponse,
)
from memory.cache import MemoryCache
from memory.database import Database
from orchestration.crew_manager import CrewManager
from orchestration.human_in_loop import HumanInTheLoop

logger = structlog.get_logger(__name__)

router = APIRouter()

# In-memory store for active workflow states keyed by project_id.
# In production this would be backed by Redis or the database.
_active_workflows: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# POST /projects  -- create a project
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new project",
)
async def create_project(
    body: ProjectCreate,
    db: Database = Depends(get_database),
) -> ProjectResponse:
    """Create a new project in the database.

    Returns the newly created project with its generated UUID.
    """
    logger.info("api.create_project", name=body.name)

    try:
        record = await db.create_project(
            name=body.name,
            description=body.description,
            config=body.config,
            budget=body.budget,
        )
    except Exception as exc:
        logger.error("api.create_project_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create project: {exc}",
        ) from exc

    return ProjectResponse.model_validate(record)


# ---------------------------------------------------------------------------
# GET /projects  -- list projects
# ---------------------------------------------------------------------------
@router.get(
    "",
    response_model=ProjectListResponse,
    summary="List projects",
)
async def list_projects(
    project_status: str = "active",
    db: Database = Depends(get_database),
) -> ProjectListResponse:
    """List projects, optionally filtered by status."""
    logger.info("api.list_projects", status=project_status)

    records = await db.list_projects(status=project_status)
    projects = [ProjectResponse.model_validate(r) for r in records]

    return ProjectListResponse(projects=projects, total=len(projects))


# ---------------------------------------------------------------------------
# GET /projects/{project_id}  -- get a single project
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get project by ID",
)
async def get_project(
    project_id: str,
    db: Database = Depends(get_database),
) -> ProjectResponse:
    """Retrieve a single project by its ID."""
    record = await db.get_project(project_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )
    return ProjectResponse.model_validate(record)


_WORKFLOW_TTL = 86400  # 24 hours


async def _persist_to_redis(cache: MemoryCache, project_id: str, state: dict) -> None:
    """Write workflow state to Redis so all workers can read it."""
    try:
        await cache.set_json(f"workflow:{project_id}", state, ttl=_WORKFLOW_TTL)
    except Exception as exc:
        logger.warning("api.workflow_redis_persist_failed", error=str(exc))


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/workflow  -- start workflow
# ---------------------------------------------------------------------------
async def _run_workflow_background(
    project_id: str,
    product_idea: str,
    target_market: str,
    crew_manager: CrewManager,
    cache: MemoryCache,
) -> None:
    """Run the full development workflow in a background task."""
    try:
        logger.info("api.workflow_background_start", project_id=project_id)
        result = await crew_manager.run_workflow(
            project_id=project_id,
            product_idea=product_idea,
            target_market=target_market,
            cache=cache,
        )
        final_state = {
            "current_phase": result.current_phase.value,
            "budget_ok": result.budget_ok,
            "design_approved": result.design_approved,
            "deploy_approved": result.deploy_approved,
            "completed_features": result.completed_features,
            "pending_batch_jobs": result.pending_batch_jobs,
            "errors": result.errors,
            "cost_report": result.cost_report,
            "github_repo_url": result.github_repo_url,
            "railway_deployment_url": result.railway_deployment_url,
        }
        _active_workflows[project_id] = final_state
        await _persist_to_redis(cache, project_id, final_state)
        logger.info(
            "api.workflow_background_complete",
            project_id=project_id,
            phase=result.current_phase.value,
        )
    except Exception as exc:
        logger.error(
            "api.workflow_background_failed",
            project_id=project_id,
            error=str(exc),
        )
        error_state = {
            "current_phase": "failed",
            "budget_ok": False,
            "design_approved": False,
            "deploy_approved": False,
            "completed_features": [],
            "pending_batch_jobs": [],
            "errors": [str(exc)],
            "cost_report": "",
            "github_repo_url": "",
            "railway_deployment_url": "",
        }
        _active_workflows[project_id] = error_state
        await _persist_to_redis(cache, project_id, error_state)


@router.post(
    "/{project_id}/workflow",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start development workflow",
)
async def start_workflow(
    project_id: str,
    body: WorkflowStartRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database),
    crew_manager: CrewManager = Depends(get_crew_manager),
    cache: MemoryCache = Depends(get_cache),
) -> WorkflowStatusResponse:
    """Start the full SaaS development workflow for a project.

    The workflow runs asynchronously in a background task.  Poll
    ``GET /projects/{id}/status`` to track progress.
    """
    # Verify project exists
    project = await db.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    # Check for an already-running workflow (in-memory OR Redis for multi-worker)
    current = _active_workflows.get(project_id)
    if current is None:
        current = await cache.get_json(f"workflow:{project_id}")
    if current and current.get("current_phase") not in ("completed", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A workflow is already running for this project.",
        )

    # Mark as starting — write to both in-memory and Redis
    initial_state: dict = {
        "current_phase": "budget_check",
        "budget_ok": True,
        "design_approved": False,
        "deploy_approved": False,
        "completed_features": [],
        "pending_batch_jobs": [],
        "errors": [],
        "cost_report": "",
        "github_repo_url": "",
        "railway_deployment_url": "",
    }
    _active_workflows[project_id] = initial_state
    await _persist_to_redis(cache, project_id, initial_state)

    background_tasks.add_task(
        _run_workflow_background,
        project_id,
        body.product_idea,
        body.target_market,
        crew_manager,
        cache,
    )

    logger.info("api.workflow_started", project_id=project_id)

    return WorkflowStatusResponse(
        project_id=project_id,
        current_phase="budget_check",
    )


# ---------------------------------------------------------------------------
# GET /projects/{project_id}/status  -- workflow status
# ---------------------------------------------------------------------------
@router.get(
    "/{project_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Get workflow status",
)
async def get_workflow_status(
    project_id: str,
    db: Database = Depends(get_database),
    cache: MemoryCache = Depends(get_cache),
) -> WorkflowStatusResponse:
    """Get the current status of a project's development workflow."""
    # Verify project exists
    project = await db.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    # Check Redis first — it is updated after every phase transition by the
    # crew_manager phase callback, so it always holds the freshest state.
    # Fall back to the in-memory dict only when Redis is unavailable.
    state = await cache.get_json(f"workflow:{project_id}")
    if state is None:
        state = _active_workflows.get(project_id)
    if state is None:
        return WorkflowStatusResponse(
            project_id=project_id,
            current_phase="not_started",
        )

    return WorkflowStatusResponse(
        project_id=project_id,
        current_phase=state.get("current_phase", "unknown"),
        budget_ok=state.get("budget_ok", True),
        design_approved=state.get("design_approved", False),
        deploy_approved=state.get("deploy_approved", False),
        completed_features=state.get("completed_features", []),
        pending_batch_jobs=state.get("pending_batch_jobs", []),
        errors=state.get("errors", []),
        cost_report=state.get("cost_report", ""),
        github_repo_url=state.get("github_repo_url", ""),
        railway_deployment_url=state.get("railway_deployment_url", ""),
    )
