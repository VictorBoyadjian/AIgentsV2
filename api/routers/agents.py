"""
Agent management endpoints.

Provides routes for executing tasks with specific agents and listing
available agent roles with their configured models.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from agents.base_agent import Task, TaskType
from api.dependencies import get_crew_manager
from api.schemas import (
    AgentExecuteRequest,
    AgentExecuteResponse,
    AgentRoleInfo,
)
from core.config import AgentRole, TaskComplexity, get_settings
from core.llm_router import ROUTING_TABLE
from orchestration.crew_manager import CrewManager

logger = structlog.get_logger(__name__)

router = APIRouter()

# Human-readable descriptions for each agent role.
_ROLE_DESCRIPTIONS: dict[str, str] = {
    "architect": "Designs system architecture, makes technology choices, and produces architecture documents.",
    "pm": "Creates PRDs, manages requirements, and coordinates feature planning.",
    "dev": "Generates production code, implements features, and performs code review.",
    "qa": "Writes unit and E2E tests, performs quality assurance, and validates functionality.",
    "security": "Audits code for vulnerabilities, checks dependencies, and ensures compliance.",
    "devops": "Generates Dockerfiles, CI/CD pipelines, and manages infrastructure configuration.",
    "research": "Performs competitor analysis, market research, and technology exploration.",
}


# ---------------------------------------------------------------------------
# POST /agents/execute  -- execute a task with a specific agent
# ---------------------------------------------------------------------------
@router.post(
    "/execute",
    response_model=AgentExecuteResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a task with a specific agent",
)
async def execute_agent_task(
    body: AgentExecuteRequest,
    crew_manager: CrewManager = Depends(get_crew_manager),
) -> AgentExecuteResponse:
    """Execute a task using the specified agent role.

    The agent is created with full infrastructure (LLM router, cost tracker,
    cache manager, batch processor) and runs the task through the standard
    pipeline including cost tracking and optional batch routing.
    """
    logger.info(
        "api.agent_execute",
        agent_role=body.agent_role,
        project_id=body.project_id,
        task_type=body.task_type,
        complexity=body.complexity,
    )

    # Map task_type string to TaskType enum
    try:
        task_type_enum = TaskType(body.task_type)
    except ValueError:
        valid = [t.value for t in TaskType]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid task_type '{body.task_type}'. Must be one of: {valid}",
        )

    # Map complexity string to TaskComplexity enum
    try:
        complexity_enum = TaskComplexity(body.complexity)
    except ValueError:
        valid = [c.value for c in TaskComplexity]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid complexity '{body.complexity}'. Must be one of: {valid}",
        )

    # Build agent set for the project
    agents = crew_manager._create_agents(body.project_id)
    agent = agents.get(body.agent_role)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent role '{body.agent_role}' not found.",
        )

    # Build the Task dataclass
    task = Task(
        id=str(uuid.uuid4()),
        type=task_type_enum,
        description=body.description,
        complexity=complexity_enum,
        project_id=body.project_id,
        context=body.context,
        is_blocking=body.blocking,
        allow_batch=not body.blocking,
    )

    try:
        output = await agent.execute(task)
    except Exception as exc:
        logger.error(
            "api.agent_execute_failed",
            agent_role=body.agent_role,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        ) from exc

    return AgentExecuteResponse(
        task_id=output.task_id,
        agent_role=output.agent_role,
        content=output.content,
        artifacts=output.artifacts,
        cost_usd=output.cost.real_cost_usd if output.cost else None,
        quality_score=output.quality_score,
        is_batch=output.metadata.get("is_batch", False),
        batch_job_id=output.metadata.get("batch_job_id"),
        metadata=output.metadata,
    )


# ---------------------------------------------------------------------------
# GET /agents/roles  -- list available agent roles
# ---------------------------------------------------------------------------
@router.get(
    "/roles",
    response_model=list[AgentRoleInfo],
    summary="List available agent roles",
)
async def list_agent_roles() -> list[AgentRoleInfo]:
    """Return all available agent roles with their configured models.

    Each entry includes the primary and fallback model names as well as a
    human-readable description of the agent's purpose.
    """
    roles: list[AgentRoleInfo] = []
    settings = get_settings()

    for role_name, route in ROUTING_TABLE.items():
        if role_name == "fallback":
            continue  # Internal-only role, not exposed to consumers
        roles.append(
            AgentRoleInfo(
                role=role_name,
                primary_model=route.primary,
                fallback_model=route.fallback,
                description=_ROLE_DESCRIPTIONS.get(role_name, ""),
            )
        )

    return roles
