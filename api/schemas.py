"""
Pydantic schemas for all API request/response models.

Defines validated schemas for projects, tasks, agents, costs, workflows,
and webhooks. Every field includes type hints and where applicable,
validators and field constraints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums used across schemas
# ---------------------------------------------------------------------------
class ProjectStatus(str, Enum):
    """Valid project statuses."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Valid task statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowPhaseEnum(str, Enum):
    """Workflow phases exposed via the API."""

    BUDGET_CHECK = "budget_check"
    CACHE_WARMUP = "cache_warmup"
    RESEARCH = "research"
    DESIGN = "design"
    HUMAN_REVIEW_DESIGN = "human_review_design"
    DEVELOPMENT = "development"
    BATCH_COLLECTION = "batch_collection"
    INTEGRATION = "integration"
    DEPLOY = "deploy"
    HUMAN_REVIEW_DEPLOY = "human_review_deploy"
    COST_REPORT = "cost_report"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Project schemas
# ---------------------------------------------------------------------------
class ProjectCreate(BaseModel):
    """Schema for creating a new project."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable project name.",
    )
    description: str = Field(
        default="",
        max_length=5000,
        description="Detailed project description.",
    )
    budget: float = Field(
        default=200.0,
        ge=0.0,
        le=10000.0,
        description="Monthly budget in USD for LLM costs.",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional project-level configuration overrides.",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        """Ensure name is not just whitespace."""
        if not v.strip():
            raise ValueError("Project name must not be blank.")
        return v.strip()


class ProjectResponse(BaseModel):
    """Schema for a single project response."""

    id: str = Field(..., description="Unique project identifier (UUID).")
    name: str
    description: str | None = None
    status: str = "active"
    config: dict[str, Any] | None = Field(default=None, alias="config_json")
    monthly_budget_usd: float = 200.0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class ProjectListResponse(BaseModel):
    """Paginated list of projects."""

    projects: list[ProjectResponse]
    total: int = Field(..., description="Total number of matching projects.")


# ---------------------------------------------------------------------------
# Task schemas
# ---------------------------------------------------------------------------
class TaskCreate(BaseModel):
    """Schema for creating a task within a project."""

    agent_role: str = Field(
        ...,
        description="Agent role to execute the task (architect, pm, dev, qa, etc.).",
    )
    task_type: str = Field(
        ...,
        description="Type of task (code_generation, test_generation, etc.).",
    )
    description: str = Field(
        default="",
        max_length=10000,
        description="Detailed description of the task.",
    )
    complexity: str = Field(
        default="medium",
        description="Task complexity: simple, medium, complex, or critical.",
    )

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        """Validate complexity is one of the allowed values."""
        allowed = {"simple", "medium", "complex", "critical"}
        if v not in allowed:
            raise ValueError(f"complexity must be one of {allowed}")
        return v


class TaskResponse(BaseModel):
    """Schema for a single task response."""

    id: str
    project_id: str
    agent_role: str
    task_type: str
    description: str | None = None
    status: str = "pending"
    complexity: str | None = None
    result_content: str | None = None
    cost_usd: float | None = None
    is_batch: bool = False
    batch_job_id: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Agent schemas
# ---------------------------------------------------------------------------
class AgentExecuteRequest(BaseModel):
    """Request to execute a task with a specific agent."""

    agent_role: str = Field(
        ...,
        description="Agent role: architect, pm, dev, qa, security, devops, research.",
    )
    project_id: str = Field(
        ...,
        description="Project ID to execute the task within.",
    )
    task_type: str = Field(
        default="code_generation",
        description="Type of task to execute.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Task description / prompt for the agent.",
    )
    complexity: str = Field(
        default="medium",
        description="Task complexity: simple, medium, complex, critical.",
    )
    blocking: bool = Field(
        default=True,
        description="If False, eligible tasks may be routed to Batch API.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context passed to the agent (e.g. existing_code).",
    )

    @field_validator("agent_role")
    @classmethod
    def validate_agent_role(cls, v: str) -> str:
        """Validate agent role is recognized."""
        allowed = {"architect", "pm", "dev", "qa", "security", "devops", "research"}
        if v not in allowed:
            raise ValueError(f"agent_role must be one of {allowed}")
        return v


class AgentExecuteResponse(BaseModel):
    """Response from agent task execution."""

    task_id: str
    agent_role: str
    content: str
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Generated file artifacts (filename -> content).",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Cost of this execution in USD.",
    )
    quality_score: float | None = Field(
        default=None,
        description="Self-assessed quality score 0.0-1.0.",
    )
    is_batch: bool = Field(
        default=False,
        description="True if the task was submitted to Batch API.",
    )
    batch_job_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRoleInfo(BaseModel):
    """Information about an available agent role."""

    role: str
    primary_model: str
    fallback_model: str
    description: str


# ---------------------------------------------------------------------------
# Cost schemas
# ---------------------------------------------------------------------------
class CostSummaryResponse(BaseModel):
    """Aggregated cost summary for a time period."""

    project_id: str
    period: str = Field(..., description="Period label: day, week, or month.")
    start: str
    end: str
    real_cost_usd: float
    theoretical_cost_usd: float
    cache_savings_usd: float
    batch_savings_usd: float
    total_savings_usd: float
    total_calls: int


class CostBreakdownResponse(BaseModel):
    """Cost breakdown by agent role for a project."""

    project_id: str
    date: str
    total_cost_usd: float
    total_calls: int
    cost_by_agent: dict[str, float] = Field(
        default_factory=dict,
        description="Cost in USD keyed by agent role.",
    )
    cache_savings_usd: float
    batch_savings_usd: float
    total_savings_usd: float


class SavingsDetail(BaseModel):
    """Dollar and percentage savings from a single optimization."""

    source: str = Field(..., description="'cache' or 'batch'.")
    saved_usd: float
    percentage: float = Field(
        ...,
        description="Percentage saved vs. theoretical cost.",
    )


class CostSavingsResponse(BaseModel):
    """Savings achieved through cache and batch optimizations."""

    project_id: str
    period_start: str
    period_end: str
    real_cost_usd: float
    theoretical_cost_usd: float
    savings: list[SavingsDetail]
    total_saved_usd: float
    total_saved_percentage: float


class CostProjectionResponse(BaseModel):
    """Monthly cost projection with trend information."""

    project_id: str
    current_month_cost: float
    projected_month_cost: float
    daily_average: float
    days_remaining: int
    budget_usd: float
    projected_over_budget: bool
    budget_usage_ratio: float = Field(
        ...,
        description="Ratio of current spend to monthly budget (0.0 - 1.0+).",
    )


class BudgetSetRequest(BaseModel):
    """Request to set a project's monthly budget."""

    monthly_budget_usd: float = Field(
        ...,
        gt=0.0,
        le=50000.0,
        description="New monthly budget in USD.",
    )


class BudgetSetResponse(BaseModel):
    """Confirmation after setting a project budget."""

    project_id: str
    monthly_budget_usd: float
    message: str = "Budget updated successfully."


class OptimizationTipResponse(BaseModel):
    """A single cost optimization suggestion."""

    category: str = Field(
        ...,
        description="Tip category: model_downgrade, cache_utilization, batch_eligible.",
    )
    description: str
    estimated_monthly_savings_usd: float
    agent_role: str | None = None


# ---------------------------------------------------------------------------
# Workflow schemas
# ---------------------------------------------------------------------------
class WorkflowStartRequest(BaseModel):
    """Request to start a development workflow for a project."""

    product_idea: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The product/feature idea to build.",
    )
    target_market: str = Field(
        default="B2B SaaS",
        max_length=200,
        description="Target market segment.",
    )


class WorkflowStatusResponse(BaseModel):
    """Current status of a project workflow."""

    project_id: str
    current_phase: str
    budget_ok: bool = True
    design_approved: bool = False
    deploy_approved: bool = False
    completed_features: list[str] = Field(default_factory=list)
    pending_batch_jobs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    cost_report: str = ""
    github_repo_url: str = ""
    railway_deployment_url: str = ""


# ---------------------------------------------------------------------------
# Checkpoint schemas
# ---------------------------------------------------------------------------
class CheckpointResolveRequest(BaseModel):
    """Request to resolve a human-in-the-loop checkpoint."""

    checkpoint_id: str = Field(
        ...,
        description="ID of the checkpoint to resolve.",
    )
    approved: bool = Field(
        ...,
        description="Whether the checkpoint is approved or rejected.",
    )
    feedback: str = Field(
        default="",
        max_length=5000,
        description="Optional reviewer feedback.",
    )
    modifications: dict[str, Any] | None = Field(
        default=None,
        description="Optional modifications to apply.",
    )


class CheckpointResponse(BaseModel):
    """Response for a checkpoint resolution."""

    checkpoint_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Webhook schemas
# ---------------------------------------------------------------------------
class WebhookEvent(BaseModel):
    """Incoming webhook event payload."""

    event_type: str = Field(
        ...,
        description="Type of webhook event (push, pull_request, etc.).",
    )
    repository: str = Field(
        default="",
        description="Repository full name (owner/repo).",
    )
    action: str = Field(
        default="",
        description="Event action (opened, closed, etc.).",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Full event payload.",
    )
    sender: str = Field(
        default="",
        description="User or service that triggered the event.",
    )


class WebhookResponse(BaseModel):
    """Response acknowledging a webhook event."""

    received: bool = True
    event_type: str
    message: str = "Webhook processed."


# ---------------------------------------------------------------------------
# Health / generic
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    environment: str = "development"
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components.",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str | None = None
    timestamp: datetime | None = None
