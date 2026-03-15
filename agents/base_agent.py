"""
Abstract base agent with integrated cost tracking, prompt caching,
batch processing, and LLM routing.

Every agent inherits from BaseAgent, which provides the full pipeline:
routing → caching → batch decision → LLM call → cost tracking → budget check.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from core.batch_processor import BatchProcessor, BatchTask
from core.cache_manager import CacheManager
from core.config import AgentRole, BatchPriority, BudgetAlertLevel, TaskComplexity, get_settings
from core.cost_tracker import CallCost, CostTracker
from core.llm_router import LLMRouter
from observability.langsmith_tracer import LangSmithTracer

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    """Categories of tasks agents can perform."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    RESEARCH = "research"
    DEPLOYMENT = "deployment"
    SECURITY_AUDIT = "security_audit"
    PROJECT_MANAGEMENT = "project_management"


@dataclass
class Task:
    """A task to be executed by an agent."""

    id: str
    type: TaskType
    description: str
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    project_id: str = "default"
    context: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    is_blocking: bool = True
    allow_batch: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentOutput:
    """Output from an agent's task execution."""

    task_id: str
    agent_role: str
    content: str
    artifacts: dict[str, str] = field(default_factory=dict)  # filename → content
    cost: CallCost | None = None
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success(self) -> bool:
        """Whether the agent produced useful output."""
        return len(self.content) > 0


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------
class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides integrated pipeline for every LLM call:
    1. Evaluate if the task can go to batch → BatchProcessor
    2. Build messages with cache_control → CacheManager
    3. Downgrade model if task is simple → LLMRouter
    4. Call the LLM
    5. Record the real cost → CostTracker
    6. Check budget alerts → CostTracker.check_budget_alerts()
    7. Self-critique the result → self.self_reflect()
    """

    def __init__(
        self,
        role: AgentRole,
        project_id: str = "default",
        llm_router: LLMRouter | None = None,
        cache_manager: CacheManager | None = None,
        cost_tracker: CostTracker | None = None,
        batch_processor: BatchProcessor | None = None,
    ) -> None:
        self.role = role
        self.project_id = project_id
        self.llm_router = llm_router or LLMRouter()
        self.cache_manager = cache_manager or CacheManager()
        self.cost_tracker = cost_tracker or CostTracker()
        self.batch_processor = batch_processor or BatchProcessor()
        self._tracer = LangSmithTracer()

        self._system_prompt = self.cache_manager.get_system_prompt(role)
        self._project_context: str | None = None
        self._emergency_mode = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    async def execute(self, task: Task) -> AgentOutput:
        """
        Execute a task through the full agent pipeline.

        Steps:
        1. Check if task can go to batch
        2. Build cached messages
        3. Route to appropriate model
        4. Call LLM
        5. Track costs
        6. Check budget
        7. Self-reflect on quality
        """
        logger.info(
            "agent.execute_start",
            agent=self.role.value,
            task_id=task.id,
            task_type=task.type.value,
            complexity=task.complexity.value,
        )

        # Step 1: Check batch eligibility
        settings = get_settings()
        if (
            settings.llm.enable_batch_api
            and task.allow_batch
            and not task.is_blocking
            and self.batch_processor.should_use_batch(task.type.value)
        ):
            return await self._execute_batch(task)

        # Steps 2-7: Real-time execution
        return await self._execute_realtime(task)

    def set_project_context(self, context: str) -> None:
        """Set the project context (architecture, PRD, etc.) for caching."""
        self._project_context = context

    # ------------------------------------------------------------------
    # Real-time execution pipeline
    # ------------------------------------------------------------------
    async def _execute_realtime(self, task: Task) -> AgentOutput:
        """Execute a task in real-time with caching and cost tracking."""
        async with self._tracer.trace_agent_call(
            agent_role=self.role.value,
            task_id=task.id,
            task_type=task.type.value,
            complexity=task.complexity.value,
            project_id=self.project_id,
        ) as span:
            # Build the task-specific prompt
            task_prompt = self._build_task_prompt(task)

            # Call LLM with full tracking
            content, cost = await self._call_llm_with_tracking(
                current_task=task_prompt,
                task_complexity=task.complexity,
                existing_code=task.context.get("existing_code"),
            )

            # Parse output
            artifacts = self._extract_artifacts(content)

            # Self-reflect if quality matters
            quality_score = None
            if task.complexity in (TaskComplexity.COMPLEX, TaskComplexity.CRITICAL):
                quality_score = await self._self_reflect(task, content)

            output = AgentOutput(
                task_id=task.id,
                agent_role=self.role.value,
                content=content,
                artifacts=artifacts,
                cost=cost,
                quality_score=quality_score,
            )

            span.end(output_text=content[:500], quality_score=quality_score)

            logger.info(
                "agent.execute_complete",
                agent=self.role.value,
                task_id=task.id,
                cost_usd=round(cost.real_cost_usd, 6) if cost else 0,
                quality=quality_score,
            )

            return output

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------
    async def _execute_batch(self, task: Task) -> AgentOutput:
        """Submit a task to batch processing."""
        task_prompt = self._build_task_prompt(task)
        model = self.llm_router.resolve_model(self.role.value, task.complexity)

        messages = self.cache_manager.build_cached_messages_litellm(
            system_prompt=self._system_prompt,
            project_context=self._project_context,
            existing_code=task.context.get("existing_code"),
            current_task=task_prompt,
        )

        batch_task = BatchTask(
            custom_id=task.id,
            model=model,
            messages=messages,
            system=self._system_prompt,
        )

        provider = "anthropic" if model.startswith("claude") else "openai"
        job = await self.batch_processor.submit_batch(
            tasks=[batch_task],
            provider=provider,
            project_id=self.project_id,
            priority=BatchPriority.NORMAL,
        )

        logger.info(
            "agent.batch_submitted",
            agent=self.role.value,
            task_id=task.id,
            job_id=job.job_id,
        )

        return AgentOutput(
            task_id=task.id,
            agent_role=self.role.value,
            content=f"[BATCH] Task submitted as job {job.job_id}. Results available within 24h.",
            metadata={"batch_job_id": job.job_id, "is_batch": True},
        )

    # ------------------------------------------------------------------
    # Core LLM call with full tracking
    # ------------------------------------------------------------------
    async def _call_llm_with_tracking(
        self,
        current_task: str,
        task_complexity: TaskComplexity,
        existing_code: str | None = None,
        allow_batch: bool = False,
    ) -> tuple[str, CallCost]:
        """
        Internal wrapper that:
        1. Applies model routing based on complexity
        2. Applies cache via CacheManager.build_cached_messages()
        3. Tracks costs via CostTracker.record_call()
        4. Retries with fallback if error
        """
        settings = get_settings()

        # Emergency mode: force cheap model
        force_model = None
        if self._emergency_mode:
            force_model = self.llm_router.get_emergency_model()

        # Build messages with caching
        if settings.llm.enable_prompt_caching:
            messages = self.cache_manager.build_cached_messages_litellm(
                system_prompt=self._system_prompt,
                project_context=self._project_context,
                existing_code=existing_code,
                current_task=current_task,
            )
        else:
            messages = [
                {"role": "system", "content": self._system_prompt},
            ]
            context_parts: list[str] = []
            if self._project_context:
                context_parts.append(f"<project_context>\n{self._project_context}\n</project_context>")
            if existing_code:
                context_parts.append(f"<existing_code>\n{existing_code}\n</existing_code>")
            context_parts.append(current_task)
            messages.append({"role": "user", "content": "\n\n".join(context_parts)})

        # Call LLM with routing
        expected_model = force_model or self.llm_router.resolve_model(self.role.value, task_complexity)
        async with self._tracer.trace_llm_call(
            model=expected_model,
            agent_role=self.role.value,
        ) as llm_span:
            content, usage_info = await self.llm_router.call_llm(
                agent_role=self.role.value,
                messages=messages,
                task_complexity=task_complexity,
                force_model=force_model,
            )
            llm_span.end(
                output_text=content[:200],
                input_tokens=usage_info.get("input_tokens", 0),
                output_tokens=usage_info.get("output_tokens", 0),
            )

        # Track cost
        model_used = usage_info.get("model", "unknown")
        cost = await self.cost_tracker.record_call(
            agent_role=self.role.value,
            project_id=self.project_id,
            model=model_used,
            input_tokens=usage_info.get("input_tokens", 0),
            output_tokens=usage_info.get("output_tokens", 0),
            cache_creation_tokens=usage_info.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage_info.get("cache_read_input_tokens", 0),
            is_batch=False,
        )

        # Update cache stats
        if settings.llm.enable_prompt_caching:
            self.cache_manager.update_stats(
                cache_creation_tokens=usage_info.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage_info.get("cache_read_input_tokens", 0),
                regular_input_tokens=usage_info.get("input_tokens", 0),
                model=model_used,
            )

        # Check budget alerts
        if settings.llm.enable_cost_tracking:
            alerts = await self.cost_tracker.check_budget_alerts(self.project_id)
            if self.cost_tracker.has_emergency_alert(alerts):
                self._emergency_mode = True
                logger.warning(
                    "agent.emergency_mode_activated",
                    agent=self.role.value,
                    project_id=self.project_id,
                )

        return content, cost

    # ------------------------------------------------------------------
    # Self-reflection
    # ------------------------------------------------------------------
    async def _self_reflect(self, task: Task, output: str) -> float:
        """
        Self-critique the output quality on a 0-1 scale.

        Uses a cheap model (haiku) to evaluate the output against
        the task requirements.
        """
        reflection_prompt = (
            f"Evaluate the quality of this output for the task: {task.description}\n\n"
            f"Output to evaluate:\n{output[:2000]}\n\n"
            "Rate the quality from 0.0 to 1.0 based on:\n"
            "- Completeness (does it address all requirements?)\n"
            "- Correctness (is it technically accurate?)\n"
            "- Clarity (is it well-structured and clear?)\n\n"
            "Respond with ONLY a JSON object: {\"score\": 0.X, \"reason\": \"brief explanation\"}"
        )

        try:
            content, usage = await self.llm_router.call_llm(
                agent_role="fallback",
                messages=[
                    {"role": "system", "content": "You are a quality evaluator. Respond only with JSON."},
                    {"role": "user", "content": reflection_prompt},
                ],
                task_complexity=TaskComplexity.SIMPLE,
                max_tokens=200,
                temperature=0.0,
            )

            # Track reflection cost (cheap model)
            await self.cost_tracker.record_call(
                agent_role=self.role.value,
                project_id=self.project_id,
                model=usage.get("model", "gpt-4o-mini"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

            result = json.loads(content.strip())
            score = float(result.get("score", 0.5))
            logger.info(
                "agent.self_reflect",
                agent=self.role.value,
                score=score,
                reason=result.get("reason", ""),
            )
            return min(max(score, 0.0), 1.0)

        except Exception as exc:
            logger.warning("agent.self_reflect_failed", error=str(exc))
            return 0.5

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_task_prompt(self, task: Task) -> str:
        """Build the task-specific prompt. Implemented by each agent."""
        ...

    def _extract_artifacts(self, content: str) -> dict[str, str]:
        """
        Extract file artifacts from LLM output.

        Looks for code blocks with filenames:
        ```python:filename.py
        ...code...
        ```
        """
        artifacts: dict[str, str] = {}
        lines = content.split("\n")
        current_file: str | None = None
        current_content: list[str] = []
        in_block = False

        for line in lines:
            if line.startswith("```") and ":" in line and not in_block:
                # Start of named code block
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_file = parts[1].strip().rstrip("`")
                    current_content = []
                    in_block = True
            elif line.startswith("```") and in_block:
                # End of code block
                if current_file:
                    artifacts[current_file] = "\n".join(current_content)
                current_file = None
                current_content = []
                in_block = False
            elif in_block:
                current_content.append(line)

        return artifacts

    # ------------------------------------------------------------------
    # Complexity estimation
    # ------------------------------------------------------------------
    async def estimate_complexity(self, task_description: str) -> TaskComplexity:
        """
        Use a cheap model to estimate task complexity.

        This determines which model will be used for the actual task.
        """
        prompt = (
            f"Classify this task's complexity as one of: simple, medium, complex, critical.\n\n"
            f"Task: {task_description}\n\n"
            "Consider:\n"
            "- simple: single file change, minor fix, straightforward template\n"
            "- medium: multi-file change, standard feature, clear requirements\n"
            "- complex: architectural decision, new service, complex logic\n"
            "- critical: security-sensitive, data migration, production deployment\n\n"
            'Respond with ONLY the word: simple, medium, complex, or critical'
        )

        try:
            content, _ = await self.llm_router.call_llm(
                agent_role="fallback",
                messages=[{"role": "user", "content": prompt}],
                task_complexity=TaskComplexity.SIMPLE,
                max_tokens=10,
                temperature=0.0,
            )
            result = content.strip().lower()
            if result in ("simple", "medium", "complex", "critical"):
                return TaskComplexity(result)
        except Exception:
            pass

        return TaskComplexity.MEDIUM
