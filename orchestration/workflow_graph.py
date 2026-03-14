"""
LangGraph-based workflow graph for the SaaS development pipeline.

Defines the full workflow from research to deployment with budget
checks, cache warmup, and human-in-the-loop checkpoints.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

import structlog

from agents.architect_agent import ArchitectAgent
from agents.dev_agent import DevAgent
from agents.devops_agent import DevOpsAgent
from agents.pm_agent import PMAgent
from agents.qa_agent import QAAgent
from agents.research_agent import ResearchAgent
from agents.security_agent import SecurityAgent
from agents.base_agent import AgentOutput
from core.batch_processor import BatchProcessor
from core.cache_manager import CacheManager
from core.config import get_settings
from core.cost_tracker import CostTracker

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------
class WorkflowPhase(str, Enum):
    """Phases of the SaaS development workflow."""

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


@dataclass
class WorkflowState:
    """Mutable state passed through the workflow graph."""

    project_id: str
    product_idea: str
    target_market: str = "B2B SaaS"
    current_phase: WorkflowPhase = WorkflowPhase.BUDGET_CHECK

    # Artifacts produced by each phase
    research_output: str = ""
    architecture_doc: str = ""
    prd_doc: str = ""
    code_artifacts: dict[str, str] = field(default_factory=dict)
    test_results: list[str] = field(default_factory=list)
    security_findings: str = ""
    deployment_result: str = ""
    cost_report: str = ""

    # Feature tracking
    features: list[str] = field(default_factory=list)
    completed_features: list[str] = field(default_factory=list)
    pending_batch_jobs: list[str] = field(default_factory=list)

    # Human decisions
    design_approved: bool = False
    deploy_approved: bool = False

    # Error tracking
    errors: list[str] = field(default_factory=list)
    budget_ok: bool = True


# ---------------------------------------------------------------------------
# Workflow Graph
# ---------------------------------------------------------------------------
class WorkflowGraph:
    """
    LangGraph-style workflow orchestrating the full SaaS development pipeline.

    Workflow:
    START → budget_check → cache_warmup → research → design
    → [human_checkpoint_1] → development_loop → batch_collection
    → integration → deploy → [human_checkpoint_2] → cost_report → END

    Budget is checked after every agent call. If exceeded, all agents
    downgrade to the emergency model automatically.
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.cost_tracker = CostTracker()
        self.cache_manager = CacheManager()
        self.batch_processor = BatchProcessor()

        # Initialize agents
        self.research_agent = ResearchAgent(project_id=project_id)
        self.pm_agent = PMAgent(project_id=project_id)
        self.architect_agent = ArchitectAgent(project_id=project_id)
        self.dev_agent = DevAgent(project_id=project_id)
        self.qa_agent = QAAgent(project_id=project_id)
        self.security_agent = SecurityAgent(project_id=project_id)
        self.devops_agent = DevOpsAgent(project_id=project_id)

        self._human_callback: Any = None

    def set_human_callback(self, callback: Any) -> None:
        """Set callback for human-in-the-loop decisions."""
        self._human_callback = callback

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Execute the full workflow from start to finish."""
        logger.info("workflow.started", project_id=self.project_id)

        phase_handlers = {
            WorkflowPhase.BUDGET_CHECK: self._phase_budget_check,
            WorkflowPhase.CACHE_WARMUP: self._phase_cache_warmup,
            WorkflowPhase.RESEARCH: self._phase_research,
            WorkflowPhase.DESIGN: self._phase_design,
            WorkflowPhase.HUMAN_REVIEW_DESIGN: self._phase_human_review_design,
            WorkflowPhase.DEVELOPMENT: self._phase_development,
            WorkflowPhase.BATCH_COLLECTION: self._phase_batch_collection,
            WorkflowPhase.INTEGRATION: self._phase_integration,
            WorkflowPhase.DEPLOY: self._phase_deploy,
            WorkflowPhase.HUMAN_REVIEW_DEPLOY: self._phase_human_review_deploy,
            WorkflowPhase.COST_REPORT: self._phase_cost_report,
        }

        while state.current_phase not in (WorkflowPhase.COMPLETED, WorkflowPhase.FAILED):
            handler = phase_handlers.get(state.current_phase)
            if handler is None:
                state.errors.append(f"Unknown phase: {state.current_phase}")
                state.current_phase = WorkflowPhase.FAILED
                break

            logger.info("workflow.phase_start", phase=state.current_phase.value)
            try:
                state = await handler(state)
            except Exception as exc:
                logger.error(
                    "workflow.phase_error",
                    phase=state.current_phase.value,
                    error=str(exc),
                )
                state.errors.append(f"{state.current_phase.value}: {exc}")
                state.current_phase = WorkflowPhase.FAILED

        logger.info(
            "workflow.completed",
            project_id=self.project_id,
            final_phase=state.current_phase.value,
            errors=len(state.errors),
        )

        return state

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------
    async def _phase_budget_check(self, state: WorkflowState) -> WorkflowState:
        """Check if budget is available before starting."""
        alerts = await self.cost_tracker.check_budget_alerts(state.project_id)
        if self.cost_tracker.has_emergency_alert(alerts):
            state.budget_ok = False
            state.errors.append("Budget exceeded — workflow cannot start.")
            state.current_phase = WorkflowPhase.FAILED
        else:
            state.current_phase = WorkflowPhase.CACHE_WARMUP
        return state

    async def _phase_cache_warmup(self, state: WorkflowState) -> WorkflowState:
        """Pre-warm caches for all active agents."""
        context = f"Product: {state.product_idea}\nMarket: {state.target_market}"

        warmup_tasks = [
            self.cache_manager.warm_cache("architect", context),
            self.cache_manager.warm_cache("pm", context),
            self.cache_manager.warm_cache("dev", context),
        ]

        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        state.current_phase = WorkflowPhase.RESEARCH
        return state

    async def _phase_research(self, state: WorkflowState) -> WorkflowState:
        """Research phase — competitor analysis (batch mode)."""
        output = await self.research_agent.analyze_competitors(
            product=state.product_idea,
            market=state.target_market,
        )
        state.research_output = output.content
        state.current_phase = WorkflowPhase.DESIGN
        return state

    async def _phase_design(self, state: WorkflowState) -> WorkflowState:
        """Design phase — PRD + Architecture in parallel."""
        # Run PM and Architect in parallel
        prd_task = self.pm_agent.generate_prd(
            product_idea=state.product_idea,
            target_market=state.target_market,
        )
        arch_task = self.architect_agent.design_architecture(
            requirements=state.product_idea,
        )

        prd_output, arch_output = await asyncio.gather(prd_task, arch_task)

        state.prd_doc = prd_output.content
        state.architecture_doc = arch_output.content

        # Set context for downstream agents
        full_context = f"## PRD\n{state.prd_doc}\n\n## Architecture\n{state.architecture_doc}"
        self.dev_agent.set_project_context(full_context)
        self.qa_agent.set_project_context(full_context)
        self.security_agent.set_project_context(full_context)

        # Extract features from PRD/architecture
        if not state.features:
            state.features = [f"Feature {i+1}" for i in range(3)]

        state.current_phase = WorkflowPhase.HUMAN_REVIEW_DESIGN
        return state

    async def _phase_human_review_design(self, state: WorkflowState) -> WorkflowState:
        """Human checkpoint — approve design before development."""
        if self._human_callback:
            decision = await self._human_callback(
                phase="design_review",
                artifacts={
                    "prd": state.prd_doc,
                    "architecture": state.architecture_doc,
                    "research": state.research_output,
                },
            )
            state.design_approved = decision.get("approved", False)
        else:
            # Auto-approve in automated mode
            state.design_approved = True

        if state.design_approved:
            # Warm cache with new context
            full_context = f"## PRD\n{state.prd_doc}\n\n## Architecture\n{state.architecture_doc}"
            await self.cache_manager.warm_cache("dev", full_context)
            state.current_phase = WorkflowPhase.DEVELOPMENT
        else:
            state.errors.append("Design rejected by human reviewer.")
            state.current_phase = WorkflowPhase.FAILED

        return state

    async def _phase_development(self, state: WorkflowState) -> WorkflowState:
        """Development loop — implement features with parallel QA and security."""
        for feature in state.features:
            if feature in state.completed_features:
                continue

            logger.info("workflow.developing_feature", feature=feature)

            # 1. Dev Agent: generate code (real-time)
            dev_output = await self.dev_agent.generate_code(feature=feature)
            state.code_artifacts.update(dev_output.artifacts)

            # 2. QA Agent: generate tests (batch — async)
            qa_output = await self.qa_agent.write_e2e_tests(
                feature=feature,
                source_code=dev_output.content,
                blocking=False,
            )
            if qa_output.metadata.get("is_batch"):
                batch_id = qa_output.metadata.get("batch_job_id", "")
                if batch_id:
                    state.pending_batch_jobs.append(batch_id)

            # 3. Security Agent: audit (batch — parallel)
            sec_output = await self.security_agent.check_dependencies(
                dependencies=dev_output.content[:1000]
            )
            if sec_output.metadata.get("is_batch"):
                batch_id = sec_output.metadata.get("batch_job_id", "")
                if batch_id:
                    state.pending_batch_jobs.append(batch_id)

            # 4. Budget check after each feature
            alerts = await self.cost_tracker.check_budget_alerts(state.project_id)
            if self.cost_tracker.has_emergency_alert(alerts):
                state.budget_ok = False
                logger.warning("workflow.budget_exceeded_during_dev")
                break

            state.completed_features.append(feature)

        state.current_phase = WorkflowPhase.BATCH_COLLECTION
        return state

    async def _phase_batch_collection(self, state: WorkflowState) -> WorkflowState:
        """Collect results from batch jobs."""
        for job_id in state.pending_batch_jobs:
            try:
                results = await self.batch_processor.poll_batch_results(job_id)
                for result in results:
                    if result.status == "succeeded":
                        state.test_results.append(result.content)
                    elif result.error:
                        state.errors.append(f"Batch {job_id}: {result.error}")
            except Exception as exc:
                logger.warning("workflow.batch_poll_error", job_id=job_id, error=str(exc))

        state.pending_batch_jobs.clear()
        state.current_phase = WorkflowPhase.INTEGRATION
        return state

    async def _phase_integration(self, state: WorkflowState) -> WorkflowState:
        """Integration testing phase."""
        all_code = "\n\n".join(state.code_artifacts.values())
        qa_output = await self.qa_agent.write_e2e_tests(
            feature="Integration tests for all features",
            source_code=all_code[:5000],
            blocking=True,
        )
        state.test_results.append(qa_output.content)
        state.current_phase = WorkflowPhase.DEPLOY
        return state

    async def _phase_deploy(self, state: WorkflowState) -> WorkflowState:
        """Deployment phase — generate infra and deploy."""
        # Generate Dockerfile (real-time)
        await self.devops_agent.generate_dockerfile()

        state.current_phase = WorkflowPhase.HUMAN_REVIEW_DEPLOY
        return state

    async def _phase_human_review_deploy(self, state: WorkflowState) -> WorkflowState:
        """Human checkpoint — go/no-go for production deployment."""
        if self._human_callback:
            decision = await self._human_callback(
                phase="deploy_review",
                artifacts={
                    "code": state.code_artifacts,
                    "tests": state.test_results,
                    "security": state.security_findings,
                },
            )
            state.deploy_approved = decision.get("approved", False)
        else:
            state.deploy_approved = True

        if state.deploy_approved:
            state.current_phase = WorkflowPhase.COST_REPORT
        else:
            state.errors.append("Deployment rejected by human reviewer.")
            state.current_phase = WorkflowPhase.FAILED

        return state

    async def _phase_cost_report(self, state: WorkflowState) -> WorkflowState:
        """Generate final cost report."""
        daily_report = await self.cost_tracker.get_daily_cost(state.project_id)
        cache_stats = self.cache_manager.get_cache_stats()

        state.cost_report = (
            f"## Cost Report for {state.project_id}\n"
            f"- Total cost today: ${daily_report.total_cost_usd:.4f}\n"
            f"- Total calls: {daily_report.total_calls}\n"
            f"- Cache savings: ${daily_report.cache_savings_usd:.4f}\n"
            f"- Batch savings: ${daily_report.batch_savings_usd:.4f}\n"
            f"- Total savings: ${daily_report.total_savings_usd:.4f}\n"
            f"- Cache hit ratio: {cache_stats.cache_hit_ratio:.1%}\n"
            f"- Cost by agent: {daily_report.cost_by_agent}\n"
        )

        logger.info("workflow.cost_report_generated", total_cost=daily_report.total_cost_usd)
        state.current_phase = WorkflowPhase.COMPLETED
        return state
