"""
LangGraph-based workflow graph for the SaaS development pipeline.

Defines the full workflow from research to deployment with budget
checks, cache warmup, and human-in-the-loop checkpoints.
"""

from __future__ import annotations

import asyncio
import json
import re
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
from core.config import get_settings, TaskComplexity
from core.cost_tracker import CostTracker
from core.tools.github_tool import GitHubTool
from core.tools.code_sandbox import CodeSandbox
from core.tools.deploy_tool import DeployTool

logger = structlog.get_logger(__name__)

# Maximum number of fix-and-retest iterations per feature / integration phase
MAX_FIX_RETRIES = 3


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

    # GitHub tracking
    github_repo_name: str = ""
    github_repo_url: str = ""
    github_branch: str = "main"

    # Railway tracking
    railway_project_id: str = ""
    railway_deployment_url: str = ""

    # Sandbox test execution results
    sandbox_test_results: list[dict] = field(default_factory=list)


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

        # Tools for autonomous workflow
        self.github_tool = GitHubTool()
        self.code_sandbox = CodeSandbox()
        self.deploy_tool = DeployTool()

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
            state.features = await self._extract_features_from_prd(state.prd_doc)
            if not state.features:
                state.features = ["Core API endpoints", "Authentication system", "Data models"]
                logger.warning("workflow.feature_extraction_fallback")

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
            # Create GitHub repository (or skip if an existing repo was provided)
            if not state.github_repo_name:
                await self._setup_github_repo(state)
            else:
                logger.info("workflow.using_existing_repo", repo=state.github_repo_name)
                # Commit initial docs to existing repo
                await self._commit_artifacts_to_github(
                    state=state,
                    files={
                        "docs/PRD.md": state.prd_doc,
                        "docs/ARCHITECTURE.md": state.architecture_doc,
                    },
                    message="docs: add PRD and architecture documents",
                )

            # Warm cache with new context
            full_context = f"## PRD\n{state.prd_doc}\n\n## Architecture\n{state.architecture_doc}"
            await self.cache_manager.warm_cache("dev", full_context)
            state.current_phase = WorkflowPhase.DEVELOPMENT
        else:
            state.errors.append("Design rejected by human reviewer.")
            state.current_phase = WorkflowPhase.FAILED

        return state

    async def _phase_development(self, state: WorkflowState) -> WorkflowState:
        """Development loop — implement features with GitHub commits, sandbox testing, and iterative correction."""
        for feature in state.features:
            if feature in state.completed_features:
                continue

            logger.info("workflow.developing_feature", feature=feature)

            # 1. Dev Agent: generate code (real-time)
            dev_output = await self.dev_agent.generate_code(feature=feature)
            state.code_artifacts.update(dev_output.artifacts)

            # 2. Commit code artifacts to GitHub
            if state.github_repo_name and dev_output.artifacts:
                await self._commit_artifacts_to_github(
                    state=state,
                    files=dev_output.artifacts,
                    message=f"feat: implement {feature}",
                )

            # 3. QA Agent: generate tests (blocking — need code immediately for E2B)
            qa_output = await self.qa_agent.write_e2e_tests(
                feature=feature,
                source_code=dev_output.content,
                blocking=True,
            )

            # 4. Run generated tests in E2B sandbox + correction loop
            if qa_output.content and not qa_output.metadata.get("is_batch"):
                sandbox_result = await self._run_tests_in_sandbox(
                    test_code=qa_output.content,
                    source_code=state.code_artifacts,
                    feature_name=feature,
                )

                # ---- Iterative correction loop ----
                current_code_output = dev_output
                for attempt in range(1, MAX_FIX_RETRIES + 1):
                    if sandbox_result["passed"]:
                        break

                    logger.warning(
                        "workflow.tests_failed_retrying",
                        feature=feature,
                        attempt=attempt,
                        max_retries=MAX_FIX_RETRIES,
                        stderr=sandbox_result["stderr"][:300],
                    )

                    # Ask Dev Agent to fix the code based on test failures
                    bug_description = (
                        f"Tests failed for feature '{feature}'.\n"
                        f"Test stderr:\n{sandbox_result['stderr'][:2000]}\n\n"
                        f"Test stdout:\n{sandbox_result['stdout'][:2000]}"
                    )
                    fix_output = await self.dev_agent.fix_bug(
                        bug_description=bug_description,
                        code=current_code_output.content,
                    )

                    # Update artifacts with fixed code
                    if fix_output.artifacts:
                        state.code_artifacts.update(fix_output.artifacts)
                        current_code_output = fix_output

                        # Commit the fix to GitHub
                        if state.github_repo_name:
                            await self._commit_artifacts_to_github(
                                state=state,
                                files=fix_output.artifacts,
                                message=f"fix: {feature} (attempt {attempt}/{MAX_FIX_RETRIES})",
                            )

                    # Re-run the same tests against fixed code
                    sandbox_result = await self._run_tests_in_sandbox(
                        test_code=qa_output.content,
                        source_code=state.code_artifacts,
                        feature_name=f"{feature} (fix attempt {attempt})",
                    )

                # Record final test result (passed or last failure)
                state.sandbox_test_results.append(sandbox_result)
                passed = sandbox_result["passed"]
                state.test_results.append(
                    f"[{feature}] {'PASSED' if passed else f'FAILED after {MAX_FIX_RETRIES} fix attempts'}: "
                    f"{sandbox_result['stdout'][:500]}"
                )
                if not passed:
                    logger.error(
                        "workflow.feature_tests_exhausted",
                        feature=feature,
                        retries=MAX_FIX_RETRIES,
                    )

            # 5. Security Agent: dependency check
            sec_output = await self.security_agent.check_dependencies(
                dependencies=dev_output.content
            )
            if sec_output.metadata.get("is_batch"):
                batch_id = sec_output.metadata.get("batch_job_id", "")
                if batch_id:
                    state.pending_batch_jobs.append(batch_id)

            # 6. Budget check after each feature
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
        """Integration testing + full security audit, with iterative correction."""
        all_code = "\n\n".join(state.code_artifacts.values())

        # 1. QA Agent: generate integration tests (blocking)
        qa_output = await self.qa_agent.write_e2e_tests(
            feature="Integration tests for all features",
            source_code=all_code[:5000],
            blocking=True,
        )
        state.test_results.append(qa_output.content)

        # 2. Run integration tests in E2B sandbox + correction loop
        if qa_output.content:
            integration_result = await self._run_tests_in_sandbox(
                test_code=qa_output.content,
                source_code=state.code_artifacts,
                feature_name="integration",
            )

            # ---- Iterative correction loop for integration tests ----
            for attempt in range(1, MAX_FIX_RETRIES + 1):
                if integration_result["passed"]:
                    break

                logger.warning(
                    "workflow.integration_tests_failed_retrying",
                    attempt=attempt,
                    max_retries=MAX_FIX_RETRIES,
                    stderr=integration_result["stderr"][:300],
                )

                # Ask Dev Agent to fix based on integration test failures
                bug_description = (
                    f"Integration tests failed.\n"
                    f"Test stderr:\n{integration_result['stderr'][:2000]}\n\n"
                    f"Test stdout:\n{integration_result['stdout'][:2000]}"
                )
                fix_output = await self.dev_agent.fix_bug(
                    bug_description=bug_description,
                    code=all_code[:5000],
                )

                # Update artifacts with fixed code
                if fix_output.artifacts:
                    state.code_artifacts.update(fix_output.artifacts)
                    all_code = "\n\n".join(state.code_artifacts.values())

                    # Commit the fix to GitHub
                    if state.github_repo_name:
                        await self._commit_artifacts_to_github(
                            state=state,
                            files=fix_output.artifacts,
                            message=f"fix: integration issues (attempt {attempt}/{MAX_FIX_RETRIES})",
                        )

                # Re-run integration tests against fixed code
                integration_result = await self._run_tests_in_sandbox(
                    test_code=qa_output.content,
                    source_code=state.code_artifacts,
                    feature_name=f"integration (fix attempt {attempt})",
                )

            state.sandbox_test_results.append(integration_result)
            if not integration_result["passed"]:
                logger.error(
                    "workflow.integration_tests_exhausted",
                    retries=MAX_FIX_RETRIES,
                )

        # 3. Full security audit on ALL code
        try:
            sec_output = await self.security_agent.audit_code(
                code=all_code,
                focus_areas="OWASP Top 10",
            )
            state.security_findings = sec_output.content
            logger.info(
                "workflow.security_audit_complete",
                findings_length=len(state.security_findings),
            )
        except Exception as exc:
            state.security_findings = f"Security audit failed: {exc}"
            state.errors.append(f"Security audit error: {exc}")
            logger.error("workflow.security_audit_error", error=str(exc))

        # 4. Commit test files to GitHub
        if state.github_repo_name and qa_output.artifacts:
            await self._commit_artifacts_to_github(
                state=state,
                files=qa_output.artifacts,
                message="test: add integration tests",
            )

        state.current_phase = WorkflowPhase.DEPLOY
        return state

    async def _phase_deploy(self, state: WorkflowState) -> WorkflowState:
        """Deployment phase — generate infra, commit to GitHub, deploy to Railway."""
        # 1. Generate Dockerfile (real-time)
        devops_output = await self.devops_agent.generate_dockerfile()

        # 2. Store Dockerfile artifacts
        if devops_output.artifacts:
            state.code_artifacts.update(devops_output.artifacts)

        # 3. Generate CI/CD pipeline
        cicd_output = await self.devops_agent.generate_cicd_pipeline(
            platform="github-actions",
            tech_stack="Python/FastAPI",
        )
        if cicd_output.artifacts:
            state.code_artifacts.update(cicd_output.artifacts)

        # 4. Commit deployment files to GitHub
        deploy_files = {**devops_output.artifacts, **cicd_output.artifacts}
        if state.github_repo_name and deploy_files:
            await self._commit_artifacts_to_github(
                state=state,
                files=deploy_files,
                message="chore: add Dockerfile, CI/CD, and deployment config",
            )

        # 5. Deploy from GitHub to Railway
        state.deployment_result = await self._deploy_to_railway(state)

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
                    "deployment": state.deployment_result,
                    "sandbox_results": state.sandbox_test_results,
                    "deployment_url": state.railway_deployment_url,
                    "github_repo": state.github_repo_url,
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

    # ------------------------------------------------------------------
    # Tool helper methods
    # ------------------------------------------------------------------
    async def _extract_features_from_prd(self, prd_text: str) -> list[str]:
        """Extract feature names from the PRD using a cheap LLM call."""
        try:
            prompt = (
                "Extract the list of feature names from this PRD. "
                "Return ONLY a JSON array of short feature name strings, nothing else.\n\n"
                f"PRD:\n{prd_text[:4000]}"
            )
            content, _ = await self.pm_agent.llm_router.call_llm(
                agent_role="fallback",
                messages=[{"role": "user", "content": prompt}],
                task_complexity=TaskComplexity.SIMPLE,
                max_tokens=500,
                temperature=0.0,
            )
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```\w*\n?", "", cleaned).strip()
            features = json.loads(cleaned)
            if isinstance(features, list) and all(isinstance(f, str) for f in features):
                logger.info("workflow.features_extracted", count=len(features))
                return features[:10]
        except Exception as exc:
            logger.warning("workflow.feature_extraction_failed", error=str(exc))
        return []

    async def _setup_github_repo(self, state: WorkflowState) -> None:
        """Create a GitHub repository for the project. Degrades gracefully on failure."""
        repo_name = f"saas-{state.project_id}"
        try:
            result = await self.github_tool.create_repository(
                name=repo_name,
                description=f"Auto-generated SaaS: {state.product_idea[:100]}",
                private=True,
                auto_init=True,
            )
            if result.success:
                state.github_repo_name = result.data.get("full_name", repo_name)
                state.github_repo_url = result.url or ""
                logger.info("workflow.github_repo_created", repo=state.github_repo_name)

                # Commit PRD and architecture docs as initial content
                await self.github_tool.commit_files(
                    repo_name=state.github_repo_name,
                    branch="main",
                    files={
                        "docs/PRD.md": state.prd_doc,
                        "docs/ARCHITECTURE.md": state.architecture_doc,
                    },
                    commit_message="docs: add PRD and architecture documents",
                )
            else:
                state.errors.append(f"GitHub repo creation failed: {result.message}")
                logger.warning("workflow.github_repo_failed", error=result.message)
        except Exception as exc:
            state.errors.append(f"GitHub setup error: {exc}")
            logger.error("workflow.github_setup_error", error=str(exc))

    async def _commit_artifacts_to_github(
        self,
        state: WorkflowState,
        files: dict[str, str],
        message: str,
    ) -> None:
        """Commit code artifacts to GitHub. No-op if GitHub is not configured."""
        if not state.github_repo_name:
            return
        try:
            result = await self.github_tool.commit_files(
                repo_name=state.github_repo_name,
                branch=state.github_branch,
                files=files,
                commit_message=message,
            )
            if result.success:
                logger.info("workflow.code_committed", files=len(files))
            else:
                state.errors.append(f"GitHub commit failed: {result.message}")
                logger.warning("workflow.commit_failed", error=result.message)
        except Exception as exc:
            state.errors.append(f"GitHub commit error: {exc}")
            logger.error("workflow.commit_error", error=str(exc))

    async def _run_tests_in_sandbox(
        self,
        test_code: str,
        source_code: dict[str, str],
        feature_name: str,
    ) -> dict:
        """Run tests in E2B sandbox. Returns structured result dict."""
        try:
            extracted_test = self._extract_test_code(test_code)
            result = await self.code_sandbox.run_tests(
                test_code=extracted_test,
                source_code=source_code,
                packages=["httpx", "pytest-asyncio"],
            )
            logger.info(
                "workflow.sandbox_tests_completed",
                feature=feature_name,
                passed=result.success,
                exit_code=result.exit_code,
            )
            return {
                "feature": feature_name,
                "passed": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }
        except Exception as exc:
            logger.error("workflow.sandbox_tests_error", feature=feature_name, error=str(exc))
            return {
                "feature": feature_name,
                "passed": False,
                "stdout": "",
                "stderr": f"Sandbox execution error: {exc}",
                "exit_code": -1,
            }

    @staticmethod
    def _extract_test_code(content: str) -> str:
        """Extract pytest code from LLM output that may contain markdown code blocks."""
        pattern = r"```python(?::[\w/._-]+)?\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            test_blocks = [m for m in matches if "test" in m.lower() or "pytest" in m.lower()]
            if test_blocks:
                return "\n\n".join(test_blocks)
            return "\n\n".join(matches)
        return content

    async def _deploy_to_railway(self, state: WorkflowState) -> str:
        """Create Railway project and deploy from GitHub. Returns status string."""
        if not state.github_repo_url:
            msg = "Skipping Railway deployment: no GitHub repository available"
            logger.warning("workflow.deploy_skipped_no_repo")
            return msg

        try:
            # Step 1: Create Railway project
            project_name = f"saas-{state.project_id}"
            project_result = await self.deploy_tool.create_project(name=project_name)
            if not project_result.success:
                msg = f"Railway project creation failed: {project_result.message}"
                state.errors.append(msg)
                return msg

            state.railway_project_id = project_result.deployment_id or ""
            logger.info("workflow.railway_project_created", id=state.railway_project_id)

            # Step 2: Deploy from GitHub
            deploy_result = await self.deploy_tool.deploy_from_github(
                repo_url=state.github_repo_url,
                branch=state.github_branch,
            )
            if not deploy_result.success:
                msg = f"Railway deployment failed: {deploy_result.message}"
                state.errors.append(msg)
                return msg

            service_id = deploy_result.deployment_id or ""

            # Step 3: Set environment variables
            settings = get_settings()
            env_vars = {
                "ENVIRONMENT": "production",
                "SECRET_KEY": settings.api.secret_key,
            }
            await self.deploy_tool.set_environment_variables(
                service_id=service_id,
                env_vars=env_vars,
            )

            # Step 4: Check deployment status
            if service_id:
                await asyncio.sleep(5)
                status = await self.deploy_tool.get_deployment_status(service_id)
                state.railway_deployment_url = status.deployment_url or ""
                return (
                    f"Deployed to Railway. "
                    f"Project: {project_name}, "
                    f"Service: {service_id}, "
                    f"Status: {status.message}, "
                    f"URL: {state.railway_deployment_url or 'pending'}"
                )

            return f"Deployment triggered for {project_name}. Service ID: {service_id}"

        except Exception as exc:
            msg = f"Railway deployment error: {exc}"
            state.errors.append(msg)
            logger.error("workflow.railway_deploy_error", error=str(exc))
            return msg
