"""
QA Agent — test planning, test generation, and quality validation.

Uses claude-haiku-4-5 by default (10x cheaper than Sonnet).
Upgrades to Sonnet if coverage < 80% for gap analysis.
Batch eligible: generate_test_plan, write_e2e_tests.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole

logger = structlog.get_logger(__name__)


class QAAgent(BaseAgent):
    """
    Senior QA Engineer agent for test automation and quality assurance.

    Capabilities:
    - generate_test_plan: Create comprehensive test plans (batch eligible)
    - write_e2e_tests: Write end-to-end tests (batch eligible)
    - run_tests: Execute tests and report results (real-time)
    - generate_bug_report: Create detailed bug reports
    - validate_coverage: Check code coverage, upgrade model if < 80%
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.QA, project_id=project_id, **kwargs)

    def _build_task_prompt(self, task: Task) -> str:
        """Build QA-specific prompt."""
        action = task.context.get("action", "")
        prompts = {
            "generate_test_plan": self._test_plan_prompt,
            "write_e2e_tests": self._e2e_tests_prompt,
            "generate_bug_report": self._bug_report_prompt,
            "validate_coverage": self._coverage_prompt,
        }
        builder = prompts.get(action, self._default_prompt)
        return builder(task)

    def _test_plan_prompt(self, task: Task) -> str:
        prd = task.context.get("prd", "")
        architecture = task.context.get("architecture", "")
        return f"""Create a comprehensive test plan for the following application.

## PRD
{prd[:3000] if prd else task.description}

## Architecture
{architecture[:2000] if architecture else "Not provided."}

## Test Plan Structure
1. **Test Strategy** — unit, integration, e2e, performance, security
2. **Test Environments** — local, CI, staging, production
3. **Test Cases** — organized by feature/module
   - Test ID, Description, Steps, Expected Result, Priority
4. **Coverage Targets** — minimum 80% code coverage
5. **Automation Strategy** — which tests to automate first
6. **Risk-Based Testing** — focus areas based on risk assessment

Output as structured markdown with tables."""

    def _e2e_tests_prompt(self, task: Task) -> str:
        feature = task.context.get("feature", task.description)
        source_code = task.context.get("source_code", "")
        return f"""Write end-to-end tests for the following feature.

## Feature
{feature}

## Source Code
{source_code[:4000] if source_code else "Not provided — infer from feature description."}

## Requirements
1. Use pytest with httpx AsyncClient for API tests
2. Test complete user flows (happy path + error paths)
3. Include setup/teardown with fixtures
4. Test authentication flows if applicable
5. Validate response schemas
6. Test pagination, filtering, sorting if applicable

## Output
```python:tests/test_e2e_{task.id}.py
# Complete test file
```"""

    def _bug_report_prompt(self, task: Task) -> str:
        error = task.context.get("error", "")
        code = task.context.get("code", "")
        return f"""Generate a detailed bug report for the following issue.

## Error
{error or task.description}

## Related Code
{code[:3000] if code else "Not provided."}

## Bug Report Format
- **Title**: Concise description
- **Severity**: Critical / Major / Minor / Cosmetic
- **Steps to Reproduce**: Numbered steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Root Cause Analysis**: Technical explanation
- **Suggested Fix**: Code-level recommendation
- **Regression Test**: Test to prevent recurrence"""

    def _coverage_prompt(self, task: Task) -> str:
        coverage_report = task.context.get("coverage_report", "")
        code = task.context.get("code", "")
        return f"""Analyze code coverage and identify gaps.

## Coverage Report
{coverage_report or "No coverage report provided."}

## Source Code
{code[:4000] if code else "Not provided."}

## Analysis Required
1. Identify uncovered code paths
2. Prioritize coverage gaps by risk
3. Generate specific test cases to fill gaps
4. Estimate effort to reach 80%+ coverage

Output test code to fill the gaps:
```python:tests/test_coverage_gaps.py
# Tests for uncovered code paths
```"""

    def _default_prompt(self, task: Task) -> str:
        return f"QA task: {task.description}\n\nProvide your testing analysis and artifacts."

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------
    async def generate_test_plan(self, prd: str = "", architecture: str = "") -> AgentOutput:
        """Create test plan (batch eligible)."""
        task = Task(
            id=f"testplan_{self.project_id}",
            type=TaskType.TEST_GENERATION,
            description="Generate comprehensive test plan",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"action": "generate_test_plan", "prd": prd, "architecture": architecture},
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)

    async def write_e2e_tests(
        self, feature: str, source_code: str = "", blocking: bool = False
    ) -> AgentOutput:
        """Write E2E tests (batch eligible when non-blocking)."""
        task = Task(
            id=f"e2e_{self.project_id}",
            type=TaskType.TEST_GENERATION,
            description=f"Write E2E tests for: {feature[:60]}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={
                "action": "write_e2e_tests",
                "feature": feature,
                "source_code": source_code,
            },
            is_blocking=blocking,
            allow_batch=not blocking,
        )
        return await self.execute(task)

    async def generate_bug_report(self, error: str, code: str = "") -> AgentOutput:
        """Generate a detailed bug report."""
        task = Task(
            id=f"bug_{self.project_id}",
            type=TaskType.TEST_GENERATION,
            description=f"Bug report: {error[:60]}",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={"action": "generate_bug_report", "error": error, "code": code},
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def validate_coverage(
        self, coverage_report: str, code: str = ""
    ) -> AgentOutput:
        """Validate coverage, upgrade model if < 80%."""
        # Parse coverage percentage
        complexity = TaskComplexity.SIMPLE
        try:
            for line in coverage_report.split("\n"):
                if "TOTAL" in line or "total" in line.lower():
                    parts = line.split()
                    for part in reversed(parts):
                        part_clean = part.strip("%")
                        if part_clean.replace(".", "").isdigit():
                            coverage_pct = float(part_clean)
                            if coverage_pct < 80:
                                complexity = TaskComplexity.COMPLEX
                                logger.info(
                                    "qa_agent.low_coverage_upgrade",
                                    coverage=coverage_pct,
                                )
                            break
        except (ValueError, IndexError):
            pass

        task = Task(
            id=f"coverage_{self.project_id}",
            type=TaskType.TEST_GENERATION,
            description="Validate and improve code coverage",
            complexity=complexity,
            project_id=self.project_id,
            context={
                "action": "validate_coverage",
                "coverage_report": coverage_report,
                "code": code,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)
