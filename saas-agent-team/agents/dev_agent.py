"""
Dev Agent — the primary code generation agent.

Uses claude-sonnet-4-5 by default, with auto-complexity detection
to downgrade to Haiku for simple tasks or upgrade to Opus for critical ones.
Maximum cost optimizations: caching project context, batch for tests/docs.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole

logger = structlog.get_logger(__name__)


class DevAgent(BaseAgent):
    """
    Senior Full-Stack Engineer agent (Python/FastAPI + TypeScript/Next.js).

    The most LLM-intensive agent — all cost optimizations are critical:
    1. Cache project context (architecture + PRD + existing code)
    2. Auto-detect complexity: simple→Haiku, complex→Sonnet, critical→Opus
    3. Test generation → batch eligible
    4. Documentation → batch eligible

    Follows TDD: tests first, implementation second.
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.DEV, project_id=project_id, **kwargs)

    def _build_task_prompt(self, task: Task) -> str:
        """Build dev-specific prompt based on task type."""
        if task.type == TaskType.CODE_GENERATION:
            return self._code_gen_prompt(task)
        elif task.type == TaskType.TEST_GENERATION:
            return self._test_gen_prompt(task)
        elif task.type == TaskType.CODE_REVIEW:
            return self._code_review_prompt(task)
        elif task.type == TaskType.DOCUMENTATION:
            return self._doc_gen_prompt(task)
        else:
            return f"Development task: {task.description}\n\nWrite production-ready code."

    def _code_gen_prompt(self, task: Task) -> str:
        """Build code generation prompt."""
        feature = task.context.get("feature", task.description)
        tech_stack = task.context.get("tech_stack", "Python/FastAPI")
        user_stories = task.context.get("user_stories", "")
        file_structure = task.context.get("file_structure", "")

        stories_section = f"\n## User Stories\n{user_stories}" if user_stories else ""
        structure_section = f"\n## File Structure\n{file_structure}" if file_structure else ""

        return f"""Implement the following feature end-to-end.

## Feature
{feature}

## Tech Stack
{tech_stack}
{stories_section}
{structure_section}

## Requirements
1. Write production-ready code — no placeholders, no TODOs
2. Full type hints (mypy-compatible)
3. Async/await for all I/O operations
4. Comprehensive error handling with structured logging
5. Docstrings on all public classes and methods
6. Follow TDD: write tests FIRST, then implementation

## Output Format
For each file, use this format:

```python:path/to/file.py
# complete file content here
```

Include ALL files needed:
- Source code files
- Test files (test_*.py)
- Any configuration changes
- Database migrations if needed

Start with the test file, then the implementation."""

    def _test_gen_prompt(self, task: Task) -> str:
        """Build test generation prompt."""
        source_code = task.context.get("source_code", "")
        test_type = task.context.get("test_type", "unit")

        return f"""Generate comprehensive {test_type} tests for the provided code.

## Code to Test
{source_code[:5000] if source_code else task.description}

## Requirements
1. Use pytest with pytest-asyncio for async tests
2. Cover happy path, edge cases, and error cases
3. Use fixtures and parametrize where appropriate
4. Mock external dependencies (DB, APIs, etc.)
5. Target >90% code coverage
6. Include type hints on all test functions

## Output Format
```python:test_*.py
import pytest
# ... complete test file
```

Write thorough tests that actually validate behavior."""

    def _code_review_prompt(self, task: Task) -> str:
        """Build code review prompt."""
        review_focus = task.context.get("review_focus", "quality")
        return f"""Review the provided code for {review_focus}.

## Review Criteria
1. Code quality and readability
2. Type safety and correctness
3. Error handling completeness
4. Performance considerations
5. Security vulnerabilities
6. Test coverage gaps

## Output Format
```json
{{
    "overall_score": 0.0,
    "issues": [
        {{
            "severity": "critical|major|minor",
            "file": "...",
            "line": 0,
            "description": "...",
            "fix": "..."
        }}
    ],
    "suggestions": ["..."]
}}
```"""

    def _doc_gen_prompt(self, task: Task) -> str:
        """Build documentation generation prompt."""
        doc_type = task.context.get("doc_type", "API")
        return f"""Generate {doc_type} documentation for the provided code.

## Documentation Requirements
1. Clear, concise descriptions
2. Usage examples with code snippets
3. API endpoint documentation (if applicable)
4. Configuration options
5. Error handling and troubleshooting

## Output Format
Structured markdown documentation."""

    # ------------------------------------------------------------------
    # High-level capability methods
    # ------------------------------------------------------------------
    async def generate_code(
        self,
        feature: str,
        tech_stack: str = "Python/FastAPI",
        user_stories: str = "",
        existing_code: str = "",
    ) -> AgentOutput:
        """Generate production-ready code for a feature."""
        # Auto-detect complexity
        complexity = await self.estimate_complexity(feature)
        logger.info("dev_agent.complexity_detected", feature=feature[:50], complexity=complexity.value)

        task = Task(
            id=f"code_{self.project_id}",
            type=TaskType.CODE_GENERATION,
            description=f"Implement: {feature[:80]}",
            complexity=complexity,
            project_id=self.project_id,
            context={
                "feature": feature,
                "tech_stack": tech_stack,
                "user_stories": user_stories,
                "existing_code": existing_code,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def generate_tests(
        self,
        source_code: str,
        test_type: str = "unit",
        blocking: bool = False,
    ) -> AgentOutput:
        """Generate tests (batch eligible when non-blocking)."""
        task = Task(
            id=f"tests_{self.project_id}",
            type=TaskType.TEST_GENERATION,
            description=f"Generate {test_type} tests",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={
                "source_code": source_code,
                "test_type": test_type,
                "action": "generate_tests",
            },
            is_blocking=blocking,
            allow_batch=not blocking,
        )
        return await self.execute(task)

    async def fix_bug(self, bug_description: str, code: str) -> AgentOutput:
        """Fix a bug in existing code (always real-time)."""
        task = Task(
            id=f"fix_{self.project_id}",
            type=TaskType.CODE_GENERATION,
            description=f"Fix bug: {bug_description[:80]}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={
                "feature": f"Bug fix: {bug_description}",
                "existing_code": code,
                "action": "fix_bug",
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def generate_documentation(self, code: str, doc_type: str = "API") -> AgentOutput:
        """Generate documentation (batch eligible)."""
        task = Task(
            id=f"docs_{self.project_id}",
            type=TaskType.DOCUMENTATION,
            description=f"Generate {doc_type} documentation",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={
                "existing_code": code,
                "doc_type": doc_type,
                "action": "generate_documentation",
            },
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)
