"""
Architect Agent — designs system architecture for SaaS applications.

Uses claude-opus-4 by default for maximum reasoning capability.
Produces structured architecture documents in JSON format.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole

logger = structlog.get_logger(__name__)


class ArchitectAgent(BaseAgent):
    """
    Senior Software Architect agent specializing in SaaS architecture.

    Capabilities:
    - design_architecture: Produce comprehensive architecture documents
    - review_code: Review code for architectural compliance
    - estimate_complexity: Classify task complexity (uses cheap model)

    Optimizations:
    - Code reviews use prompt caching on the code being reviewed
    - estimate_complexity auto-downgrades to Haiku
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.ARCHITECT, project_id=project_id, **kwargs)

    def _build_task_prompt(self, task: Task) -> str:
        """Build architecture-specific prompt based on task type."""
        if task.type == TaskType.ARCHITECTURE:
            return self._build_architecture_prompt(task)
        elif task.type == TaskType.CODE_REVIEW:
            return self._build_review_prompt(task)
        else:
            return f"Task: {task.description}\n\nProvide your analysis and recommendations."

    def _build_architecture_prompt(self, task: Task) -> str:
        """Build prompt for architecture design tasks."""
        requirements = task.context.get("requirements", "")
        tech_constraints = task.context.get("tech_constraints", "")
        target_users = task.context.get("target_users", "")

        return f"""Design a complete system architecture for the following SaaS application.

## Requirements
{requirements or task.description}

## Technical Constraints
{tech_constraints or "No specific constraints. Choose the best stack for a modern SaaS."}

## Target Users
{target_users or "B2B SaaS users"}

## Expected Output (JSON)
Produce a structured JSON document with the following sections:

```json
{{
    "project_name": "...",
    "architecture_pattern": "monolith_modular | microservices | serverless",
    "justification": "Why this pattern was chosen",
    "tech_stack": {{
        "backend": {{"language": "...", "framework": "...", "justification": "..."}},
        "frontend": {{"framework": "...", "justification": "..."}},
        "database": {{"primary": "...", "cache": "...", "justification": "..."}},
        "infrastructure": {{"hosting": "...", "ci_cd": "...", "monitoring": "..."}}
    }},
    "modules": [
        {{
            "name": "...",
            "responsibility": "...",
            "api_endpoints": ["..."],
            "database_tables": ["..."],
            "dependencies": ["..."]
        }}
    ],
    "data_model": {{
        "entities": [
            {{"name": "...", "fields": [{{"name": "...", "type": "...", "constraints": "..."}}]}}
        ],
        "relationships": ["..."]
    }},
    "security": {{
        "authentication": "...",
        "authorization": "...",
        "data_encryption": "...",
        "compliance": ["..."]
    }},
    "scalability": {{
        "strategy": "...",
        "bottlenecks": ["..."],
        "solutions": ["..."]
    }},
    "estimated_complexity": "simple | medium | complex | critical",
    "estimated_dev_days": 0,
    "risks": [{{"risk": "...", "mitigation": "...", "severity": "low|medium|high"}}]
}}
```

Be exhaustive. Justify every technical decision."""

    def _build_review_prompt(self, task: Task) -> str:
        """Build prompt for code review tasks."""
        review_focus = task.context.get("review_focus", "architecture compliance")
        return f"""Review the provided code for {review_focus}.

Task: {task.description}

Evaluate:
1. Architecture compliance — Does it follow the established patterns?
2. Separation of concerns — Are responsibilities well-distributed?
3. Scalability — Will this work under 10x, 100x load?
4. Maintainability — Is the code clean and well-structured?
5. Security — Are there any vulnerability risks?

Provide your review as structured JSON:
```json
{{
    "overall_score": 0.0,
    "issues": [
        {{
            "severity": "critical|major|minor|suggestion",
            "location": "file:line or module",
            "description": "...",
            "suggestion": "..."
        }}
    ],
    "strengths": ["..."],
    "recommendations": ["..."]
}}
```"""

    async def design_architecture(
        self,
        requirements: str,
        tech_constraints: str = "",
        target_users: str = "",
    ) -> AgentOutput:
        """Design a complete system architecture."""
        task = Task(
            id=f"arch_{self.project_id}",
            type=TaskType.ARCHITECTURE,
            description=f"Design architecture for: {requirements[:100]}",
            complexity=TaskComplexity.COMPLEX,
            project_id=self.project_id,
            context={
                "requirements": requirements,
                "tech_constraints": tech_constraints,
                "target_users": target_users,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def review_code(self, code: str, review_focus: str = "architecture") -> AgentOutput:
        """Review code for architectural compliance."""
        task = Task(
            id=f"review_{self.project_id}",
            type=TaskType.CODE_REVIEW,
            description=f"Architecture review focused on: {review_focus}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"existing_code": code, "review_focus": review_focus},
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)
