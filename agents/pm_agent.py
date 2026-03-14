"""
Product Manager Agent — generates PRDs, user stories, and roadmaps.

Uses claude-sonnet-4-5 for writing + specs quality.
Batch-eligible: prioritize_backlog, generate_roadmap.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole

logger = structlog.get_logger(__name__)


class PMAgent(BaseAgent):
    """
    Senior Product Manager agent for SaaS B2B products.

    Capabilities:
    - generate_prd: Create a Product Requirements Document
    - create_user_stories: Write user stories with Gherkin acceptance criteria
    - prioritize_backlog: RICE-based prioritization (batch eligible)
    - generate_roadmap: Product roadmap generation (batch eligible)
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.PM, project_id=project_id, **kwargs)

    def _build_task_prompt(self, task: Task) -> str:
        """Build PM-specific prompt."""
        task_map = {
            "generate_prd": self._prd_prompt,
            "create_user_stories": self._stories_prompt,
            "prioritize_backlog": self._backlog_prompt,
            "generate_roadmap": self._roadmap_prompt,
        }
        builder = task_map.get(task.context.get("action", ""), self._default_prompt)
        return builder(task)

    def _prd_prompt(self, task: Task) -> str:
        """Build PRD generation prompt."""
        product_idea = task.context.get("product_idea", task.description)
        target_market = task.context.get("target_market", "B2B SaaS")
        architecture = task.context.get("architecture", "")

        arch_section = f"\n## Architecture Reference\n{architecture}" if architecture else ""

        return f"""Generate a complete Product Requirements Document (PRD) for:

## Product Idea
{product_idea}

## Target Market
{target_market}
{arch_section}

## PRD Structure Required

### 1. Executive Summary
- Product vision (1-2 sentences)
- Problem statement
- Target users and personas

### 2. Goals & Success Metrics
- Primary KPIs
- Success criteria with measurable targets

### 3. User Personas (2-3 personas)
For each:
- Name, role, company size
- Goals, frustrations
- Jobs-to-be-Done

### 4. Feature Specifications
For each feature:
- Title
- User story: "As a [persona], I want [goal] so that [benefit]"
- Priority: P0 (must-have) / P1 (should-have) / P2 (nice-to-have)
- Acceptance criteria in Gherkin format (Given/When/Then)
- Dependencies

### 5. MVP Scope
- In-scope features (P0 only)
- Out-of-scope (explicitly listed)
- MVP success criteria

### 6. Technical Requirements
- Performance requirements
- Scalability requirements
- Security requirements
- Compliance requirements

### 7. Timeline (high-level)
- Phase 1: MVP
- Phase 2: Growth features
- Phase 3: Scale features

Output as structured markdown."""

    def _stories_prompt(self, task: Task) -> str:
        """Build user stories generation prompt."""
        prd = task.context.get("prd", "")
        feature = task.context.get("feature", task.description)

        return f"""Create detailed user stories for the following feature:

## Feature
{feature}

## PRD Context
{prd[:3000] if prd else "No PRD provided — infer reasonable requirements."}

## Output Format
For each user story, provide:

### US-XXX: [Title]
**As a** [persona]
**I want** [goal]
**So that** [benefit]

**Priority:** P0/P1/P2
**Story Points:** 1/2/3/5/8/13

**Acceptance Criteria:**
```gherkin
Feature: [Feature name]

  Scenario: [Happy path]
    Given [precondition]
    When [action]
    Then [expected result]

  Scenario: [Edge case]
    Given [precondition]
    When [action]
    Then [expected result]
```

**Technical Notes:**
- Implementation hints
- API endpoints needed
- Database changes needed

Generate 5-10 user stories covering the full feature scope."""

    def _backlog_prompt(self, task: Task) -> str:
        """Build backlog prioritization prompt."""
        stories = task.context.get("stories", "")
        return f"""Prioritize the following backlog using RICE scoring:

## Backlog Items
{stories or task.description}

## RICE Framework
For each item calculate:
- **Reach**: How many users will this impact in a quarter? (1-10)
- **Impact**: How much will this move the needle? (0.25=minimal, 0.5=low, 1=medium, 2=high, 3=massive)
- **Confidence**: How sure are we? (0.5=low, 0.8=medium, 1.0=high)
- **Effort**: Person-weeks to implement (1-10)

**RICE Score** = (Reach × Impact × Confidence) / Effort

Output as a sorted table (highest RICE first) with justifications."""

    def _roadmap_prompt(self, task: Task) -> str:
        """Build roadmap generation prompt."""
        prd = task.context.get("prd", "")
        priorities = task.context.get("priorities", "")
        return f"""Generate a product roadmap based on:

## PRD
{prd[:3000] if prd else "No PRD provided."}

## Priorities
{priorities or "Use RICE scores to determine order."}

## Output Format
### Q1: Foundation
- Feature 1 (P0) — [description] — [team] — [weeks]
- Feature 2 (P0) — ...

### Q2: Growth
...

### Q3: Scale
...

### Q4: Optimization
...

Include milestones, dependencies between features, and risk markers."""

    def _default_prompt(self, task: Task) -> str:
        """Default prompt for unrecognized PM tasks."""
        return f"Product management task: {task.description}\n\nProvide your analysis and deliverables."

    # ------------------------------------------------------------------
    # High-level capability methods
    # ------------------------------------------------------------------
    async def generate_prd(
        self,
        product_idea: str,
        target_market: str = "B2B SaaS",
        architecture: str = "",
    ) -> AgentOutput:
        """Generate a Product Requirements Document."""
        task = Task(
            id=f"prd_{self.project_id}",
            type=TaskType.PROJECT_MANAGEMENT,
            description=f"Generate PRD for: {product_idea[:80]}",
            complexity=TaskComplexity.COMPLEX,
            project_id=self.project_id,
            context={
                "action": "generate_prd",
                "product_idea": product_idea,
                "target_market": target_market,
                "architecture": architecture,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def create_user_stories(self, feature: str, prd: str = "") -> AgentOutput:
        """Create user stories with Gherkin acceptance criteria."""
        task = Task(
            id=f"stories_{self.project_id}",
            type=TaskType.PROJECT_MANAGEMENT,
            description=f"User stories for: {feature[:80]}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"action": "create_user_stories", "feature": feature, "prd": prd},
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def prioritize_backlog(self, stories: str) -> AgentOutput:
        """RICE-based backlog prioritization (batch eligible)."""
        task = Task(
            id=f"backlog_{self.project_id}",
            type=TaskType.PROJECT_MANAGEMENT,
            description="Prioritize backlog with RICE",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"action": "prioritize_backlog", "stories": stories},
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)

    async def generate_roadmap(self, prd: str = "", priorities: str = "") -> AgentOutput:
        """Generate product roadmap (batch eligible)."""
        task = Task(
            id=f"roadmap_{self.project_id}",
            type=TaskType.PROJECT_MANAGEMENT,
            description="Generate product roadmap",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"action": "generate_roadmap", "prd": prd, "priorities": priorities},
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)
