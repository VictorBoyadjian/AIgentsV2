"""
Research Agent — web search, competitor analysis, documentation lookup.

Uses gpt-4o by default (superior web search capabilities via OpenAI).
Results are persisted in Weaviate vector store and cached in Redis.
Batch eligible: analyze_competitors.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole, get_settings
from core.tools.browser_tool import BrowserTool

logger = structlog.get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Senior Research Analyst agent for web research and analysis.

    Capabilities:
    - search_web: Search and synthesize web results
    - analyze_competitors: Competitive analysis (batch eligible)
    - find_documentation: Locate and summarize documentation
    - benchmark_technologies: Compare technologies

    Optimizations:
    - Results persisted in Weaviate to avoid re-searching
    - 24h Redis cache for identical queries
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.RESEARCH, project_id=project_id, **kwargs)
        self._browser = BrowserTool()
        self._redis: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection for result caching."""
        settings = get_settings()
        self._redis = aioredis.from_url(settings.redis.redis_url, decode_responses=True)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    def _build_task_prompt(self, task: Task) -> str:
        """Build research-specific prompt."""
        action = task.context.get("action", "")
        prompts = {
            "search_web": self._search_prompt,
            "analyze_competitors": self._competitors_prompt,
            "find_documentation": self._docs_prompt,
            "benchmark_technologies": self._benchmark_prompt,
        }
        builder = prompts.get(action, self._default_prompt)
        return builder(task)

    def _search_prompt(self, task: Task) -> str:
        query = task.context.get("query", task.description)
        search_results = task.context.get("search_results", "")
        return f"""Analyze and synthesize the following web search results.

## Search Query
{query}

## Search Results
{search_results}

## Requirements
1. Synthesize information from multiple sources
2. Highlight key findings and trends
3. Note conflicting information between sources
4. Cite sources with URLs
5. Provide actionable recommendations

## Output Format
### Key Findings
- Finding 1 [Source: URL]
- Finding 2 [Source: URL]

### Analysis
Detailed synthesis...

### Recommendations
1. ...
2. ..."""

    def _competitors_prompt(self, task: Task) -> str:
        product = task.context.get("product", task.description)
        market = task.context.get("market", "SaaS")
        search_results = task.context.get("search_results", "")
        return f"""Perform a competitive analysis for:

## Product
{product}

## Market
{market}

## Research Data
{search_results}

## Analysis Framework
For each competitor, provide:

| Aspect | Details |
|--------|---------|
| Company | Name, founding year, funding |
| Product | Core features |
| Pricing | Plans and pricing model |
| Target Market | Who they serve |
| Strengths | What they do well |
| Weaknesses | Where they fall short |
| Differentiator | Their unique value prop |

### Market Positioning Map
Describe positioning on axes:
- X: Price (Low → High)
- Y: Feature Completeness (Basic → Advanced)

### Competitive Advantage Opportunities
Identify gaps in the market that our product can exploit.

### Pricing Strategy Recommendation
Based on competitor pricing, recommend our pricing approach."""

    def _docs_prompt(self, task: Task) -> str:
        technology = task.context.get("technology", task.description)
        search_results = task.context.get("search_results", "")
        return f"""Find and summarize documentation for: {technology}

## Research Data
{search_results}

## Required Information
1. Official documentation links
2. Getting started guide summary
3. Key API reference points
4. Best practices and patterns
5. Common pitfalls and solutions
6. Community resources (forums, Discord, GitHub)

Provide structured, actionable documentation summary."""

    def _benchmark_prompt(self, task: Task) -> str:
        technologies = task.context.get("technologies", task.description)
        criteria = task.context.get("criteria", "performance, cost, ecosystem, learning curve")
        search_results = task.context.get("search_results", "")
        return f"""Benchmark and compare the following technologies:

## Technologies
{technologies}

## Evaluation Criteria
{criteria}

## Research Data
{search_results}

## Output Format

### Comparison Matrix
| Criterion | Tech A | Tech B | Tech C |
|-----------|--------|--------|--------|
| Performance | ... | ... | ... |
| Cost | ... | ... | ... |
| Ecosystem | ... | ... | ... |
| Learning Curve | ... | ... | ... |

### Detailed Analysis
For each technology, provide pros/cons and ideal use cases.

### Recommendation
Clear recommendation with justification."""

    def _default_prompt(self, task: Task) -> str:
        return f"Research task: {task.description}\n\nProvide comprehensive research findings."

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    async def _get_cached_result(self, query: str) -> str | None:
        """Check Redis cache for previous search results (24h TTL)."""
        if not self._redis:
            return None
        cache_key = f"research:{hashlib.sha256(query.encode()).hexdigest()}"
        return await self._redis.get(cache_key)

    async def _cache_result(self, query: str, result: str) -> None:
        """Cache search results in Redis with 24h TTL."""
        if not self._redis:
            return
        cache_key = f"research:{hashlib.sha256(query.encode()).hexdigest()}"
        await self._redis.setex(cache_key, 86400, result)  # 24h TTL

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------
    async def search_web(self, query: str) -> AgentOutput:
        """Search the web and synthesize results."""
        # Check cache first
        cached = await self._get_cached_result(query)
        if cached:
            logger.info("research_agent.cache_hit", query=query[:50])
            return AgentOutput(
                task_id=f"search_{self.project_id}",
                agent_role=self.role.value,
                content=cached,
                metadata={"cache_hit": True},
            )

        # Perform web search
        results = await self._browser.search(query, max_results=5)
        search_text = "\n\n".join(
            f"### {r.title}\nURL: {r.url}\n{r.snippet}" for r in results
        )

        task = Task(
            id=f"search_{self.project_id}",
            type=TaskType.RESEARCH,
            description=f"Web search: {query[:60]}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"action": "search_web", "query": query, "search_results": search_text},
            is_blocking=True,
            allow_batch=False,
        )
        output = await self.execute(task)

        # Cache result
        await self._cache_result(query, output.content)

        return output

    async def analyze_competitors(
        self, product: str, market: str = "SaaS"
    ) -> AgentOutput:
        """Competitive analysis (batch eligible)."""
        # Pre-fetch search results
        results = await self._browser.search(f"{product} competitors {market}", max_results=5)
        search_text = "\n\n".join(
            f"### {r.title}\nURL: {r.url}\n{r.snippet}" for r in results
        )

        task = Task(
            id=f"competitors_{self.project_id}",
            type=TaskType.RESEARCH,
            description=f"Competitor analysis: {product[:60]}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={
                "action": "analyze_competitors",
                "product": product,
                "market": market,
                "search_results": search_text,
            },
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)

    async def find_documentation(self, technology: str) -> AgentOutput:
        """Find and summarize documentation for a technology."""
        results = await self._browser.search(f"{technology} documentation guide", max_results=5)
        search_text = "\n\n".join(
            f"### {r.title}\nURL: {r.url}\n{r.snippet}" for r in results
        )

        task = Task(
            id=f"docs_{self.project_id}",
            type=TaskType.RESEARCH,
            description=f"Documentation: {technology[:60]}",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={
                "action": "find_documentation",
                "technology": technology,
                "search_results": search_text,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def benchmark_technologies(
        self, technologies: str, criteria: str = ""
    ) -> AgentOutput:
        """Compare and benchmark technologies."""
        results = await self._browser.search(f"{technologies} comparison benchmark", max_results=5)
        search_text = "\n\n".join(
            f"### {r.title}\nURL: {r.url}\n{r.snippet}" for r in results
        )

        task = Task(
            id=f"benchmark_{self.project_id}",
            type=TaskType.RESEARCH,
            description=f"Benchmark: {technologies[:60]}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={
                "action": "benchmark_technologies",
                "technologies": technologies,
                "criteria": criteria or "performance, cost, ecosystem, learning curve",
                "search_results": search_text,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)
