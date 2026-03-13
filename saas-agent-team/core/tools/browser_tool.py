"""
Browser / web scraping tool for research and documentation fetching.

Uses Firecrawl for structured web scraping and Tavily for web search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    score: float = 0.0


@dataclass
class ScrapedPage:
    """Result of scraping a web page."""

    url: str
    title: str
    content: str  # Markdown content
    metadata: dict[str, Any] = field(default_factory=dict)


class BrowserTool:
    """
    Web research tool combining search (Tavily) and scraping (Firecrawl).

    Used by the Research Agent for competitor analysis, documentation
    lookup, and technology benchmarking.
    """

    def __init__(
        self,
        tavily_api_key: str | None = None,
        firecrawl_api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self._tavily_key = tavily_api_key or settings.tools.tavily_api_key
        self._firecrawl_key = firecrawl_api_key or settings.tools.firecrawl_api_key

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_answer: bool = True,
    ) -> list[SearchResult]:
        """
        Search the web using Tavily API.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            include_answer: Whether to include AI-generated answer.

        Returns:
            List of SearchResult objects.
        """
        if not self._tavily_key:
            logger.warning("browser_tool.tavily_not_configured")
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._tavily_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": include_answer,
                        "search_depth": "advanced",
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score", 0.0),
                ))

            logger.info("browser_tool.search_completed", query=query, results=len(results))
            return results

        except Exception as exc:
            logger.error("browser_tool.search_failed", query=query, error=str(exc))
            return []

    async def scrape(self, url: str) -> ScrapedPage | None:
        """
        Scrape a web page using Firecrawl API.

        Returns the page content as clean markdown.
        """
        if not self._firecrawl_key:
            logger.warning("browser_tool.firecrawl_not_configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    json={
                        "url": url,
                        "formats": ["markdown"],
                    },
                    headers={
                        "Authorization": f"Bearer {self._firecrawl_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()

            page_data = data.get("data", {})
            result = ScrapedPage(
                url=url,
                title=page_data.get("metadata", {}).get("title", ""),
                content=page_data.get("markdown", ""),
                metadata=page_data.get("metadata", {}),
            )

            logger.info("browser_tool.scrape_completed", url=url, content_len=len(result.content))
            return result

        except Exception as exc:
            logger.error("browser_tool.scrape_failed", url=url, error=str(exc))
            return None

    async def search_and_scrape(
        self, query: str, max_results: int = 3
    ) -> list[ScrapedPage]:
        """Search the web and scrape the top results for full content."""
        results = await self.search(query, max_results=max_results)
        pages: list[ScrapedPage] = []

        for result in results:
            page = await self.scrape(result.url)
            if page:
                pages.append(page)

        return pages
