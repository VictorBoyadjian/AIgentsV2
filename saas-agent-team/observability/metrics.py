"""
Prometheus metrics definitions for the SaaS Agent Team platform.

Defines counters, histograms, and gauges covering HTTP requests, agent
execution, LLM token usage, cost tracking, prompt-cache efficiency, and
batch processing.  All metric objects are module-level singletons so they
can be imported and used from anywhere in the application.

A ``setup_metrics()`` function is provided to expose the ``/metrics``
endpoint in a FastAPI application via ``prometheus_client``.

Usage::

    from observability.metrics import (
        HTTP_REQUEST_TOTAL,
        HTTP_REQUEST_DURATION_SECONDS,
        setup_metrics,
    )

    setup_metrics(app)  # call once during app lifespan

    HTTP_REQUEST_TOTAL.labels(method="GET", endpoint="/health", status="200").inc()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


# ---------------------------------------------------------------------------
# HTTP request metrics
# ---------------------------------------------------------------------------

HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests received.",
    labelnames=["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latency of HTTP requests in seconds.",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed.",
    labelnames=["method"],
)

# ---------------------------------------------------------------------------
# Agent execution metrics
# ---------------------------------------------------------------------------

AGENT_EXECUTIONS_TOTAL = Counter(
    "agent_executions_total",
    "Total number of agent task executions.",
    labelnames=["agent_role", "task_type", "status"],
)

AGENT_EXECUTION_DURATION_SECONDS = Histogram(
    "agent_execution_duration_seconds",
    "Duration of agent task execution in seconds.",
    labelnames=["agent_role", "task_type", "complexity"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

AGENT_SELF_REFLECT_SCORE = Histogram(
    "agent_self_reflect_score",
    "Distribution of agent self-reflection quality scores (0.0-1.0).",
    labelnames=["agent_role"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ---------------------------------------------------------------------------
# LLM token usage metrics
# ---------------------------------------------------------------------------

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens consumed across all LLM calls.",
    labelnames=["model", "token_type"],
)

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "Total number of LLM API calls.",
    labelnames=["model", "agent_role", "is_batch"],
)

LLM_CALL_DURATION_SECONDS = Histogram(
    "llm_call_duration_seconds",
    "Latency of individual LLM API calls in seconds.",
    labelnames=["model", "agent_role"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total number of LLM API call errors.",
    labelnames=["model", "agent_role", "error_type"],
)

# ---------------------------------------------------------------------------
# Cost metrics
# ---------------------------------------------------------------------------

COST_DAILY_USD = Gauge(
    "cost_daily_usd",
    "Current estimated daily LLM cost in USD.",
    labelnames=["project_id"],
)

COST_CUMULATIVE_USD = Counter(
    "cost_cumulative_usd",
    "Cumulative LLM cost in USD since process start.",
    labelnames=["project_id", "model", "agent_role"],
)

COST_SAVINGS_USD = Counter(
    "cost_savings_usd",
    "Cumulative cost savings in USD (cache + batch combined).",
    labelnames=["project_id", "saving_type"],
)

BUDGET_UTILISATION_RATIO = Gauge(
    "budget_utilisation_ratio",
    "Current budget utilisation as a ratio (0.0 = unused, 1.0 = fully spent).",
    labelnames=["project_id", "period"],
)

# ---------------------------------------------------------------------------
# Cache metrics
# ---------------------------------------------------------------------------

CACHE_OPERATIONS_TOTAL = Counter(
    "cache_operations_total",
    "Total prompt-cache operations (hits and misses).",
    labelnames=["result"],
)

CACHE_HIT_RATIO = Gauge(
    "cache_hit_ratio",
    "Rolling prompt-cache hit ratio (0.0-1.0).",
    labelnames=["agent_role"],
)

CACHE_TOKENS_SAVED = Counter(
    "cache_tokens_saved_total",
    "Total tokens served from prompt cache (avoiding full-price input charges).",
    labelnames=["model"],
)

# ---------------------------------------------------------------------------
# Batch processing metrics
# ---------------------------------------------------------------------------

BATCH_JOBS_SUBMITTED_TOTAL = Counter(
    "batch_jobs_submitted_total",
    "Total number of batch jobs submitted.",
    labelnames=["provider", "priority"],
)

BATCH_JOBS_COMPLETED_TOTAL = Counter(
    "batch_jobs_completed_total",
    "Total number of batch jobs that completed successfully.",
    labelnames=["provider"],
)

BATCH_JOBS_FAILED_TOTAL = Counter(
    "batch_jobs_failed_total",
    "Total number of batch jobs that failed.",
    labelnames=["provider"],
)

BATCH_TASKS_IN_FLIGHT = Gauge(
    "batch_tasks_in_flight",
    "Number of batch tasks currently in-flight (submitted but not yet completed).",
    labelnames=["provider"],
)

BATCH_TURNAROUND_SECONDS = Histogram(
    "batch_turnaround_seconds",
    "Wall-clock time from batch submission to result retrieval.",
    labelnames=["provider"],
    buckets=(60, 300, 900, 1800, 3600, 7200, 14400, 43200, 86400),
)


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup_metrics(app: "FastAPI") -> None:
    """Register the ``/metrics`` endpoint on *app* for Prometheus scraping.

    This wires up a plain-text endpoint that returns all registered
    ``prometheus_client`` metrics in the Prometheus exposition format.

    Parameters
    ----------
    app:
        The FastAPI application instance (imported lazily to avoid circular
        dependencies at module level).
    """
    from fastapi import Response

    @app.get(
        "/metrics",
        include_in_schema=False,
        tags=["observability"],
    )
    async def metrics_endpoint() -> Response:
        """Prometheus metrics scrape endpoint."""
        body = generate_latest(REGISTRY)
        return Response(
            content=body,
            media_type=CONTENT_TYPE_LATEST,
        )
