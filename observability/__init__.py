"""
Observability package for the SaaS Agent Team platform.

Provides structured logging, Prometheus metrics, and LangSmith tracing.

Quick start::

    from observability import configure_logging, setup_metrics, LangSmithTracer

    # At application boot:
    configure_logging()
    setup_metrics(app)
    tracer = LangSmithTracer()
"""

from observability.langsmith_tracer import LangSmithTracer, TracingSpan
from observability.logger import configure_logging, get_logger
from observability.metrics import (
    AGENT_EXECUTION_DURATION_SECONDS,
    AGENT_EXECUTIONS_TOTAL,
    AGENT_SELF_REFLECT_SCORE,
    BATCH_JOBS_COMPLETED_TOTAL,
    BATCH_JOBS_FAILED_TOTAL,
    BATCH_JOBS_SUBMITTED_TOTAL,
    BATCH_TASKS_IN_FLIGHT,
    BATCH_TURNAROUND_SECONDS,
    BUDGET_UTILISATION_RATIO,
    CACHE_HIT_RATIO,
    CACHE_OPERATIONS_TOTAL,
    CACHE_TOKENS_SAVED,
    COST_CUMULATIVE_USD,
    COST_DAILY_USD,
    COST_SAVINGS_USD,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUEST_TOTAL,
    HTTP_REQUESTS_IN_PROGRESS,
    LLM_CALL_DURATION_SECONDS,
    LLM_CALLS_TOTAL,
    LLM_ERRORS_TOTAL,
    LLM_TOKENS_TOTAL,
    setup_metrics,
)

__all__ = [
    # Logger
    "configure_logging",
    "get_logger",
    # Metrics
    "setup_metrics",
    "HTTP_REQUEST_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "HTTP_REQUESTS_IN_PROGRESS",
    "AGENT_EXECUTIONS_TOTAL",
    "AGENT_EXECUTION_DURATION_SECONDS",
    "AGENT_SELF_REFLECT_SCORE",
    "LLM_TOKENS_TOTAL",
    "LLM_CALLS_TOTAL",
    "LLM_CALL_DURATION_SECONDS",
    "LLM_ERRORS_TOTAL",
    "COST_DAILY_USD",
    "COST_CUMULATIVE_USD",
    "COST_SAVINGS_USD",
    "BUDGET_UTILISATION_RATIO",
    "CACHE_OPERATIONS_TOTAL",
    "CACHE_HIT_RATIO",
    "CACHE_TOKENS_SAVED",
    "BATCH_JOBS_SUBMITTED_TOTAL",
    "BATCH_JOBS_COMPLETED_TOTAL",
    "BATCH_JOBS_FAILED_TOTAL",
    "BATCH_TASKS_IN_FLIGHT",
    "BATCH_TURNAROUND_SECONDS",
    # LangSmith
    "LangSmithTracer",
    "TracingSpan",
]
