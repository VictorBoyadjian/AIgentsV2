"""
LangSmith tracing integration for the SaaS Agent Team platform.

Provides ``LangSmithTracer``, a thin wrapper around the ``langsmith`` SDK
that exposes context-manager helpers for tracing agent and LLM calls.
When the ``LANGSMITH_API_KEY`` environment variable (or the corresponding
``ObservabilityConfig.langsmith_api_key`` setting) is empty, all tracing
operations silently become no-ops so that the rest of the application is
not affected.

Usage::

    from observability.langsmith_tracer import LangSmithTracer

    tracer = LangSmithTracer()

    async with tracer.trace_agent_call(
        agent_role="dev",
        task_id="task-42",
        task_type="code_generation",
    ) as span:
        # ... do work ...
        span.metadata["files_generated"] = 3

    async with tracer.trace_llm_call(
        model="claude-sonnet-4-5",
        agent_role="dev",
    ) as span:
        # ... call LLM ...
        span.end(
            output_text=response_text,
            input_tokens=120,
            output_tokens=450,
        )
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Span data holder
# ---------------------------------------------------------------------------

@dataclass
class TracingSpan:
    """Lightweight data object representing a single tracing span.

    Callers may mutate ``metadata`` and ``outputs`` while the span is open
    and call ``end()`` to record final outputs.  ``LangSmithTracer`` reads
    these fields when the context-manager block exits to submit the run.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    run_type: str = "chain"
    metadata: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    parent_run_id: str | None = None

    # Convenience counters for LLM spans
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def end(
        self,
        *,
        output_text: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
        **extra_outputs: Any,
    ) -> None:
        """Finalise the span with output data.

        Parameters
        ----------
        output_text:
            The main textual output of the traced operation.
        input_tokens:
            Number of input tokens consumed (LLM calls).
        output_tokens:
            Number of output tokens produced (LLM calls).
        error:
            Error message if the operation failed.
        **extra_outputs:
            Arbitrary key-value pairs added to ``outputs``.
        """
        self.end_time = datetime.now(timezone.utc)
        if output_text is not None:
            self.outputs["output"] = output_text
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        if error is not None:
            self.error = error
        self.outputs.update(extra_outputs)


# ---------------------------------------------------------------------------
# LangSmithTracer
# ---------------------------------------------------------------------------

class LangSmithTracer:
    """Wrapper around the ``langsmith`` SDK for distributed tracing.

    Initialises a LangSmith ``Client`` from configuration.  If the API key
    is missing or the ``langsmith`` package cannot be imported, the tracer
    enters **noop mode** -- all public methods return immediately without
    side-effects, so the rest of the codebase never needs to guard against
    an unconfigured tracer.

    Parameters
    ----------
    api_key:
        LangSmith API key.  When *None*, read from settings /
        ``LANGSMITH_API_KEY`` environment variable.
    project_name:
        LangSmith project name.  When *None*, read from settings /
        ``LANGSMITH_PROJECT`` environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        project_name: str | None = None,
    ) -> None:
        self._enabled: bool = False
        self._client: Any = None
        self._project_name: str = "saas-agent-team"

        # Resolve configuration -------------------------------------------------
        try:
            settings = get_settings()
            resolved_key = api_key or settings.observability.langsmith_api_key
            resolved_project = (
                project_name or settings.observability.langsmith_project
            )
        except Exception:
            import os

            resolved_key = api_key or os.getenv("LANGSMITH_API_KEY", "")
            resolved_project = project_name or os.getenv(
                "LANGSMITH_PROJECT", "saas-agent-team"
            )

        if not resolved_key:
            logger.info(
                "langsmith_tracer.disabled",
                reason="LANGSMITH_API_KEY not set",
            )
            return

        self._project_name = resolved_project or "saas-agent-team"

        # Import and initialise the SDK -----------------------------------------
        try:
            from langsmith import Client as LangSmithClient  # type: ignore[import-untyped]

            self._client = LangSmithClient(api_key=resolved_key)
            self._enabled = True
            logger.info(
                "langsmith_tracer.enabled",
                project=self._project_name,
            )
        except ImportError:
            logger.warning(
                "langsmith_tracer.disabled",
                reason="langsmith package not installed",
            )
        except Exception as exc:
            logger.warning(
                "langsmith_tracer.init_failed",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return *True* if LangSmith tracing is active."""
        return self._enabled

    @property
    def project_name(self) -> str:
        """Return the configured LangSmith project name."""
        return self._project_name

    # ------------------------------------------------------------------
    # Agent-level tracing
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def trace_agent_call(
        self,
        *,
        agent_role: str,
        task_id: str,
        task_type: str = "",
        complexity: str = "medium",
        project_id: str = "default",
        parent_run_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TracingSpan]:
        """Context manager that traces an entire agent task execution.

        Yields a ``TracingSpan`` that the caller can enrich while work
        proceeds.  On exit, the completed run is submitted to LangSmith
        (or silently skipped in noop mode).

        Parameters
        ----------
        agent_role:
            The role of the agent (``dev``, ``qa``, ``architect``, ...).
        task_id:
            Unique identifier of the task being executed.
        task_type:
            Semantic category (``code_generation``, ``security_audit``, ...).
        complexity:
            Task complexity level (``simple``, ``medium``, ``complex``,
            ``critical``).
        project_id:
            Project identifier for multi-tenant isolation.
        parent_run_id:
            Optional parent run ID for hierarchical traces.
        extra_metadata:
            Arbitrary metadata merged into the span.
        """
        span = TracingSpan(
            name=f"agent:{agent_role}/{task_type or 'execute'}",
            run_type="chain",
            inputs={
                "task_id": task_id,
                "task_type": task_type,
                "complexity": complexity,
            },
            metadata={
                "agent_role": agent_role,
                "project_id": project_id,
                "task_id": task_id,
                **(extra_metadata or {}),
            },
            parent_run_id=parent_run_id,
        )

        start_time = time.perf_counter()

        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            if span.end_time is None:
                span.end_time = datetime.now(timezone.utc)
            span.metadata["duration_seconds"] = round(elapsed, 4)

            await self._submit_run(span)

    # ------------------------------------------------------------------
    # LLM-level tracing
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def trace_llm_call(
        self,
        *,
        model: str,
        agent_role: str = "",
        prompt_tokens: int | None = None,
        parent_run_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TracingSpan]:
        """Context manager that traces a single LLM API call.

        Yields a ``TracingSpan`` that the caller populates with token counts
        and output text via ``span.end(...)``.

        Parameters
        ----------
        model:
            LLM model identifier (``claude-sonnet-4-5``, ``gpt-4o``, ...).
        agent_role:
            Which agent initiated the call.
        prompt_tokens:
            Known prompt token count (if available before the call).
        parent_run_id:
            Optional parent run ID for hierarchical traces.
        extra_metadata:
            Arbitrary metadata merged into the span.
        """
        span = TracingSpan(
            name=f"llm:{model}",
            run_type="llm",
            inputs={"model": model},
            metadata={
                "model": model,
                "agent_role": agent_role,
                **(extra_metadata or {}),
            },
            parent_run_id=parent_run_id,
        )

        if prompt_tokens is not None:
            span.input_tokens = prompt_tokens

        start_time = time.perf_counter()

        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            if span.end_time is None:
                span.end_time = datetime.now(timezone.utc)
            span.metadata["duration_seconds"] = round(elapsed, 4)
            span.metadata["input_tokens"] = span.input_tokens
            span.metadata["output_tokens"] = span.output_tokens
            span.metadata["total_tokens"] = span.total_tokens

            await self._submit_run(span)

    # ------------------------------------------------------------------
    # Generic tracing
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent_run_id: str | None = None,
    ) -> AsyncIterator[TracingSpan]:
        """General-purpose tracing context manager.

        Useful for tracing tool calls, retrieval operations, or any
        arbitrary block of work.

        Parameters
        ----------
        name:
            Human-readable span name.
        run_type:
            LangSmith run type (``chain``, ``tool``, ``retriever``, ``llm``).
        inputs:
            Input data recorded on the span.
        metadata:
            Arbitrary metadata merged into the span.
        parent_run_id:
            Optional parent run ID for hierarchical traces.
        """
        span = TracingSpan(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            metadata=metadata or {},
            parent_run_id=parent_run_id,
        )

        start_time = time.perf_counter()

        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            if span.end_time is None:
                span.end_time = datetime.now(timezone.utc)
            span.metadata["duration_seconds"] = round(elapsed, 4)

            await self._submit_run(span)

    # ------------------------------------------------------------------
    # Internal: submit run to LangSmith
    # ------------------------------------------------------------------

    async def _submit_run(self, span: TracingSpan) -> None:
        """Submit a completed span to LangSmith.

        Runs the blocking ``client.create_run`` call in a thread-pool
        executor to avoid blocking the event loop.  Errors are logged and
        swallowed so that tracing failures never crash the application.
        """
        if not self._enabled or self._client is None:
            return

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._create_run_sync, span)
        except Exception as exc:
            logger.warning(
                "langsmith_tracer.submit_failed",
                run_id=span.run_id,
                name=span.name,
                error=str(exc),
            )

    def _create_run_sync(self, span: TracingSpan) -> None:
        """Synchronous call to create a LangSmith run (called from executor)."""
        run_kwargs: dict[str, Any] = {
            "id": span.run_id,
            "name": span.name,
            "run_type": span.run_type,
            "inputs": span.inputs,
            "outputs": span.outputs if span.outputs else None,
            "error": span.error,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "extra": {"metadata": span.metadata},
            "project_name": self._project_name,
        }

        if span.parent_run_id:
            run_kwargs["parent_run_id"] = span.parent_run_id

        # Include token usage for LLM runs
        if span.run_type == "llm" and (span.input_tokens or span.output_tokens):
            run_kwargs["extra"]["runtime"] = {
                "completion_tokens": span.output_tokens,
                "prompt_tokens": span.input_tokens,
                "total_tokens": span.total_tokens,
            }

        self._client.create_run(**run_kwargs)

        logger.debug(
            "langsmith_tracer.run_submitted",
            run_id=span.run_id,
            name=span.name,
            run_type=span.run_type,
        )
