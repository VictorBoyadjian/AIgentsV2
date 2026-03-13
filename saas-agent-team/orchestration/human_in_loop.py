"""
Human-in-the-loop checkpoint management.

Provides mechanisms for pausing workflow execution to get human
approval, feedback, or decisions at critical points.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)


class CheckpointStatus(str, Enum):
    """Status of a human checkpoint."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMED_OUT = "timed_out"


@dataclass
class HumanCheckpoint:
    """A decision point requiring human input."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: str = ""
    description: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    status: CheckpointStatus = CheckpointStatus.PENDING
    decision: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None


class HumanInTheLoop:
    """
    Manages human checkpoints in the development workflow.

    Provides two modes:
    1. Async callback: register a callback that is invoked at checkpoints
    2. Polling: checkpoints are stored and polled via API

    In automated mode (no callback), checkpoints auto-approve after timeout.
    """

    def __init__(self, auto_approve_timeout: float = 3600.0) -> None:
        self._checkpoints: dict[str, HumanCheckpoint] = {}
        self._callback: Callable[..., Awaitable[dict[str, Any]]] | None = None
        self._auto_timeout = auto_approve_timeout
        self._events: dict[str, asyncio.Event] = {}

    def set_callback(
        self,
        callback: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        """Set async callback for checkpoint decisions."""
        self._callback = callback

    async def request_approval(
        self,
        phase: str,
        description: str,
        artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Request human approval at a checkpoint.

        If a callback is registered, calls it immediately.
        If polling, stores the checkpoint and waits for resolution.
        Auto-approves after timeout if no response.
        """
        checkpoint = HumanCheckpoint(
            phase=phase,
            description=description,
            artifacts=artifacts or {},
        )
        self._checkpoints[checkpoint.id] = checkpoint

        logger.info(
            "human_in_loop.checkpoint_created",
            id=checkpoint.id,
            phase=phase,
        )

        if self._callback:
            try:
                decision = await self._callback(
                    phase=phase,
                    artifacts=artifacts or {},
                )
                checkpoint.decision = decision
                checkpoint.status = (
                    CheckpointStatus.APPROVED
                    if decision.get("approved", False)
                    else CheckpointStatus.REJECTED
                )
                checkpoint.resolved_at = datetime.now(timezone.utc)
                return decision
            except Exception as exc:
                logger.error("human_in_loop.callback_error", error=str(exc))
                # Fall through to auto-approve

        # Polling mode: wait for external resolution
        event = asyncio.Event()
        self._events[checkpoint.id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=self._auto_timeout)
        except asyncio.TimeoutError:
            checkpoint.status = CheckpointStatus.TIMED_OUT
            checkpoint.decision = {"approved": True, "auto_approved": True}
            checkpoint.resolved_at = datetime.now(timezone.utc)
            logger.warning(
                "human_in_loop.auto_approved",
                id=checkpoint.id,
                phase=phase,
            )

        self._events.pop(checkpoint.id, None)
        return checkpoint.decision

    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        approved: bool,
        feedback: str = "",
        modifications: dict[str, Any] | None = None,
    ) -> bool:
        """Resolve a pending checkpoint (called from API)."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint or checkpoint.status != CheckpointStatus.PENDING:
            return False

        checkpoint.status = (
            CheckpointStatus.APPROVED if approved else CheckpointStatus.REJECTED
        )
        checkpoint.feedback = feedback
        checkpoint.decision = {
            "approved": approved,
            "feedback": feedback,
            **(modifications or {}),
        }
        checkpoint.resolved_at = datetime.now(timezone.utc)

        # Signal waiting coroutine
        event = self._events.get(checkpoint_id)
        if event:
            event.set()

        logger.info(
            "human_in_loop.checkpoint_resolved",
            id=checkpoint_id,
            approved=approved,
        )
        return True

    def get_pending_checkpoints(self) -> list[HumanCheckpoint]:
        """Get all pending checkpoints awaiting human decision."""
        return [
            cp for cp in self._checkpoints.values()
            if cp.status == CheckpointStatus.PENDING
        ]

    def get_checkpoint(self, checkpoint_id: str) -> HumanCheckpoint | None:
        """Get a specific checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
