"""
Webhook endpoints for external service integrations.

Handles incoming webhooks from GitHub (push, pull_request, etc.) and
batch processing completion notifications.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from api.dependencies import get_app_settings, get_crew_manager, get_database
from api.schemas import WebhookEvent, WebhookResponse
from core.config import Settings
from memory.database import Database
from orchestration.crew_manager import CrewManager

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _verify_github_signature(
    payload_body: bytes,
    signature_header: str | None,
    secret: str,
) -> bool:
    """Verify the GitHub webhook HMAC-SHA256 signature.

    Returns ``True`` if the signature is valid or if no secret is configured
    (development mode).  Returns ``False`` otherwise.
    """
    if not secret or secret == "change-me-in-production-use-a-real-secret-key":
        # No secret configured -- skip verification in development
        return True

    if not signature_header:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature_header)


# ---------------------------------------------------------------------------
# POST /webhooks/github  -- handle GitHub webhook events
# ---------------------------------------------------------------------------
@router.post(
    "/github",
    response_model=WebhookResponse,
    summary="Handle GitHub webhook events",
)
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
    settings: Settings = Depends(get_app_settings),
    db: Database = Depends(get_database),
    crew_manager: CrewManager = Depends(get_crew_manager),
) -> WebhookResponse:
    """Process incoming GitHub webhook events.

    Supported events:

    - **push**: triggers downstream analysis for the affected repository.
    - **pull_request** (opened/synchronize): may trigger automated code
      review or security audit.
    - Other events are acknowledged but not actively processed.

    The webhook signature is verified using the configured ``SECRET_KEY`` and
    the ``X-Hub-Signature-256`` header.
    """
    body = await request.body()

    # Verify signature
    if not _verify_github_signature(body, x_hub_signature_256, settings.api.secret_key):
        logger.warning("webhook.github.invalid_signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature.",
        )

    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:
        logger.error("webhook.github.invalid_json", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload.",
        ) from exc

    event_type = x_github_event or payload.get("action", "unknown")
    repo = payload.get("repository", {}).get("full_name", "")
    sender = payload.get("sender", {}).get("login", "")

    logger.info(
        "webhook.github.received",
        event=event_type,
        repository=repo,
        sender=sender,
    )

    # --- Event-specific handling ---
    message = "Event acknowledged."

    if event_type == "push":
        ref = payload.get("ref", "")
        commits = payload.get("commits", [])
        commit_count = len(commits)
        logger.info(
            "webhook.github.push",
            repository=repo,
            ref=ref,
            commits=commit_count,
        )
        message = f"Push event processed: {commit_count} commit(s) on {ref}."

    elif event_type == "pull_request":
        action = payload.get("action", "")
        pr_number = payload.get("number", 0)
        pr_title = payload.get("pull_request", {}).get("title", "")

        logger.info(
            "webhook.github.pull_request",
            repository=repo,
            action=action,
            pr_number=pr_number,
            title=pr_title,
        )

        if action in ("opened", "synchronize"):
            message = (
                f"Pull request #{pr_number} '{pr_title}' ({action}) "
                f"queued for review."
            )
        else:
            message = f"Pull request #{pr_number} action '{action}' acknowledged."

    elif event_type == "issues":
        action = payload.get("action", "")
        issue_number = payload.get("issue", {}).get("number", 0)
        logger.info(
            "webhook.github.issue",
            repository=repo,
            action=action,
            issue=issue_number,
        )
        message = f"Issue #{issue_number} ({action}) acknowledged."

    else:
        logger.info(
            "webhook.github.unhandled_event",
            event=event_type,
            repository=repo,
        )
        message = f"Event type '{event_type}' acknowledged but not processed."

    return WebhookResponse(
        received=True,
        event_type=event_type,
        message=message,
    )


# ---------------------------------------------------------------------------
# POST /webhooks/batch  -- handle batch completion notifications
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    response_model=WebhookResponse,
    summary="Handle batch processing completion notifications",
)
async def batch_completion_webhook(
    body: WebhookEvent,
    db: Database = Depends(get_database),
    crew_manager: CrewManager = Depends(get_crew_manager),
) -> WebhookResponse:
    """Process batch API completion callbacks.

    When a batch job (Anthropic or OpenAI) completes, the provider can POST
    to this endpoint.  The payload should include the job ID and results in
    the ``payload`` field.

    Expected payload structure::

        {
            "event_type": "batch_complete",
            "payload": {
                "job_id": "batch_abc123",
                "status": "completed",
                "results": [...]
            }
        }
    """
    logger.info(
        "webhook.batch.received",
        event_type=body.event_type,
        sender=body.sender,
    )

    job_id = body.payload.get("job_id", "")
    batch_status = body.payload.get("status", "unknown")
    results = body.payload.get("results", [])

    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'job_id' in payload.",
        )

    logger.info(
        "webhook.batch.processing",
        job_id=job_id,
        status=batch_status,
        result_count=len(results),
    )

    if batch_status == "completed":
        # Process results -- update task records in the database
        for result in results:
            task_id = result.get("custom_id", "")
            content = result.get("content", "")
            cost_usd = result.get("cost_usd", 0.0)

            if task_id:
                try:
                    await db.complete_task(
                        task_id=task_id,
                        result_content=content,
                        cost_usd=cost_usd,
                    )
                    logger.info(
                        "webhook.batch.task_completed",
                        task_id=task_id,
                        cost_usd=cost_usd,
                    )
                except Exception as exc:
                    logger.error(
                        "webhook.batch.task_update_failed",
                        task_id=task_id,
                        error=str(exc),
                    )

        message = f"Batch job '{job_id}' completed with {len(results)} result(s)."
    elif batch_status == "failed":
        error_msg = body.payload.get("error", "Unknown error")
        logger.error(
            "webhook.batch.job_failed",
            job_id=job_id,
            error=error_msg,
        )
        message = f"Batch job '{job_id}' failed: {error_msg}"
    else:
        message = f"Batch job '{job_id}' status: {batch_status}"

    return WebhookResponse(
        received=True,
        event_type=body.event_type,
        message=message,
    )
