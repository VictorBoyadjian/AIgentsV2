"""
Structured logging configuration using structlog.

Provides a ``configure_logging()`` entry-point that wires up structlog with
JSON output for production and coloured console output for development.
Processors include ISO-8601 timestamps, log level, caller information, and
full exception rendering.  Standard-library ``logging`` is bridged so that
third-party libraries (uvicorn, httpx, sqlalchemy, etc.) emit structured
log events as well.

Usage::

    from observability.logger import configure_logging, get_logger

    configure_logging()           # call once at application startup
    logger = get_logger(__name__)
    logger.info("server.started", port=8000)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from core.config import Environment, get_settings


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------

def _add_environment(
    logger: logging.Logger,                    # noqa: ARG001
    method_name: str,                          # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject the current runtime environment into every log event."""
    try:
        settings = get_settings()
        event_dict["environment"] = settings.api.environment.value
    except Exception:
        event_dict["environment"] = os.getenv("ENVIRONMENT", "development")
    return event_dict


def _drop_color_message(
    logger: logging.Logger,                    # noqa: ARG001
    method_name: str,                          # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Remove the ``color_message`` key that uvicorn injects.

    Uvicorn adds an ANSI-coloured duplicate of the log message under the key
    ``color_message``.  In JSON output this is noise, so we strip it.
    """
    event_dict.pop("color_message", None)
    return event_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    *,
    log_level: str | None = None,
    force_json: bool | None = None,
) -> None:
    """Configure structured logging for the entire application.

    Parameters
    ----------
    log_level:
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
        ``CRITICAL``).  Falls back to the ``LOG_LEVEL`` environment variable,
        then to ``INFO``.
    force_json:
        When *True*, always render as JSON regardless of environment.  When
        *None* (the default), JSON is used in ``staging`` and ``production``
        while coloured console output is used in ``development``.
    """

    resolved_level = (
        log_level
        or os.getenv("LOG_LEVEL", "INFO")
    ).upper()

    numeric_level = getattr(logging, resolved_level, logging.INFO)

    # Decide output format --------------------------------------------------
    if force_json is not None:
        use_json = force_json
    else:
        try:
            settings = get_settings()
            use_json = settings.api.environment in (
                Environment.STAGING,
                Environment.PRODUCTION,
            )
        except Exception:
            env_str = os.getenv("ENVIRONMENT", "development").lower()
            use_json = env_str in ("staging", "production")

    # Shared processors (run for EVERY log event) ---------------------------
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_environment,
        _drop_color_message,
    ]

    if use_json:
        # JSON path: render exceptions as structured dicts
        shared_processors.append(
            structlog.processors.format_exc_info,
        )
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Console path: pretty colours + readable exception traces
        shared_processors.append(
            structlog.processors.ExceptionRenderer(
                exception_formatter=structlog.dev.plain_traceback,
            ),
        )
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    # structlog configuration -----------------------------------------------
    structlog.configure(
        processors=[
            *shared_processors,
            # Prepare event dict for stdlib integration
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # stdlib logging integration --------------------------------------------
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(numeric_level)

    # Tame noisy third-party loggers ----------------------------------------
    for noisy_logger_name in (
        "httpx",
        "httpcore",
        "uvicorn.access",
        "sqlalchemy.engine",
        "litellm",
        "celery",
        "watchfiles",
    ):
        logging.getLogger(noisy_logger_name).setLevel(
            max(numeric_level, logging.WARNING),
        )

    structlog.get_logger("observability.logger").info(
        "logging.configured",
        level=resolved_level,
        renderer="json" if use_json else "console",
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally named.

    This is a thin convenience wrapper around ``structlog.get_logger`` that
    carries the correct return type for IDE auto-complete and type-checkers.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]
