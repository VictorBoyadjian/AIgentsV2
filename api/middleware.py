"""
FastAPI middleware stack.

Provides:
- ``RequestTimingMiddleware``  -- logs wall-clock duration of every request.
- ``RateLimitMiddleware``      -- simple sliding-window in-memory rate limiter.
- ``configure_cors``           -- helper that wires up ``CORSMiddleware``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from core.config import get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request timing
# ---------------------------------------------------------------------------
class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Log the wall-clock duration of every HTTP request.

    Emits a structured log line at *info* level that includes method, path,
    status code, and elapsed milliseconds.  Useful for debugging and for
    feeding into Prometheus via the ``log`` exporter.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            status_code = response.status_code if response else 500
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(elapsed_ms, 2),
                client=request.client.host if request.client else "unknown",
            )


# ---------------------------------------------------------------------------
# Rate limiting (in-memory sliding window)
# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter.

    Tracks requests per client IP in a rolling window.  When the limit is
    exceeded, the middleware returns *429 Too Many Requests* with a
    ``Retry-After`` header.

    Parameters
    ----------
    app:
        The ASGI application.
    max_requests:
        Maximum number of requests allowed in *window_seconds*.
    window_seconds:
        Size of the sliding window in seconds.
    """

    def __init__(
        self,
        app: FastAPI,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_ip -> list of request timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window_seconds

        # Prune timestamps outside the current window
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(
                "rate_limit.exceeded",
                client=client_ip,
                path=request.url.path,
                limit=self.max_requests,
                window=self.window_seconds,
            )
            retry_after = int(
                self.window_seconds
                - (now - self._requests[client_ip][0])
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please retry later.",
                },
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# CORS helper
# ---------------------------------------------------------------------------
def configure_cors(app: FastAPI) -> None:
    """Add ``CORSMiddleware`` based on application settings.

    Origins are read from ``Settings.api.api_cors_origins``.  In development
    mode all methods and headers are allowed.
    """
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Duration-Ms", "X-RateLimit-Remaining"],
    )
