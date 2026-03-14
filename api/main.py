"""
FastAPI application entry point.

Creates the ASGI application, registers routers and middleware, wires up
the lifespan (startup / shutdown), and exposes ``/health`` plus ``/metrics``
endpoints.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from api.dependencies import (
    get_app_settings,
    get_cache,
    get_cost_tracker,
    get_crew_manager,
    get_database,
)
from api.middleware import (
    RateLimitMiddleware,
    RequestTimingMiddleware,
    configure_cors,
)
from api.routers import agents, costs, projects, tasks, webhooks
from api.schemas import ErrorResponse, HealthResponse

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup + shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise all infrastructure on startup; tear down on shutdown.

    Each component has a 15-second timeout so that a slow or unreachable
    dependency (DB, Redis, Weaviate) does not prevent the app from starting
    and passing Railway's healthcheck.
    """
    settings = get_app_settings()
    logger.info(
        "app.startup",
        environment=settings.api.environment.value,
        host=settings.api.api_host,
        port=settings.api.api_port,
    )

    INIT_TIMEOUT = 15  # seconds per component

    # -- Startup ----------------------------------------------------------
    database = get_database()
    cost_tracker = get_cost_tracker()
    cache = get_cache()
    crew_manager = get_crew_manager()

    for name, coro in [
        ("database", database.initialize()),
        ("cost_tracker", cost_tracker.initialize()),
        ("cache", cache.initialize()),
        ("crew_manager", crew_manager.initialize()),
    ]:
        try:
            await asyncio.wait_for(coro, timeout=INIT_TIMEOUT)
            logger.info("app.component_ready", component=name)
        except asyncio.TimeoutError:
            logger.error("app.component_timeout", component=name, timeout=INIT_TIMEOUT)
        except Exception as exc:
            logger.error(f"app.{name}_init_failed", error=str(exc))

    logger.info("app.startup_complete")

    yield

    # -- Shutdown ---------------------------------------------------------
    logger.info("app.shutdown")
    try:
        await crew_manager.shutdown()
    except Exception as exc:
        logger.error("app.crew_manager_shutdown_failed", error=str(exc))

    try:
        await cache.close()
    except Exception as exc:
        logger.error("app.cache_shutdown_failed", error=str(exc))

    try:
        await database.close()
    except Exception as exc:
        logger.error("app.database_shutdown_failed", error=str(exc))

    logger.info("app.shutdown_complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_app_settings()

    app = FastAPI(
        title="SaaS Agent Team API",
        description=(
            "Multi-agent AI team API for SaaS development. "
            "Manages projects, agents, tasks, costs, and workflows."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- Middleware (order matters: outermost first) ----------------------
    # Rate limiting sits outermost so it short-circuits early.
    app.add_middleware(RateLimitMiddleware, max_requests=200, window_seconds=60)
    # Request timing wraps the actual handler.
    app.add_middleware(RequestTimingMiddleware)
    # CORS must be added after Starlette-based middleware.
    configure_cors(app)

    # -- Routers ---------------------------------------------------------
    app.include_router(projects.router, prefix="/projects", tags=["Projects"])
    app.include_router(agents.router, prefix="/agents", tags=["Agents"])
    app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
    app.include_router(costs.router, prefix="/costs", tags=["Costs"])
    app.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])

    # -- Global exception handler ----------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "app.unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="Internal server error.",
                error_code="INTERNAL_ERROR",
            ).model_dump(),
        )

    # -- Health check ----------------------------------------------------
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
    )
    async def health_check() -> HealthResponse:
        """Return application health status and component readiness.

        Always returns HTTP 200 so Railway's healthcheck passes once
        uvicorn is accepting connections.  The ``status`` field indicates
        whether all components are operational (``ok``) or not (``degraded``).
        """
        components: dict[str, str] = {}

        # Check database
        try:
            db = get_database()
            async with db.get_session() as session:
                await session.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )
            components["database"] = "healthy"
        except Exception:
            components["database"] = "unhealthy"

        # Check cache (Redis)
        try:
            cache = get_cache()
            await cache.set("health_check", "ok", ttl=10)
            components["cache"] = "healthy"
        except Exception:
            components["cache"] = "unhealthy"

        overall = (
            "ok"
            if all(v == "healthy" for v in components.values())
            else "degraded"
        )

        # Always return 200 — Railway only needs the HTTP response to
        # confirm the process is alive.  Component status is in the body.
        return HealthResponse(
            status=overall,
            version="0.1.0",
            environment=settings.api.environment.value,
            components=components,
        )

    # -- Prometheus metrics endpoint -------------------------------------
    @app.get(
        "/metrics",
        tags=["System"],
        summary="Prometheus metrics",
        response_class=PlainTextResponse,
    )
    async def prometheus_metrics() -> PlainTextResponse:
        """Expose Prometheus-compatible metrics."""
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


# Singleton application instance used by ``uvicorn api.main:app``.
app: FastAPI = create_app()
