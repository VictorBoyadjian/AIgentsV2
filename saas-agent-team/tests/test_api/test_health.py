"""Tests for API endpoints — health, projects, costs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


class TestHealthEndpoint:
    """Test the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self) -> None:
        """Health endpoint should return status ok."""
        # We need to mock external services to avoid actual connections
        with patch("api.main.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()

            # Import after patching
            from api.main import create_app

            app = create_app()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] in ("healthy", "degraded")
