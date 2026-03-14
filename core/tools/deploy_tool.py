"""
Deployment tool for pushing applications to Railway.

Handles project creation, environment configuration, and deployment
triggering via the Railway API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)

RAILWAY_API_URL = "https://backboard.railway.com/graphql/v2"


@dataclass
class DeployResult:
    """Result of a deployment operation."""

    success: bool
    message: str
    deployment_url: str | None = None
    deployment_id: str | None = None
    logs: str | None = None


class DeployTool:
    """
    Railway deployment tool for deploying SaaS applications.

    Manages project creation, service setup, environment variables,
    and deployment triggering via Railway's GraphQL API.
    """

    def __init__(
        self,
        railway_token: str | None = None,
        project_id: str | None = None,
    ) -> None:
        settings = get_settings()
        self._token = railway_token or settings.tools.railway_token
        self._project_id = project_id or settings.tools.railway_project_id

    async def _graphql_request(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a GraphQL request against Railway API."""
        if not self._token:
            raise RuntimeError("Railway token not configured")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                RAILWAY_API_URL,
                json={"query": query, "variables": variables or {}},
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                errors = data["errors"]
                raise RuntimeError(f"Railway API error: {errors}")

            return data.get("data", {})

    async def create_project(self, name: str) -> DeployResult:
        """Create a new Railway project."""
        query = """
        mutation ProjectCreate($input: ProjectCreateInput!) {
            projectCreate(input: $input) {
                id
                name
            }
        }
        """
        try:
            data = await self._graphql_request(
                query, {"input": {"name": name}}
            )
            project = data.get("projectCreate", {})
            self._project_id = project.get("id")

            logger.info("deploy_tool.project_created", name=name, id=self._project_id)
            return DeployResult(
                success=True,
                message=f"Project '{name}' created",
                deployment_id=self._project_id,
            )
        except Exception as exc:
            logger.error("deploy_tool.create_project_failed", error=str(exc))
            return DeployResult(success=False, message=f"Failed to create project: {exc}")

    async def set_environment_variables(
        self,
        service_id: str,
        env_vars: dict[str, str],
        environment_id: str | None = None,
    ) -> DeployResult:
        """Set environment variables on a Railway service."""
        query = """
        mutation VariableCollectionUpsert($input: VariableCollectionUpsertInput!) {
            variableCollectionUpsert(input: $input)
        }
        """
        try:
            variables_input: dict[str, Any] = {
                "input": {
                    "projectId": self._project_id,
                    "serviceId": service_id,
                    "variables": env_vars,
                }
            }
            if environment_id:
                variables_input["input"]["environmentId"] = environment_id

            await self._graphql_request(query, variables_input)

            logger.info(
                "deploy_tool.env_vars_set",
                service_id=service_id,
                var_count=len(env_vars),
            )
            return DeployResult(
                success=True,
                message=f"Set {len(env_vars)} environment variables",
            )
        except Exception as exc:
            logger.error("deploy_tool.set_env_vars_failed", error=str(exc))
            return DeployResult(success=False, message=f"Failed to set env vars: {exc}")

    async def deploy_from_github(
        self,
        repo_url: str,
        branch: str = "main",
    ) -> DeployResult:
        """Trigger a deployment from a GitHub repository."""
        query = """
        mutation ServiceCreate($input: ServiceCreateInput!) {
            serviceCreate(input: $input) {
                id
                name
            }
        }
        """
        try:
            data = await self._graphql_request(query, {
                "input": {
                    "projectId": self._project_id,
                    "source": {"repo": repo_url},
                    "branch": branch,
                }
            })
            service = data.get("serviceCreate", {})
            service_id = service.get("id")

            logger.info(
                "deploy_tool.deploy_triggered",
                repo=repo_url,
                branch=branch,
                service_id=service_id,
            )
            return DeployResult(
                success=True,
                message=f"Deployment triggered from {repo_url}:{branch}",
                deployment_id=service_id,
            )
        except Exception as exc:
            logger.error("deploy_tool.deploy_failed", error=str(exc))
            return DeployResult(success=False, message=f"Deployment failed: {exc}")

    async def get_deployment_status(self, deployment_id: str) -> DeployResult:
        """Check the status of a deployment."""
        query = """
        query DeploymentStatus($id: String!) {
            deployment(id: $id) {
                id
                status
                staticUrl
            }
        }
        """
        try:
            data = await self._graphql_request(query, {"id": deployment_id})
            deployment = data.get("deployment", {})

            status = deployment.get("status", "unknown")
            url = deployment.get("staticUrl")

            return DeployResult(
                success=status == "SUCCESS",
                message=f"Deployment status: {status}",
                deployment_url=f"https://{url}" if url else None,
                deployment_id=deployment_id,
            )
        except Exception as exc:
            logger.error("deploy_tool.status_check_failed", error=str(exc))
            return DeployResult(success=False, message=f"Status check failed: {exc}")
