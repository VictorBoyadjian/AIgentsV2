"""
DevOps Agent — Dockerfile, CI/CD, infrastructure, and deployment.

Uses claude-haiku-4-5 by default (repetitive templates, no advanced reasoning).
Batch eligible: generate_cicd_pipeline, provision_infrastructure.
Real-time: deploy_to_railway (immediate feedback needed).
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole
from core.tools.deploy_tool import DeployTool

logger = structlog.get_logger(__name__)


class DevOpsAgent(BaseAgent):
    """
    Senior DevOps/SRE agent for CI/CD, containerization, and deployment.

    Capabilities:
    - generate_dockerfile: Create optimized Dockerfiles
    - generate_cicd_pipeline: CI/CD pipeline configs (batch eligible)
    - provision_infrastructure: Terraform/IaC (batch eligible)
    - deploy_to_railway: Deploy to Railway (real-time)
    - setup_monitoring: Monitoring configuration
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.DEVOPS, project_id=project_id, **kwargs)
        self._deploy_tool = DeployTool()

    def _build_task_prompt(self, task: Task) -> str:
        """Build DevOps-specific prompt."""
        action = task.context.get("action", "")
        prompts = {
            "generate_dockerfile": self._dockerfile_prompt,
            "generate_cicd_pipeline": self._cicd_prompt,
            "provision_infrastructure": self._iac_prompt,
            "setup_monitoring": self._monitoring_prompt,
        }
        builder = prompts.get(action, self._default_prompt)
        return builder(task)

    def _dockerfile_prompt(self, task: Task) -> str:
        tech_stack = task.context.get("tech_stack", "Python/FastAPI")
        app_type = task.context.get("app_type", "web API")
        return f"""Create an optimized, production-ready Dockerfile for:

## Application
- Type: {app_type}
- Stack: {tech_stack}
- Task: {task.description}

## Requirements
1. Multi-stage build (builder + runtime)
2. Non-root user
3. Minimal base image (slim/alpine)
4. Layer caching optimization (.dockerignore, copy order)
5. Health check endpoint
6. Security hardening (no unnecessary packages)
7. Environment variable configuration
8. Proper signal handling (exec form CMD)

Also provide:
- `.dockerignore` file
- `docker-compose.yml` for local development
- Build and run instructions

## Output
```dockerfile:Dockerfile
# Multi-stage production Dockerfile
```

```text:.dockerignore
# Docker ignore rules
```

```yaml:docker-compose.yml
# Local development compose
```"""

    def _cicd_prompt(self, task: Task) -> str:
        platform = task.context.get("platform", "github_actions")
        tech_stack = task.context.get("tech_stack", "Python")
        return f"""Generate a complete CI/CD pipeline for {platform}.

## Application
- Stack: {tech_stack}
- Task: {task.description}

## Pipeline Stages Required
1. **Lint & Format** — ruff, black, mypy
2. **Unit Tests** — pytest with coverage
3. **Integration Tests** — database + API tests
4. **Security Scan** — dependency audit, SAST
5. **Build** — Docker image build and push
6. **Deploy Staging** — auto-deploy to staging
7. **Deploy Production** — manual approval gate

## Requirements
- Cache dependencies between runs
- Parallel jobs where possible
- Secrets management (no hardcoded values)
- Branch protection rules
- Status checks required for merge
- Artifact upload for test reports

## Output
```yaml:.github/workflows/ci.yml
# Complete CI pipeline
```

```yaml:.github/workflows/deploy.yml
# Complete CD pipeline
```"""

    def _iac_prompt(self, task: Task) -> str:
        cloud = task.context.get("cloud_provider", "Railway")
        return f"""Generate infrastructure-as-code for {cloud}.

## Requirements
{task.description}

## Infrastructure Components
1. Application runtime (containers/serverless)
2. Database (PostgreSQL)
3. Cache (Redis)
4. Object storage (if needed)
5. CDN (if frontend)
6. DNS configuration
7. SSL/TLS certificates
8. Monitoring and alerting

## Output
Provide complete IaC configuration files with:
- Resource definitions
- Networking
- Security groups
- Environment variables
- Scaling policies

Use Terraform HCL format if multi-cloud, or provider-specific if single-cloud."""

    def _monitoring_prompt(self, task: Task) -> str:
        return f"""Set up monitoring and observability for the application.

## Task
{task.description}

## Components Required
1. **Prometheus** metrics configuration
   - Application metrics (request latency, error rate, throughput)
   - Custom business metrics
   - Infrastructure metrics

2. **Grafana** dashboards
   - Application overview dashboard
   - LLM cost monitoring dashboard
   - Error tracking dashboard

3. **Alerting rules**
   - Error rate > 5% → Warning
   - Error rate > 10% → Critical
   - Latency p99 > 2s → Warning
   - Budget exceeded → Emergency

4. **Structured logging**
   - JSON format
   - Correlation IDs
   - Log levels configuration

## Output
```yaml:prometheus/prometheus.yml
# Prometheus config
```

```json:grafana/dashboards/app-overview.json
# Grafana dashboard
```

```yaml:prometheus/alert-rules.yml
# Alert rules
```"""

    def _default_prompt(self, task: Task) -> str:
        return f"DevOps task: {task.description}\n\nProvide production-ready configurations."

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------
    async def generate_dockerfile(
        self, tech_stack: str = "Python/FastAPI", app_type: str = "web API"
    ) -> AgentOutput:
        """Generate optimized Dockerfile."""
        task = Task(
            id=f"docker_{self.project_id}",
            type=TaskType.DEPLOYMENT,
            description=f"Generate Dockerfile for {tech_stack} {app_type}",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={
                "action": "generate_dockerfile",
                "tech_stack": tech_stack,
                "app_type": app_type,
            },
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def generate_cicd_pipeline(
        self,
        platform: str = "github_actions",
        tech_stack: str = "Python",
    ) -> AgentOutput:
        """Generate CI/CD pipeline (batch eligible)."""
        task = Task(
            id=f"cicd_{self.project_id}",
            type=TaskType.DEPLOYMENT,
            description=f"Generate CI/CD for {platform}",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={
                "action": "generate_cicd_pipeline",
                "platform": platform,
                "tech_stack": tech_stack,
            },
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)

    async def provision_infrastructure(
        self, description: str, cloud_provider: str = "Railway"
    ) -> AgentOutput:
        """Generate IaC configuration (batch eligible)."""
        task = Task(
            id=f"iac_{self.project_id}",
            type=TaskType.DEPLOYMENT,
            description=f"Provision infrastructure on {cloud_provider}",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={
                "action": "provision_infrastructure",
                "cloud_provider": cloud_provider,
            },
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)

    async def deploy_to_railway(self, repo_url: str, branch: str = "main") -> AgentOutput:
        """Deploy to Railway (real-time, immediate feedback)."""
        result = await self._deploy_tool.deploy_from_github(repo_url, branch)
        return AgentOutput(
            task_id=f"deploy_{self.project_id}",
            agent_role=self.role.value,
            content=result.message,
            metadata={
                "success": result.success,
                "deployment_url": result.deployment_url,
                "deployment_id": result.deployment_id,
            },
        )

    async def setup_monitoring(self) -> AgentOutput:
        """Set up monitoring configuration."""
        task = Task(
            id=f"monitoring_{self.project_id}",
            type=TaskType.DEPLOYMENT,
            description="Set up monitoring and observability",
            complexity=TaskComplexity.MEDIUM,
            project_id=self.project_id,
            context={"action": "setup_monitoring"},
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)
