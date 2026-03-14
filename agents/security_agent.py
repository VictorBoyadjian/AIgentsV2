"""
Security Agent — code audits, dependency checks, auth review.

Uses claude-sonnet-4-5 (critical analysis, no quality compromise).
Upgrades to Opus on CRITICAL vulnerability detection.
Batch eligible: generate_security_checklist, check_dependencies.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.base_agent import AgentOutput, BaseAgent, Task, TaskComplexity, TaskType
from core.config import AgentRole

logger = structlog.get_logger(__name__)


class SecurityAgent(BaseAgent):
    """
    Senior Security Engineer agent for AppSec.

    Capabilities:
    - audit_code: OWASP Top 10 code audit (prompt caching on code)
    - check_dependencies: Dependency vulnerability scan (batch eligible)
    - review_auth: Authentication/authorization review
    - generate_security_checklist: Pre-deployment checklist (batch eligible)

    Auto-upgrades to Opus on CRITICAL findings for deeper analysis.
    """

    def __init__(self, project_id: str = "default", **kwargs: Any) -> None:
        super().__init__(role=AgentRole.SECURITY, project_id=project_id, **kwargs)

    def _build_task_prompt(self, task: Task) -> str:
        """Build security-specific prompt."""
        action = task.context.get("action", "")
        prompts = {
            "audit_code": self._audit_prompt,
            "check_dependencies": self._deps_prompt,
            "review_auth": self._auth_prompt,
            "generate_security_checklist": self._checklist_prompt,
        }
        builder = prompts.get(action, self._default_prompt)
        return builder(task)

    def _audit_prompt(self, task: Task) -> str:
        focus_areas = task.context.get("focus_areas", "OWASP Top 10")
        return f"""Perform a comprehensive security audit of the provided code.

## Focus Areas
{focus_areas}

## OWASP Top 10 Checklist
Analyze for each vulnerability category:
1. **A01 Broken Access Control** — missing authorization checks, IDOR
2. **A02 Cryptographic Failures** — weak encryption, exposed secrets
3. **A03 Injection** — SQL injection, command injection, XSS
4. **A04 Insecure Design** — missing threat modeling, business logic flaws
5. **A05 Security Misconfiguration** — default configs, verbose errors
6. **A06 Vulnerable Components** — outdated dependencies
7. **A07 Authentication Failures** — weak passwords, missing MFA
8. **A08 Data Integrity Failures** — insecure deserialization, unsigned updates
9. **A09 Logging Failures** — missing audit logs, log injection
10. **A10 SSRF** — server-side request forgery

## Output Format
```json
{{
    "overall_risk_level": "low|medium|high|critical",
    "findings": [
        {{
            "id": "SEC-001",
            "severity": "critical|high|medium|low|informational",
            "category": "OWASP A01-A10",
            "title": "...",
            "description": "...",
            "affected_code": "file:line",
            "proof_of_concept": "...",
            "remediation": "...",
            "cwe_id": "CWE-XXX"
        }}
    ],
    "recommendations": ["..."],
    "positive_findings": ["things done well"]
}}
```"""

    def _deps_prompt(self, task: Task) -> str:
        dependencies = task.context.get("dependencies", "")
        return f"""Analyze project dependencies for known vulnerabilities.

## Dependencies
{dependencies or task.description}

## Analysis Required
1. Check each dependency against known CVE databases
2. Identify outdated packages with known vulnerabilities
3. Flag packages with permissive licenses (if commercial use)
4. Suggest secure alternatives for vulnerable packages
5. Check for dependency confusion risks

## Output Format
```json
{{
    "vulnerable_packages": [
        {{
            "package": "...",
            "current_version": "...",
            "vulnerability": "CVE-XXXX-XXXX",
            "severity": "critical|high|medium|low",
            "fixed_version": "...",
            "recommendation": "..."
        }}
    ],
    "outdated_packages": [...],
    "license_issues": [...],
    "overall_risk": "low|medium|high"
}}
```"""

    def _auth_prompt(self, task: Task) -> str:
        return f"""Review the authentication and authorization implementation.

## Task
{task.description}

## Review Checklist
1. **Authentication**
   - Password hashing algorithm (bcrypt/argon2?)
   - Session management (token expiry, rotation)
   - MFA implementation
   - OAuth2/OIDC flows
   - Rate limiting on auth endpoints

2. **Authorization**
   - RBAC/ABAC implementation
   - Permission checks on every endpoint
   - Resource-level access control (no IDOR)
   - API key management

3. **Token Security**
   - JWT validation (algorithm, expiry, issuer)
   - Refresh token rotation
   - Token storage (httpOnly cookies vs localStorage)

4. **Session Security**
   - CSRF protection
   - Session fixation prevention
   - Concurrent session handling

Provide findings as structured JSON with severity levels."""

    def _checklist_prompt(self, task: Task) -> str:
        app_type = task.context.get("app_type", "SaaS web application")
        return f"""Generate a pre-deployment security checklist for: {app_type}

## Checklist Categories
1. **Infrastructure Security** — TLS, firewalls, network segmentation
2. **Application Security** — input validation, output encoding, CSP headers
3. **Authentication & Authorization** — MFA, session management, RBAC
4. **Data Protection** — encryption at rest/transit, PII handling, backups
5. **Logging & Monitoring** — audit logs, intrusion detection, alerting
6. **API Security** — rate limiting, input validation, versioning
7. **Dependency Security** — vulnerability scanning, license compliance
8. **CI/CD Security** — secret management, signed builds, SAST/DAST
9. **Compliance** — GDPR, SOC2, HIPAA (as applicable)

Format as a checklist with [ ] markers and priority levels."""

    def _default_prompt(self, task: Task) -> str:
        return f"Security analysis task: {task.description}\n\nProvide your security assessment."

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------
    async def audit_code(self, code: str, focus_areas: str = "OWASP Top 10") -> AgentOutput:
        """Full security audit of code (prompt caching on code)."""
        task = Task(
            id=f"audit_{self.project_id}",
            type=TaskType.SECURITY_AUDIT,
            description=f"Security audit: {focus_areas}",
            complexity=TaskComplexity.COMPLEX,
            project_id=self.project_id,
            context={
                "action": "audit_code",
                "existing_code": code,
                "focus_areas": focus_areas,
            },
            is_blocking=True,
            allow_batch=False,
        )
        output = await self.execute(task)

        # Auto-upgrade to Opus if critical finding detected
        if '"critical"' in output.content.lower():
            logger.warning("security_agent.critical_finding_detected", project=self.project_id)
            # Re-run with Opus for deeper analysis
            task.complexity = TaskComplexity.CRITICAL
            task.id = f"audit_deep_{self.project_id}"
            output = await self.execute(task)

        return output

    async def check_dependencies(self, dependencies: str) -> AgentOutput:
        """Check dependencies for vulnerabilities (batch eligible)."""
        task = Task(
            id=f"deps_{self.project_id}",
            type=TaskType.SECURITY_AUDIT,
            description="Dependency vulnerability scan",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={"action": "check_dependencies", "dependencies": dependencies},
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)

    async def review_auth(self, code: str) -> AgentOutput:
        """Review authentication/authorization implementation."""
        task = Task(
            id=f"auth_{self.project_id}",
            type=TaskType.SECURITY_AUDIT,
            description="Authentication/authorization review",
            complexity=TaskComplexity.COMPLEX,
            project_id=self.project_id,
            context={"action": "review_auth", "existing_code": code},
            is_blocking=True,
            allow_batch=False,
        )
        return await self.execute(task)

    async def generate_security_checklist(self, app_type: str = "SaaS web application") -> AgentOutput:
        """Generate pre-deployment security checklist (batch eligible)."""
        task = Task(
            id=f"checklist_{self.project_id}",
            type=TaskType.SECURITY_AUDIT,
            description="Pre-deployment security checklist",
            complexity=TaskComplexity.SIMPLE,
            project_id=self.project_id,
            context={"action": "generate_security_checklist", "app_type": app_type},
            is_blocking=False,
            allow_batch=True,
        )
        return await self.execute(task)
