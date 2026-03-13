"""
E2B-based code sandbox for safe code execution and testing.

Provides isolated execution environments for running generated code,
tests, and build processes without affecting the host system.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class SandboxResult:
    """Result of code execution in the sandbox."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float = 0.0
    files_created: list[str] | None = None


class CodeSandbox:
    """
    E2B-based code sandbox for isolated code execution.

    Runs generated code, tests, and build processes in isolated
    cloud sandboxes via E2B. Each execution gets a fresh environment.
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.tools.e2b_api_key

    async def execute_python(
        self,
        code: str,
        packages: list[str] | None = None,
        timeout_seconds: int = 120,
    ) -> SandboxResult:
        """
        Execute Python code in an isolated sandbox.

        Args:
            code: Python code to execute.
            packages: List of pip packages to install before execution.
            timeout_seconds: Maximum execution time.

        Returns:
            SandboxResult with stdout, stderr, and exit code.
        """
        try:
            from e2b_code_interpreter import AsyncSandbox

            async with AsyncSandbox(api_key=self._api_key) as sandbox:
                # Install packages if needed
                if packages:
                    pkg_list = " ".join(packages)
                    await sandbox.commands.run(
                        f"pip install {pkg_list}",
                        timeout=60,
                    )

                # Execute code
                execution = await sandbox.run_code(code, timeout=timeout_seconds)

                stdout_parts: list[str] = []
                stderr_parts: list[str] = []

                for log in execution.logs.stdout:
                    stdout_parts.append(log)
                for log in execution.logs.stderr:
                    stderr_parts.append(log)

                success = execution.error is None

                result = SandboxResult(
                    success=success,
                    stdout="\n".join(stdout_parts),
                    stderr="\n".join(stderr_parts) + (f"\n{execution.error}" if execution.error else ""),
                    exit_code=0 if success else 1,
                )

                logger.info(
                    "code_sandbox.python_executed",
                    success=success,
                    stdout_len=len(result.stdout),
                )
                return result

        except ImportError:
            logger.error("code_sandbox.e2b_not_installed")
            return SandboxResult(
                success=False,
                stdout="",
                stderr="e2b_code_interpreter package not installed",
                exit_code=1,
            )
        except Exception as exc:
            logger.error("code_sandbox.execution_failed", error=str(exc))
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Sandbox execution failed: {exc}",
                exit_code=1,
            )

    async def execute_shell(
        self,
        commands: list[str],
        timeout_seconds: int = 120,
    ) -> SandboxResult:
        """Execute shell commands in an isolated sandbox."""
        try:
            from e2b_code_interpreter import AsyncSandbox

            async with AsyncSandbox(api_key=self._api_key) as sandbox:
                all_stdout: list[str] = []
                all_stderr: list[str] = []
                final_exit_code = 0

                for cmd in commands:
                    result = await sandbox.commands.run(cmd, timeout=timeout_seconds)
                    if result.stdout:
                        all_stdout.append(result.stdout)
                    if result.stderr:
                        all_stderr.append(result.stderr)
                    if result.exit_code != 0:
                        final_exit_code = result.exit_code
                        break

                return SandboxResult(
                    success=final_exit_code == 0,
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    exit_code=final_exit_code,
                )

        except ImportError:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="e2b_code_interpreter package not installed",
                exit_code=1,
            )
        except Exception as exc:
            logger.error("code_sandbox.shell_failed", error=str(exc))
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Sandbox shell execution failed: {exc}",
                exit_code=1,
            )

    async def run_tests(
        self,
        test_code: str,
        source_code: dict[str, str],
        packages: list[str] | None = None,
    ) -> SandboxResult:
        """
        Run pytest tests in the sandbox.

        Args:
            test_code: pytest test file content.
            source_code: Dict of {filename: content} for source files.
            packages: Additional packages to install.
        """
        try:
            from e2b_code_interpreter import AsyncSandbox

            async with AsyncSandbox(api_key=self._api_key) as sandbox:
                # Install packages
                base_packages = ["pytest", "pytest-asyncio"] + (packages or [])
                await sandbox.commands.run(
                    f"pip install {' '.join(base_packages)}",
                    timeout=60,
                )

                # Write source files
                for filename, content in source_code.items():
                    await sandbox.files.write(f"/home/user/{filename}", content)

                # Write test file
                await sandbox.files.write("/home/user/test_code.py", test_code)

                # Run tests
                result = await sandbox.commands.run(
                    "cd /home/user && python -m pytest test_code.py -v --tb=short",
                    timeout=120,
                )

                return SandboxResult(
                    success=result.exit_code == 0,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    exit_code=result.exit_code,
                )

        except ImportError:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="e2b_code_interpreter package not installed",
                exit_code=1,
            )
        except Exception as exc:
            logger.error("code_sandbox.tests_failed", error=str(exc))
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Test execution failed: {exc}",
                exit_code=1,
            )
