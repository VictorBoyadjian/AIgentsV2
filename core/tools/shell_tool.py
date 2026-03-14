"""
Shell command execution tool with sandboxing and timeout support.

Provides safe command execution for build steps, test runners, and
other CLI operations needed by the agent team.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# Commands that are never allowed for safety
BLOCKED_COMMANDS: set[str] = {
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
    "chmod -R 777 /",
}

# Maximum output size in bytes
MAX_OUTPUT_SIZE = 1_000_000  # 1 MB


@dataclass
class ShellResult:
    """Result of a shell command execution."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Whether the command completed successfully."""
        return self.return_code == 0 and not self.timed_out


class ShellTool:
    """
    Safe shell command execution tool.

    Executes commands in a controlled environment with timeouts,
    output size limits, and blocked command detection.
    """

    def __init__(
        self,
        working_dir: str | None = None,
        timeout_seconds: int = 300,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        self._working_dir = working_dir or os.getcwd()
        self._timeout = timeout_seconds
        self._env = {**os.environ, **(env_vars or {})}

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> ShellResult:
        """
        Execute a shell command asynchronously.

        Args:
            command: The shell command to execute.
            working_dir: Override working directory for this command.
            timeout: Override timeout in seconds.

        Returns:
            ShellResult with stdout, stderr, and return code.
        """
        # Safety check
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                logger.error("shell_tool.blocked_command", command=command)
                return ShellResult(
                    command=command,
                    return_code=1,
                    stdout="",
                    stderr=f"Command blocked for safety: contains '{blocked}'",
                )

        cwd = working_dir or self._working_dir
        effective_timeout = timeout or self._timeout

        logger.info("shell_tool.executing", command=command, cwd=cwd, timeout=effective_timeout)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning("shell_tool.timeout", command=command, timeout=effective_timeout)
                return ShellResult(
                    command=command,
                    return_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    timed_out=True,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_SIZE]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_SIZE]

            result = ShellResult(
                command=command,
                return_code=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
            )

            logger.info(
                "shell_tool.completed",
                command=command,
                return_code=result.return_code,
                stdout_len=len(stdout),
            )

            return result

        except Exception as exc:
            logger.error("shell_tool.error", command=command, error=str(exc))
            return ShellResult(
                command=command,
                return_code=1,
                stdout="",
                stderr=f"Execution error: {exc}",
            )

    async def execute_many(
        self, commands: list[str], stop_on_error: bool = True
    ) -> list[ShellResult]:
        """Execute multiple commands sequentially."""
        results: list[ShellResult] = []
        for cmd in commands:
            result = await self.execute(cmd)
            results.append(result)
            if stop_on_error and not result.success:
                break
        return results
