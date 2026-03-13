"""
GitHub integration tool for repository management.

Provides operations for creating repos, managing branches, commits,
pull requests, and issues via the PyGithub library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from github import Github, GithubException
from github.Repository import Repository

from core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class GitHubResult:
    """Result of a GitHub operation."""

    success: bool
    message: str
    data: dict[str, Any] | None = None
    url: str | None = None


class GitHubTool:
    """
    GitHub operations tool for managing repositories, branches, PRs, and issues.

    Uses PyGithub for all GitHub API interactions. Requires a valid
    GitHub token with appropriate permissions.
    """

    def __init__(self, token: str | None = None, org: str | None = None) -> None:
        settings = get_settings()
        self._token = token or settings.tools.github_token
        self._org = org or settings.tools.github_org
        self._client = Github(self._token) if self._token else None

    def _get_client(self) -> Github:
        """Get authenticated GitHub client."""
        if not self._client:
            raise RuntimeError("GitHub token not configured")
        return self._client

    async def create_repository(
        self,
        name: str,
        description: str = "",
        private: bool = True,
        auto_init: bool = True,
    ) -> GitHubResult:
        """Create a new GitHub repository."""
        try:
            client = self._get_client()
            if self._org:
                org = client.get_organization(self._org)
                repo = org.create_repo(
                    name=name,
                    description=description,
                    private=private,
                    auto_init=auto_init,
                )
            else:
                user = client.get_user()
                repo = user.create_repo(
                    name=name,
                    description=description,
                    private=private,
                    auto_init=auto_init,
                )

            logger.info("github_tool.repo_created", name=name, url=repo.html_url)
            return GitHubResult(
                success=True,
                message=f"Repository '{name}' created successfully",
                data={"full_name": repo.full_name},
                url=repo.html_url,
            )
        except GithubException as exc:
            logger.error("github_tool.create_repo_failed", name=name, error=str(exc))
            return GitHubResult(success=False, message=f"Failed to create repo: {exc}")

    async def create_branch(
        self, repo_name: str, branch_name: str, from_branch: str = "main"
    ) -> GitHubResult:
        """Create a new branch from an existing branch."""
        try:
            repo = self._get_repo(repo_name)
            source = repo.get_branch(from_branch)
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=source.commit.sha,
            )
            logger.info("github_tool.branch_created", branch=branch_name)
            return GitHubResult(
                success=True,
                message=f"Branch '{branch_name}' created from '{from_branch}'",
            )
        except GithubException as exc:
            logger.error("github_tool.create_branch_failed", error=str(exc))
            return GitHubResult(success=False, message=f"Failed to create branch: {exc}")

    async def commit_files(
        self,
        repo_name: str,
        branch: str,
        files: dict[str, str],
        commit_message: str,
    ) -> GitHubResult:
        """Commit multiple files to a branch."""
        try:
            repo = self._get_repo(repo_name)
            branch_ref = repo.get_branch(branch)
            base_tree = repo.get_git_tree(sha=branch_ref.commit.sha)

            tree_elements = []
            from github import InputGitTreeElement

            for path, content in files.items():
                tree_elements.append(
                    InputGitTreeElement(path=path, mode="100644", type="blob", content=content)
                )

            new_tree = repo.create_git_tree(tree_elements, base_tree)
            new_commit = repo.create_git_commit(
                message=commit_message,
                tree=new_tree,
                parents=[repo.get_git_commit(branch_ref.commit.sha)],
            )
            ref = repo.get_git_ref(f"heads/{branch}")
            ref.edit(sha=new_commit.sha)

            logger.info(
                "github_tool.files_committed",
                repo=repo_name,
                branch=branch,
                file_count=len(files),
            )
            return GitHubResult(
                success=True,
                message=f"Committed {len(files)} files to '{branch}'",
                data={"sha": new_commit.sha},
            )
        except GithubException as exc:
            logger.error("github_tool.commit_failed", error=str(exc))
            return GitHubResult(success=False, message=f"Failed to commit: {exc}")

    async def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> GitHubResult:
        """Create a pull request."""
        try:
            repo = self._get_repo(repo_name)
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head,
                base=base,
            )
            logger.info("github_tool.pr_created", number=pr.number, title=title)
            return GitHubResult(
                success=True,
                message=f"PR #{pr.number} created: {title}",
                data={"number": pr.number},
                url=pr.html_url,
            )
        except GithubException as exc:
            logger.error("github_tool.create_pr_failed", error=str(exc))
            return GitHubResult(success=False, message=f"Failed to create PR: {exc}")

    async def create_issue(
        self,
        repo_name: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> GitHubResult:
        """Create an issue in the repository."""
        try:
            repo = self._get_repo(repo_name)
            issue = repo.create_issue(
                title=title,
                body=body,
                labels=labels or [],
            )
            logger.info("github_tool.issue_created", number=issue.number)
            return GitHubResult(
                success=True,
                message=f"Issue #{issue.number} created: {title}",
                data={"number": issue.number},
                url=issue.html_url,
            )
        except GithubException as exc:
            logger.error("github_tool.create_issue_failed", error=str(exc))
            return GitHubResult(success=False, message=f"Failed to create issue: {exc}")

    async def get_file_content(self, repo_name: str, path: str, branch: str = "main") -> str | None:
        """Read a file from the repository."""
        try:
            repo = self._get_repo(repo_name)
            content = repo.get_contents(path, ref=branch)
            if isinstance(content, list):
                return None
            return content.decoded_content.decode("utf-8")
        except GithubException:
            return None

    def _get_repo(self, repo_name: str) -> Repository:
        """Get repository by name (with org prefix if configured)."""
        client = self._get_client()
        full_name = f"{self._org}/{repo_name}" if self._org else repo_name
        return client.get_repo(full_name)
