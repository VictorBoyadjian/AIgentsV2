"""Tests for the CacheManager — message building and cache control markers."""

from __future__ import annotations

import pytest

from core.cache_manager import CacheManager, AGENT_SYSTEM_PROMPTS
from core.config import AgentRole


class TestBuildCachedMessages:
    """Test Anthropic cache_control message formatting."""

    @pytest.fixture
    def cache_manager(self) -> CacheManager:
        return CacheManager(redis_url="redis://localhost:6379/0")

    def test_system_prompt_has_cache_control(self, cache_manager: CacheManager) -> None:
        """System prompt should always have cache_control marker."""
        system_blocks, user_msgs = cache_manager.build_cached_messages(
            system_prompt="You are a test agent.",
            current_task="Do something.",
        )

        assert len(system_blocks) == 1
        assert system_blocks[0]["type"] == "text"
        assert system_blocks[0]["text"] == "You are a test agent."
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_current_task_has_no_cache_control(self, cache_manager: CacheManager) -> None:
        """Current task should NOT have cache_control (it's variable)."""
        _, user_msgs = cache_manager.build_cached_messages(
            system_prompt="System prompt.",
            current_task="Variable task content.",
        )

        assert len(user_msgs) == 1
        task_block = user_msgs[0]["content"][-1]
        assert task_block["text"] == "Variable task content."
        assert "cache_control" not in task_block

    def test_project_context_has_cache_control(self, cache_manager: CacheManager) -> None:
        """Project context should have cache_control marker."""
        _, user_msgs = cache_manager.build_cached_messages(
            system_prompt="System prompt.",
            project_context="Architecture document here.",
            current_task="Build feature X.",
        )

        content_blocks = user_msgs[0]["content"]
        context_block = content_blocks[0]
        assert "<project_context>" in context_block["text"]
        assert context_block["cache_control"] == {"type": "ephemeral"}

    def test_existing_code_has_cache_control(self, cache_manager: CacheManager) -> None:
        """Existing code should have cache_control marker."""
        _, user_msgs = cache_manager.build_cached_messages(
            system_prompt="System prompt.",
            existing_code="def hello(): pass",
            current_task="Fix the function.",
        )

        content_blocks = user_msgs[0]["content"]
        code_block = content_blocks[0]
        assert "<existing_code>" in code_block["text"]
        assert code_block["cache_control"] == {"type": "ephemeral"}

    def test_full_message_structure(self, cache_manager: CacheManager) -> None:
        """Verify complete message structure with all components."""
        system_blocks, user_msgs = cache_manager.build_cached_messages(
            system_prompt="System prompt.",
            project_context="Project context.",
            existing_code="def foo(): pass",
            current_task="Implement bar().",
        )

        # System should have 1 cached block
        assert len(system_blocks) == 1

        # User message should have 3 blocks: context (cached), code (cached), task (not cached)
        content = user_msgs[0]["content"]
        assert len(content) == 3

        # First two have cache_control
        assert "cache_control" in content[0]
        assert "cache_control" in content[1]

        # Last one (task) does NOT
        assert "cache_control" not in content[2]


class TestBuildLiteLLMMessages:
    """Test LiteLLM-formatted messages with cache control."""

    @pytest.fixture
    def cache_manager(self) -> CacheManager:
        return CacheManager(redis_url="redis://localhost:6379/0")

    def test_litellm_format_system_message(self, cache_manager: CacheManager) -> None:
        """System message should use content blocks format for LiteLLM."""
        messages = cache_manager.build_cached_messages_litellm(
            system_prompt="You are helpful.",
            current_task="Do a thing.",
        )

        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_litellm_format_with_context(self, cache_manager: CacheManager) -> None:
        """Messages with context should have proper ordering."""
        messages = cache_manager.build_cached_messages_litellm(
            system_prompt="System.",
            project_context="Context.",
            current_task="Task.",
        )

        assert len(messages) == 2  # system + user
        user_content = messages[1]["content"]
        assert len(user_content) == 2  # context + task
        assert "cache_control" in user_content[0]
        assert "cache_control" not in user_content[1]


class TestSystemPrompts:
    """Test system prompt retrieval."""

    @pytest.fixture
    def cache_manager(self) -> CacheManager:
        return CacheManager(redis_url="redis://localhost:6379/0")

    def test_all_roles_have_system_prompts(self) -> None:
        """Every agent role should have a system prompt defined."""
        for role in [AgentRole.ARCHITECT, AgentRole.PM, AgentRole.DEV,
                     AgentRole.QA, AgentRole.SECURITY, AgentRole.DEVOPS,
                     AgentRole.RESEARCH]:
            assert role.value in AGENT_SYSTEM_PROMPTS or role in AGENT_SYSTEM_PROMPTS

    def test_get_system_prompt_by_string(self, cache_manager: CacheManager) -> None:
        """Should work with string role names."""
        prompt = cache_manager.get_system_prompt("dev")
        assert "Full-Stack" in prompt or "Senior" in prompt

    def test_get_system_prompt_by_enum(self, cache_manager: CacheManager) -> None:
        """Should work with AgentRole enum."""
        prompt = cache_manager.get_system_prompt(AgentRole.ARCHITECT)
        assert "Architecte" in prompt or "architect" in prompt.lower()
