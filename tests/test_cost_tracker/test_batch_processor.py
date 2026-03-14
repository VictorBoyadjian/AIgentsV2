"""Tests for the BatchProcessor — eligibility logic and data structures."""

from __future__ import annotations

import pytest

from core.batch_processor import (
    BatchProcessor,
    BATCH_ELIGIBLE_TASKS,
    REALTIME_ONLY_TASKS,
    BatchStatus,
    BatchTask,
)


class TestBatchEligibility:
    """Test the should_use_batch decision logic."""

    @pytest.fixture
    def processor(self) -> BatchProcessor:
        return BatchProcessor.__new__(BatchProcessor)

    @pytest.mark.parametrize("task_type", list(BATCH_ELIGIBLE_TASKS))
    def test_eligible_tasks_use_batch(self, processor: BatchProcessor, task_type: str) -> None:
        """All tasks in BATCH_ELIGIBLE_TASKS should route to batch."""
        assert processor.should_use_batch(task_type, is_blocking=False) is True

    @pytest.mark.parametrize("task_type", list(REALTIME_ONLY_TASKS))
    def test_realtime_tasks_skip_batch(self, processor: BatchProcessor, task_type: str) -> None:
        """All REALTIME_ONLY_TASKS should never use batch."""
        assert processor.should_use_batch(task_type, is_blocking=False) is False

    def test_blocking_tasks_never_batch(self, processor: BatchProcessor) -> None:
        """Even batch-eligible tasks should be real-time if blocking."""
        for task_type in BATCH_ELIGIBLE_TASKS:
            assert processor.should_use_batch(task_type, is_blocking=True) is False

    def test_unknown_tasks_default_realtime(self, processor: BatchProcessor) -> None:
        """Unknown task types should default to real-time."""
        assert processor.should_use_batch("unknown_task_type") is False


class TestBatchEligibleCoverage:
    """Verify batch-eligible categories match the spec."""

    def test_test_generation_is_eligible(self) -> None:
        assert "generate_tests" in BATCH_ELIGIBLE_TASKS

    def test_documentation_is_eligible(self) -> None:
        assert "generate_documentation" in BATCH_ELIGIBLE_TASKS

    def test_security_checklist_is_eligible(self) -> None:
        assert "generate_security_checklist" in BATCH_ELIGIBLE_TASKS

    def test_cicd_is_eligible(self) -> None:
        assert "generate_cicd_pipeline" in BATCH_ELIGIBLE_TASKS

    def test_iac_is_eligible(self) -> None:
        assert "generate_iac" in BATCH_ELIGIBLE_TASKS

    def test_code_generation_is_realtime(self) -> None:
        assert "generate_code_feature" in REALTIME_ONLY_TASKS

    def test_deploy_is_realtime(self) -> None:
        assert "deploy" in REALTIME_ONLY_TASKS

    def test_run_tests_is_realtime(self) -> None:
        assert "run_tests" in REALTIME_ONLY_TASKS


class TestBatchStatus:
    """Test BatchStatus enum values."""

    def test_all_statuses_defined(self) -> None:
        expected = {"pending", "in_progress", "completed", "failed", "expired", "cancelling", "cancelled"}
        actual = {s.value for s in BatchStatus}
        assert expected == actual


class TestBatchTask:
    """Test BatchTask dataclass defaults."""

    def test_default_values(self) -> None:
        task = BatchTask(
            custom_id="test-1",
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert task.max_tokens == 4096
        assert task.temperature == 0.7
        assert task.system is None
        assert task.metadata == {}
