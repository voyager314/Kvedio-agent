import pytest

from kvedio_agent1.core.config import ModelBinding, Role
from kvedio_agent1.core.domain import JobStatus, TaskStatus
from kvedio_agent1.interface.tui.format import (
    agent_label,
    format_container,
    format_duration,
    format_progress,
    format_size,
    job_status_label,
    model_label,
    role_label,
    task_status_label,
)


def binding(model: str = "some-model", key: str = "sk-live-zzz") -> ModelBinding:
    return ModelBinding(
        model=model, url="https://example.invalid/v1/chat/completions", api_key=key
    )


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [(0, "00:00:00"), (999, "00:00:00"), (3_725_000, "01:02:05"), (-5, "00:00:00")],
)
def test_a_duration_reads_as_hours_minutes_seconds(milliseconds, expected):
    assert format_duration(milliseconds) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, "0%"), (0.615, "62%"), (1.0, "100%"), (1.4, "100%"), (-1, "0%")]
)
def test_progress_is_rendered_as_whole_percent(value, expected):
    assert format_progress(value) == expected


def test_size_reads_in_megabytes_and_drops_to_kilobytes():
    assert format_size(2 * 1024 * 1024) == "2.0 MB"
    assert format_size(20 * 1024) == "20 KB"


def test_a_container_reads_as_its_demuxer_family():
    assert format_container("mov,mp4,m4a,3gp,3g2,mj2") == "mov"
    assert format_container(None) == "未知"


def test_a_usable_model_shows_its_name_only():
    label = model_label(binding(), configured=True)
    assert label == "some-model"
    assert "sk-live" not in label
    assert "zzz" not in label


def test_an_unusable_model_says_so():
    assert model_label(binding(), configured=False) == "some-model(未配置)"


def test_a_role_without_a_model_says_so():
    assert model_label(None, configured=False) == "未配置"


def test_labels_are_words_not_enum_values():
    assert role_label(Role.SUPERVISOR) == "监督"
    assert job_status_label(JobStatus.PARTIAL) == "部分完成"
    assert task_status_label(TaskStatus.SKIPPED) == "跳过"
    assert agent_label("media") == "媒体"
    assert agent_label("unknown") == "unknown"
