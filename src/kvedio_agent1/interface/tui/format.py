"""How stored values read on screen.

Every function here is pure and takes values as they are stored, so the wording
lives in one place and the panels stay free of formatting branches.
"""

from __future__ import annotations

from ...core.config import Role, Selection
from ...core.domain import JobStatus, TaskStatus

_ROLE_LABELS = {Role.SUPERVISOR: "监督", Role.ASR: "转录", Role.VIDEO: "视觉"}

_JOB_LABELS = {
    JobStatus.QUEUED: "已排队",
    JobStatus.RUNNING: "运行中",
    JobStatus.SUCCEEDED: "完成",
    JobStatus.PARTIAL: "部分完成",
    JobStatus.FAILED: "失败",
    JobStatus.CANCELED: "已取消",
}

_TASK_LABELS = {
    TaskStatus.PENDING: "等待",
    TaskStatus.READY: "就绪",
    TaskStatus.RUNNING: "运行中",
    TaskStatus.SUCCEEDED: "完成",
    TaskStatus.FAILED: "失败",
    TaskStatus.SKIPPED: "跳过",
    TaskStatus.CANCELED: "已取消",
}

_AGENT_LABELS = {"media": "媒体", "asr": "转录", "video": "视觉", "supervisor": "监督"}


def format_duration(milliseconds: int) -> str:
    """``HH:MM:SS`` from a duration in milliseconds; a negative value reads as zero."""
    total = max(0, milliseconds) // 1000
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_progress(value: float) -> str:
    """A 0.0 to 1.0 fraction as whole percent, clamped at both ends."""
    return f"{min(1.0, max(0.0, value)) * 100:.0f}%"


def format_size(size_bytes: int) -> str:
    """A file size in MB, or in KB below one megabyte so a short clip does not read as 0.0."""
    megabytes = size_bytes / (1024 * 1024)
    if megabytes >= 1:
        return f"{megabytes:.1f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def role_label(role: Role) -> str:
    return _ROLE_LABELS[role]


def job_status_label(status: JobStatus) -> str:
    return _JOB_LABELS[status]


def task_status_label(status: TaskStatus) -> str:
    return _TASK_LABELS[status]


def agent_label(agent: str) -> str:
    """Display name of a task's agent; an unknown agent is shown as declared."""
    return _AGENT_LABELS.get(agent, agent)


def binding_label(selection: Selection, configured: bool) -> str:
    """Name of a role's active binding and whether it can call out.

    Reports only the binding name and a state word: no part of a key, and not the
    environment variable that may hold one.
    """
    if selection.binding is None:
        return "未绑定"
    return selection.binding.name if configured else f"{selection.binding.name}(未配置)"
