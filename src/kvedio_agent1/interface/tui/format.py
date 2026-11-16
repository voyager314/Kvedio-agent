"""How stored values read on screen.

Every function here is pure and takes values as they are stored, so the wording
lives in one place and the panels stay free of formatting branches.
"""

from __future__ import annotations

from ...core.config import ModelBinding, Role
from ...core.domain import JobStatus, TaskStatus

_ROLE_LABELS = {
    Role.SUPERVISOR: "监督",
    Role.ASR: "转录",
    Role.VIDEO: "视觉",
    Role.EMBEDDING: "嵌入",
}

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


def format_container(name: str | None) -> str:
    """The demuxer family behind a container string.

    The decoder reports every alias of a format as one comma-separated name; the
    first entry is the family a reader recognises.
    """
    if not name:
        return "未知"
    return name.split(",")[0]


def role_label(role: Role) -> str:
    return _ROLE_LABELS[role]


def job_status_label(status: JobStatus) -> str:
    return _JOB_LABELS[status]


def task_status_label(status: TaskStatus) -> str:
    return _TASK_LABELS[status]


def agent_label(agent: str) -> str:
    """Display name of a task's agent; an unknown agent is shown as declared."""
    return _AGENT_LABELS.get(agent, agent)


def model_label(binding: ModelBinding | None, configured: bool) -> str:
    """The model behind a role and whether it can call out.

    Names the model only: no part of a key, and not the environment variable that
    may hold one.
    """
    if binding is None:
        return "未配置"
    return binding.model if configured else f"{binding.model}(未配置)"
