"""One analysis request and the task graph that satisfies it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class JobStatus(StrEnum):
    """Lifecycle of one request over one video."""

    QUEUED = "queued"  # recorded, no node has started
    RUNNING = "running"  # at least one node has started
    SUCCEEDED = "succeeded"  # every node succeeded
    PARTIAL = "partial"  # some nodes succeeded, some did not
    FAILED = "failed"  # no node succeeded
    CANCELED = "canceled"  # stopped on request before finishing


class TaskStatus(StrEnum):
    """Lifecycle of one node of the graph."""

    PENDING = "pending"  # dependencies not resolved yet
    READY = "ready"  # dependencies satisfied, waiting for a worker
    RUNNING = "running"  # taken by a worker
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # exhausted its attempts
    CANCELED = "canceled"
    SKIPPED = "skipped"  # dependencies made it impossible to run


class DependencyMode(StrEnum):
    """How a node reads the outcome of the nodes it depends on."""

    ALL = "all"  # runs only when every dependency succeeded
    ANY = "any"  # runs as soon as one dependency succeeded
    PARTIAL = "partial"  # runs once every dependency is terminal, whatever the outcome


TERMINAL: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED, TaskStatus.SKIPPED}
)


@dataclass(frozen=True)
class NewTask:
    """A node to persist before its job starts; the executable half stays in code.

    ``id`` is unique within the job and referenced by other nodes' ``depends_on``.
    ``agent`` names the capability that will run it, ``action`` the operation.
    ``max_attempts`` counts total runs, so ``1`` means no retry.
    """

    id: str
    agent: str
    action: str
    depends_on: tuple[str, ...] = ()
    dependency_mode: DependencyMode = DependencyMode.ALL
    max_attempts: int = 1


@dataclass(frozen=True)
class TaskRecord:
    """One persisted node of a job's graph."""

    id: str  # unique within the job
    job_id: str
    agent: str  # permitted values: media, asr, video, supervisor
    action: str
    status: TaskStatus
    depends_on: tuple[str, ...]  # ids of nodes in the same job
    dependency_mode: DependencyMode
    attempt: int  # completed runs, 0 before the first one starts
    max_attempts: int  # total runs allowed; 1 means no retry
    progress: float  # 0.0 to 1.0, monotonic within an attempt
    started_at: datetime | None  # first start of the current attempt
    finished_at: datetime | None  # when the node reached a terminal status
    error: str | None  # last failure message; None while running or after success


@dataclass(frozen=True)
class JobRecord:
    """One analysis request: which video, what was asked, and how far it got."""

    id: str
    video_id: int
    question: str  # the user's own wording, never rewritten
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None  # set when the job ends as failed or partial
    agents: tuple[str, ...] = ("asr", "video")  # branches this request asked to run
