"""State changes published by work and rendered by the interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class EventKind(StrEnum):
    """What happened, at the granularity the interface renders."""

    VIDEO_REGISTERED = "video.registered"
    JOB_QUEUED = "job.queued"
    JOB_STARTED = "job.started"
    JOB_SUCCEEDED = "job.succeeded"
    JOB_PARTIAL = "job.partial"
    JOB_FAILED = "job.failed"
    JOB_CANCELED = "job.canceled"
    TASK_READY = "task.ready"
    TASK_STARTED = "task.started"
    TASK_PROGRESS = "task.progress"
    TASK_SUCCEEDED = "task.succeeded"
    TASK_FAILED = "task.failed"
    TASK_CANCELED = "task.canceled"
    TASK_SKIPPED = "task.skipped"
    LOG_LINE = "log.line"
    ANSWER_DELTA = "answer.delta"


@dataclass(frozen=True)
class Event:
    """One published change, readable by every subscriber.

    ``payload`` carries detail the kind does not fix: progress as a 0.0 to 1.0
    float, a one-line message for the interface, a count, an elapsed duration.
    Subscribers render it and must not modify it.
    """

    kind: EventKind
    at: datetime  # UTC instant the change was recorded
    job_id: str | None = None
    task_id: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def stamp(
        cls,
        kind: EventKind,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> Event:
        """Build an event carrying the current UTC instant."""
        return cls(
            kind=kind,
            at=datetime.now(UTC),
            job_id=job_id,
            task_id=task_id,
            payload=payload or {},
        )
