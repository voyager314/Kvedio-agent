"""The legal status changes of a task.

The table is the only statement of the state machine. Every write path checks it,
so an impossible change fails where it is attempted instead of leaving a record
whose status contradicts its own timestamps.
"""

from __future__ import annotations

from ..domain import TaskStatus


class IllegalTransition(Exception):
    """A status change the machine does not permit."""


_ALLOWED: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.READY, TaskStatus.CANCELED, TaskStatus.SKIPPED}),
    TaskStatus.READY: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELED, TaskStatus.SKIPPED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}),
    TaskStatus.FAILED: frozenset({TaskStatus.READY, TaskStatus.CANCELED}),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.CANCELED: frozenset(),
    TaskStatus.SKIPPED: frozenset(),
}


def check_transition(source: TaskStatus, target: TaskStatus) -> None:
    """Raise ``IllegalTransition`` unless ``source`` may become ``target``."""
    if target not in _ALLOWED[source]:
        raise IllegalTransition(f"任务状态不能从 {source.value} 变为 {target.value}")
