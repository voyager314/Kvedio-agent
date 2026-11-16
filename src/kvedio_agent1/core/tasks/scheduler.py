"""Execution of one job's task graph on a bounded pool.

Owns readiness evaluation, retry within each node's attempt ceiling, wall-clock
budgets, and the job's final status. A node runs only once its dependency mode is
satisfied, and a transition is written to the store before it is published, so an
observer never sees a state the store does not hold.

A budget is a deadline, not a preemption: the scheduler marks the context expired
and the stage raises at its next progress or check point. A stage blocked inside a
single call cannot be stopped from another thread, so the budget bounds stages
that make progress and the watchdog names the ones that do not.

Does not decide what a handler does with its input, and does not resume work that
an earlier process left behind.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import UTC, datetime

from ..domain import (
    TERMINAL,
    DependencyMode,
    Event,
    EventKind,
    JobRecord,
    JobStatus,
    TaskRecord,
    TaskStatus,
)
from ..storage import Storage
from .bus import EventBus
from .cancellation import Cancellation, Cancelled
from .state import check_transition
from .timeout import TaskTimeout

Handler = Callable[["TaskContext"], None]
SKIPPED_REASON = "依赖未成功，已跳过"
WATCHDOG_SECONDS = 0.5

_JOB_EVENT: dict[JobStatus, EventKind] = {
    JobStatus.SUCCEEDED: EventKind.JOB_SUCCEEDED,
    JobStatus.PARTIAL: EventKind.JOB_PARTIAL,
    JobStatus.FAILED: EventKind.JOB_FAILED,
    JobStatus.CANCELED: EventKind.JOB_CANCELED,
}


class TaskContext:
    """What a handler receives: the node it runs, the store, and a progress report.

    ``report`` clamps to 0.0..1.0 and never moves backwards within a run, so a
    panel cannot show a task regressing while it is still working. It also checks
    the node's budget before doing any work: a stage that has spent its window
    fails where it next reports, not after it finishes.

    ``expire`` is called by the scheduler's watchdog, never by a handler. A stage
    observes it through ``check`` or its next ``report``.
    """

    def __init__(
        self,
        job: JobRecord,
        task: TaskRecord,
        store: Storage,
        bus: EventBus,
        cancellation: Cancellation | None = None,
        deadline: float | None = None,
        budget_seconds: float = 0.0,
    ) -> None:
        self._job = job
        self._task = task
        self._store = store
        self._bus = bus
        self._cancellation = cancellation or Cancellation()
        self._deadline = deadline
        self._budget_seconds = budget_seconds
        self._expired = threading.Event()

    @property
    def job(self) -> JobRecord:
        return self._job

    @property
    def task(self) -> TaskRecord:
        return self._task

    @property
    def cancelled(self) -> bool:
        """Whether the interface asked the run to stop; stages check it between units."""
        return self._cancellation.cancelled

    @property
    def deadline(self) -> float | None:
        """The monotonic instant the budget runs out, or ``None`` when unbudgeted."""
        return self._deadline

    @property
    def budget_seconds(self) -> float:
        """The window the node was given, for the failure message and the watchdog."""
        return self._budget_seconds

    @property
    def expired(self) -> bool:
        """Whether the watchdog has marked this node as past its budget."""
        return self._expired.is_set()

    def expire(self) -> None:
        """Mark the node as past its budget; the next check raises ``TaskTimeout``."""
        self._expired.set()

    def raise_if_cancelled(self) -> None:
        """Abort the calling stage when cancellation has been requested."""
        self._cancellation.raise_if_cancelled()

    def check(self) -> None:
        """Stop the stage when cancellation arrived or the budget is spent."""
        self._cancellation.raise_if_cancelled()
        if self._expired.is_set() or (
            self._deadline is not None and time.monotonic() >= self._deadline
        ):
            raise TaskTimeout(self._task.action, self._budget_seconds)

    def report(self, progress: float, message: str | None = None) -> None:
        """Persist progress and, when given, publish one line for the interface."""
        self.check()
        value = min(1.0, max(self._task.progress, float(progress)))
        self._task = replace(self._task, progress=value)
        self._store.save_task(self._task)
        payload: dict[str, object] = {"progress": value}
        if message:
            payload["message"] = message
        self._bus.publish(
            Event.stamp(
                EventKind.TASK_PROGRESS,
                job_id=self._job.id,
                task_id=self._task.id,
                payload=payload,
            )
        )


class Scheduler:
    """Runs a job's persisted graph and reports the job's final state."""

    def __init__(self, store: Storage, bus: EventBus, max_workers: int = 4) -> None:
        self._store = store
        self._bus = bus
        self._max_workers = max(1, max_workers)

    def run(
        self,
        job: JobRecord,
        handlers: Mapping[str, Handler],
        cancellation: Cancellation | None = None,
        budget: Callable[[TaskRecord], float | None] | None = None,
    ) -> JobRecord:
        """Execute every node of ``job`` and return it in its final state.

        ``handlers`` maps node id to its executable half and must cover the whole
        graph: a missing entry is a wiring error and raises before any node
        starts, rather than leaving a job that can only ever be failed.

        ``budget`` gives one node the seconds it may run; ``None`` leaves the node
        unbudgeted. The loop checks deadlines twice a second, marks an expired
        context, and logs the overrun once; the stage itself raises at its next
        progress or check point.
        """
        tasks = {task.id: task for task in self._store.tasks_for_job(job.id)}
        if not tasks:
            raise ValueError(f"作业没有任务：{job.id}")
        unhandled = sorted(set(tasks) - set(handlers))
        if unhandled:
            raise ValueError(f"缺少任务处理器：{', '.join(unhandled)}")

        signal = cancellation or Cancellation()
        running = replace(job, status=JobStatus.RUNNING, started_at=job.started_at or _now())
        self._store.save_job(running)
        self._publish_job(EventKind.JOB_STARTED, running)

        with ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="kvedio") as pool:
            futures: dict[Future[None], str] = {}
            contexts: dict[str, TaskContext] = {}
            while True:
                if signal.cancelled:
                    self._cancel_queued(tasks)
                else:
                    self._advance(tasks)
                    for identifier, task in tuple(tasks.items()):
                        if task.status is not TaskStatus.READY:
                            continue
                        started = replace(
                            task,
                            status=TaskStatus.RUNNING,
                            attempt=task.attempt + 1,
                            progress=0.0,
                            started_at=_now(),
                            error=None,
                        )
                        self._transition(task, started, EventKind.TASK_STARTED)
                        tasks[identifier] = started
                        handler = handlers[identifier]
                        seconds = budget(started) if budget is not None else None
                        context = TaskContext(
                            running,
                            started,
                            self._store,
                            self._bus,
                            signal,
                            deadline=time.monotonic() + seconds if seconds is not None else None,
                            budget_seconds=seconds or 0.0,
                        )
                        contexts[identifier] = context
                        futures[pool.submit(self._execute, context, handler)] = identifier

                if not futures:
                    break
                self._expire(contexts)
                done, _ = wait(
                    tuple(futures), timeout=WATCHDOG_SECONDS, return_when=FIRST_COMPLETED
                )
                if not done:
                    continue
                for future in done:
                    identifier = futures.pop(future)
                    contexts.pop(identifier, None)
                    tasks[identifier] = self._settle(tasks[identifier], future)

        return self._finish(running, tasks, signal)

    def _expire(self, contexts: Mapping[str, TaskContext]) -> None:
        """Mark every running node whose budget has run out and say so once."""
        moment = time.monotonic()
        for identifier, context in contexts.items():
            if context.expired or context.deadline is None or moment < context.deadline:
                continue
            context.expire()
            self._bus.publish(
                Event.stamp(
                    EventKind.LOG_LINE,
                    job_id=context.job.id,
                    task_id=identifier,
                    payload={
                        "message": (
                            f"阶段超时：{context.task.action} 超过预算 "
                            f"{context.budget_seconds:.0f} 秒，等待其在检查点停止"
                        )
                    },
                )
            )

    def _advance(self, tasks: dict[str, TaskRecord]) -> None:
        """Resolve every node whose dependencies settled, repeating until nothing changes."""
        moved = True
        while moved:
            moved = False
            for identifier, task in tuple(tasks.items()):
                if task.status is not TaskStatus.PENDING:
                    continue
                dependencies = [tasks[name] for name in task.depends_on]
                target = _next_status(task, dependencies)
                if target is TaskStatus.READY:
                    ready = replace(task, status=TaskStatus.READY)
                    self._transition(task, ready, EventKind.TASK_READY)
                    tasks[identifier] = ready
                elif target is TaskStatus.SKIPPED:
                    skipped = replace(
                        task, status=TaskStatus.SKIPPED, finished_at=_now(), error=SKIPPED_REASON
                    )
                    self._transition(task, skipped, EventKind.TASK_SKIPPED)
                    tasks[identifier] = skipped
                else:
                    continue
                moved = True

    def _cancel_queued(self, tasks: dict[str, TaskRecord]) -> None:
        """Cancel every node that has not started; the running one stops itself."""
        for identifier, task in tuple(tasks.items()):
            if task.status not in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.FAILED):
                continue
            canceled = replace(
                task, status=TaskStatus.CANCELED, finished_at=_now(), error="已取消"
            )
            self._transition(task, canceled, EventKind.TASK_CANCELED)
            tasks[identifier] = canceled

    def _settle(self, task: TaskRecord, future: Future[None]) -> TaskRecord:
        """Record the outcome of one run, retrying while the attempt ceiling allows it."""
        failure = future.exception()
        if isinstance(failure, Cancelled):
            canceled = replace(
                task, status=TaskStatus.CANCELED, finished_at=_now(), error="已取消"
            )
            self._transition(task, canceled, EventKind.TASK_CANCELED)
            return canceled
        if failure is None:
            succeeded = replace(
                task,
                status=TaskStatus.SUCCEEDED,
                progress=1.0,
                finished_at=_now(),
                error=None,
            )
            self._transition(task, succeeded, EventKind.TASK_SUCCEEDED)
            return succeeded

        message = _describe(failure)
        failed = replace(task, status=TaskStatus.FAILED, finished_at=_now(), error=message)
        retrying = task.attempt < task.max_attempts
        payload: dict[str, object] = {"error": message}
        if retrying:
            payload["retrying"] = True
        self._transition(task, failed, EventKind.TASK_FAILED, payload)
        if not retrying:
            return failed
        ready = replace(failed, status=TaskStatus.READY, started_at=None, progress=0.0)
        self._transition(failed, ready, EventKind.TASK_READY)
        return ready

    def _finish(
        self, job: JobRecord, tasks: Mapping[str, TaskRecord], cancellation: Cancellation
    ) -> JobRecord:
        succeeded = sum(1 for task in tasks.values() if task.status is TaskStatus.SUCCEEDED)
        if succeeded == len(tasks):
            status = JobStatus.SUCCEEDED
        elif cancellation.cancelled:
            status = JobStatus.CANCELED
        elif succeeded == 0:
            status = JobStatus.FAILED
        else:
            status = JobStatus.PARTIAL
        error = None if status is JobStatus.SUCCEEDED else _first_error(tasks.values())
        finished = replace(job, status=status, finished_at=_now(), error=error)
        self._store.save_job(finished)
        self._publish_job(
            _JOB_EVENT[status], finished, {"error": error} if error is not None else None
        )
        return finished

    def _execute(self, context: TaskContext, handler: Handler) -> None:
        handler(context)

    def _transition(
        self,
        before: TaskRecord,
        after: TaskRecord,
        kind: EventKind,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        check_transition(before.status, after.status)
        self._store.save_task(after)
        self._publish_task(kind, after, payload)

    def _publish_task(
        self, kind: EventKind, task: TaskRecord, payload: Mapping[str, object] | None = None
    ) -> None:
        self._bus.publish(
            Event.stamp(kind, job_id=task.job_id, task_id=task.id, payload=payload)
        )

    def _publish_job(
        self, kind: EventKind, job: JobRecord, payload: Mapping[str, object] | None = None
    ) -> None:
        self._bus.publish(Event.stamp(kind, job_id=job.id, payload=payload))


def _next_status(task: TaskRecord, dependencies: Sequence[TaskRecord]) -> TaskStatus | None:
    """READY when the dependency mode is met, SKIPPED when it never can be, None while waiting."""
    if not dependencies:
        return TaskStatus.READY
    succeeded = sum(1 for node in dependencies if node.status is TaskStatus.SUCCEEDED)
    settled = all(node.status in TERMINAL for node in dependencies)
    match task.dependency_mode:
        case DependencyMode.ALL:
            if succeeded == len(dependencies):
                return TaskStatus.READY
            return TaskStatus.SKIPPED if settled else None
        case DependencyMode.ANY:
            if succeeded:
                return TaskStatus.READY
            return TaskStatus.SKIPPED if settled else None
        case _:
            return TaskStatus.READY if settled else None


def _first_error(tasks: Sequence[TaskRecord]) -> str | None:
    for task in tasks:
        if task.status is not TaskStatus.SUCCEEDED and task.error:
            return task.error
    return None


def _describe(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return (text.splitlines()[0] if text else type(error).__name__)[:300]


def _now() -> datetime:
    return datetime.now(UTC)
