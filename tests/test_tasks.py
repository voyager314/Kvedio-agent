import queue
from pathlib import Path

import pytest

from kvedio_agent1.core.domain import (
    DependencyMode,
    Event,
    EventKind,
    JobStatus,
    NewTask,
    TaskStatus,
)
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import (
    Cancellation,
    EventBus,
    IllegalTransition,
    Scheduler,
    check_transition,
)


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


@pytest.fixture
def bus():
    return EventBus()


def prepare(store: Storage, tasks: list[NewTask]):
    video = store.register_video(
        Path("video.mp4"),
        size_bytes=1,
        mtime_ns=1,
        fingerprint=f"fp-{len(store.videos())}",
    )
    return store.create_job(video.id, "讲了什么", tasks)


def drain(channel: queue.Queue[Event]) -> list[Event]:
    collected: list[Event] = []
    while not channel.empty():
        collected.append(channel.get_nowait())
    return collected


def test_the_state_machine_permits_and_refuses_the_right_changes():
    check_transition(TaskStatus.PENDING, TaskStatus.READY)
    check_transition(TaskStatus.RUNNING, TaskStatus.SUCCEEDED)
    check_transition(TaskStatus.FAILED, TaskStatus.READY)
    with pytest.raises(IllegalTransition):
        check_transition(TaskStatus.PENDING, TaskStatus.SUCCEEDED)
    with pytest.raises(IllegalTransition):
        check_transition(TaskStatus.SUCCEEDED, TaskStatus.RUNNING)


def test_every_subscriber_receives_every_event(bus):
    first, second = bus.subscribe(), bus.subscribe()
    bus.publish(Event.stamp(EventKind.LOG_LINE, payload={"message": "hi"}))
    assert len(drain(first)) == 1
    assert len(drain(second)) == 1


def test_an_unsubscribed_channel_receives_nothing(bus):
    channel = bus.subscribe()
    bus.unsubscribe(channel)
    bus.publish(Event.stamp(EventKind.LOG_LINE))
    assert drain(channel) == []


def test_a_single_node_runs_and_reports_its_transitions(store, bus):
    job = prepare(store, [NewTask(id="probe", agent="media", action="probe")])
    channel = bus.subscribe()
    seen: list[str] = []

    def handler(context):
        seen.append(context.task.id)
        context.report(0.5, "读取容器信息")

    finished = Scheduler(store, bus, max_workers=2).run(job, {"probe": handler})

    assert finished.status is JobStatus.SUCCEEDED
    assert seen == ["probe"]
    assert [event.kind for event in drain(channel)] == [
        EventKind.JOB_STARTED,
        EventKind.TASK_READY,
        EventKind.TASK_STARTED,
        EventKind.TASK_PROGRESS,
        EventKind.TASK_SUCCEEDED,
        EventKind.JOB_SUCCEEDED,
    ]


def test_a_failing_node_fails_the_job_and_keeps_the_message(store, bus):
    job = prepare(store, [NewTask(id="probe", agent="media", action="probe")])

    def handler(context):
        raise RuntimeError("容器损坏")

    finished = Scheduler(store, bus).run(job, {"probe": handler})

    assert finished.status is JobStatus.FAILED
    assert "容器损坏" in finished.error
    assert store.tasks_for_job(job.id)[0].status is TaskStatus.FAILED


def test_a_node_retries_within_its_ceiling(store, bus):
    job = prepare(
        store, [NewTask(id="probe", agent="media", action="probe", max_attempts=2)]
    )
    attempts: list[int] = []
    channel = bus.subscribe()

    def handler(context):
        attempts.append(context.task.attempt)
        if len(attempts) == 1:
            raise RuntimeError("首次失败")

    finished = Scheduler(store, bus).run(job, {"probe": handler})

    assert finished.status is JobStatus.SUCCEEDED
    assert attempts == [1, 2]
    assert store.tasks_for_job(job.id)[0].attempt == 2
    failures = [event for event in drain(channel) if event.kind is EventKind.TASK_FAILED]
    assert failures[0].payload["retrying"] is True


def test_a_dependent_node_is_skipped_when_a_dependency_fails(store, bus):
    job = prepare(
        store,
        [
            NewTask(id="probe", agent="media", action="probe"),
            NewTask(id="transcribe", agent="asr", action="transcribe", depends_on=("probe",)),
        ],
    )
    ran: list[str] = []

    def boom(context):
        raise RuntimeError("坏文件")

    finished = Scheduler(store, bus).run(
        job, {"probe": boom, "transcribe": lambda context: ran.append("transcribe")}
    )

    statuses = {task.id: task.status for task in store.tasks_for_job(job.id)}
    assert statuses["probe"] is TaskStatus.FAILED
    assert statuses["transcribe"] is TaskStatus.SKIPPED
    assert ran == []
    assert finished.status is JobStatus.FAILED


def test_a_skipped_node_skips_its_own_dependents(store, bus):
    job = prepare(
        store,
        [
            NewTask(id="a", agent="media", action="x"),
            NewTask(id="b", agent="asr", action="y", depends_on=("a",)),
            NewTask(id="c", agent="supervisor", action="z", depends_on=("b",)),
        ],
    )

    def boom(context):
        raise RuntimeError("失败")

    Scheduler(store, bus).run(
        job,
        {"a": boom, "b": lambda context: None, "c": lambda context: None},
    )
    statuses = {task.id: task.status for task in store.tasks_for_job(job.id)}
    assert statuses == {
        "a": TaskStatus.FAILED,
        "b": TaskStatus.SKIPPED,
        "c": TaskStatus.SKIPPED,
    }


def test_an_any_dependency_runs_after_one_success(store, bus):
    job = prepare(
        store,
        [
            NewTask(id="asr", agent="asr", action="x"),
            NewTask(id="video", agent="video", action="y"),
            NewTask(
                id="fuse",
                agent="supervisor",
                action="z",
                depends_on=("asr", "video"),
                dependency_mode=DependencyMode.ANY,
            ),
        ],
    )
    ran: list[str] = []

    def boom(context):
        raise RuntimeError("视觉失败")

    def ok(context):
        ran.append(context.task.id)

    finished = Scheduler(store, bus).run(job, {"asr": boom, "video": ok, "fuse": ok})

    assert "fuse" in ran
    assert finished.status is JobStatus.PARTIAL


def test_a_partial_dependency_runs_after_every_upstream_settles(store, bus):
    job = prepare(
        store,
        [
            NewTask(id="asr", agent="asr", action="x"),
            NewTask(
                id="answer",
                agent="supervisor",
                action="y",
                depends_on=("asr",),
                dependency_mode=DependencyMode.PARTIAL,
            ),
        ],
    )
    ran: list[str] = []

    def boom(context):
        raise RuntimeError("转录失败")

    finished = Scheduler(store, bus).run(
        job, {"asr": boom, "answer": lambda context: ran.append("answer")}
    )

    assert ran == ["answer"]
    assert finished.status is JobStatus.PARTIAL
    assert "转录失败" in finished.error


def test_progress_never_moves_backwards(store, bus):
    job = prepare(store, [NewTask(id="probe", agent="media", action="probe")])
    observed: list[float] = []

    def handler(context):
        context.report(0.6)
        context.report(0.2)
        observed.append(context.task.progress)

    Scheduler(store, bus).run(job, {"probe": handler})
    assert observed == [pytest.approx(0.6)]


def test_a_missing_handler_is_refused_before_anything_runs(store, bus):
    job = prepare(store, [NewTask(id="probe", agent="media", action="probe")])
    with pytest.raises(ValueError, match="缺少任务处理器"):
        Scheduler(store, bus).run(job, {})
    assert store.job(job.id).status is JobStatus.QUEUED

def test_cancellation_stops_the_graph_without_retrying(store, bus):
    job = prepare(
        store,
        [
            NewTask(id="a", agent="x", action="a", max_attempts=3),
            NewTask(id="b", agent="x", action="b", depends_on=("a",)),
        ],
    )
    signal = Cancellation()
    calls: list[str] = []

    def first(context) -> None:
        calls.append("a")
        signal.cancel()
        context.raise_if_cancelled()

    def second(context) -> None:
        calls.append("b")

    finished = Scheduler(store, bus).run(job, {"a": first, "b": second}, signal)

    tasks = {task.id: task for task in store.tasks_for_job(job.id)}
    assert finished.status is JobStatus.CANCELED
    assert tasks["a"].status is TaskStatus.CANCELED
    assert tasks["a"].attempt == 1  # not retried
    assert tasks["b"].status is TaskStatus.CANCELED
    assert calls == ["a"]


def test_a_cancel_before_start_runs_nothing(store, bus):
    job = prepare(store, [NewTask(id="a", agent="x", action="a")])

    def handler(context) -> None:
        raise AssertionError("不应运行")

    signal = Cancellation()
    signal.cancel()
    finished = Scheduler(store, bus).run(job, {"a": handler}, signal)

    assert finished.status is JobStatus.CANCELED
    assert store.tasks_for_job(job.id)[0].status is TaskStatus.CANCELED
