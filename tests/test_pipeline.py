import queue
import threading
import time
from pathlib import Path

import pytest

from kvedio_agent1.core.asr import AsrStage
from kvedio_agent1.core.config.schema import RuntimeLimits
from kvedio_agent1.core.domain import (
    DependencyMode,
    Event,
    EventKind,
    JobStatus,
    Scene,
    Segment,
    TaskStatus,
)
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.memory import MemoryBuilder
from kvedio_agent1.core.pipeline import Pipeline, Stages
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus
from kvedio_agent1.core.vision import EventStats, VisionStats
from kvedio_agent1.core.workspace import Workspace


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "workspace")


def spoken(audio: Path) -> list[Segment]:
    return [Segment(start_ms=0, end_ms=500, text="开场")]


def build(
    store, workspace, answering, transcribe=spoken, memory=None, vision=None, events=None, **runtime
):
    bus = EventBus()
    asr = None
    if transcribe is not None:
        asr = AsrStage(
            store,
            workspace,
            transcribe,
            model="asr-model",
            address="https://asr.example.invalid/v1/audio/transcriptions",
            chunk_ms=300_000,
            concurrency=1,
        )
    stages = Stages(
        runtime=RuntimeLimits(**runtime),
        workspace=workspace,
        asr=asr,
        memory=memory,
        vision=vision,
        events=events,
        answering=answering,
    )
    return Pipeline(store, bus, stages), bus


def drain(channel: queue.Queue[Event]) -> list[Event]:
    collected: list[Event] = []
    while not channel.empty():
        collected.append(channel.get_nowait())
    return collected


def test_referencing_a_missing_file_leaves_no_trace(store, workspace, answerer, tmp_path):
    pipeline, _ = build(store, workspace, answerer)
    with pytest.raises(OSError):
        pipeline.submit(tmp_path / "gone.mp4", "讲了什么")
    assert store.videos() == []


def test_a_reference_is_probed_transcribed_answered(store, workspace, answerer, sample_video):
    pipeline, bus = build(store, workspace, answerer)
    channel = bus.subscribe()
    job = pipeline.submit(sample_video, "讲了什么")

    finished = pipeline.run(job)

    assert finished.status is JobStatus.SUCCEEDED, finished.error
    video = store.video(job.video_id)
    assert video.media is not None
    assert video.media.video_codec == "h264"
    assert workspace.audio(video.id).exists()
    assert store.transcript(video.id).ready is True
    answer = store.answer(job.id)
    assert answer is not None
    assert answer.citations[0].quote == "开场白"
    assert [task.id for task in store.tasks_for_job(job.id)] == [
        "probe",
        "audio",
        "transcribe",
        "answer",
    ]
    assert all(task.status is TaskStatus.SUCCEEDED for task in store.tasks_for_job(job.id))
    kinds = [event.kind for event in drain(channel)]
    assert kinds[0] is EventKind.JOB_QUEUED
    assert EventKind.VIDEO_REGISTERED in kinds
    assert kinds[-1] is EventKind.JOB_SUCCEEDED


class _Chat:
    def complete(self, messages, *, json_mode: bool = False) -> Reply:
        return Reply(text="一块", model="stub", prompt_tokens=0, completion_tokens=0)


class _Embedder:
    model = "stub-embed"

    def embed(self, texts):
        return [[1.0] for _ in texts]


def test_memory_is_built_between_transcript_and_answer(
    store, workspace, answerer, sample_video
):
    builder = MemoryBuilder(store, _Chat(), _Embedder(), model="stub")
    pipeline, _ = build(store, workspace, answerer, memory=builder)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.SUCCEEDED, finished.error
    assert [task.id for task in store.tasks_for_job(finished.id)] == [
        "probe",
        "audio",
        "transcribe",
        "memory",
        "answer",
    ]
    assert store.memory_summary(finished.video_id)[1] == 1


def test_a_second_question_reuses_every_finished_stage(store, workspace, answerer, sample_video):
    pipeline, _ = build(store, workspace, answerer)
    first = pipeline.submit(sample_video, "第一个问题")
    pipeline.run(first)

    second = pipeline.submit(sample_video, "第二个问题")
    finished = pipeline.run(second)

    assert first.video_id == second.video_id
    assert len(store.videos()) == 1
    assert [task.id for task in store.tasks_for_job(second.id)] == ["answer"]
    assert finished.status is JobStatus.SUCCEEDED
    assert store.answer(second.id) is not None


def test_a_file_that_is_not_media_fails_the_job(store, workspace, answerer, tmp_path):
    impostor = tmp_path / "notes.mp4"
    impostor.write_text("这不是视频", encoding="utf-8")
    pipeline, _ = build(store, workspace, answerer)

    finished = pipeline.run(pipeline.submit(impostor, "讲了什么"))

    assert finished.status is JobStatus.FAILED
    assert "无法读取媒体文件" in finished.error


def test_a_recording_over_the_ceiling_is_refused_before_the_record_is_written(
    store, workspace, answerer, sample_video
):
    pipeline, _ = build(store, workspace, answerer, max_video_duration_seconds=1)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.FAILED
    assert "超过上限" in finished.error
    assert store.video(finished.video_id).media is None


def test_without_an_asr_model_the_job_stops_before_transcription(
    store, workspace, answerer, sample_video
):
    pipeline, _ = build(store, workspace, answerer, transcribe=None)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert finished.status is JobStatus.PARTIAL
    assert tasks["audio"].status is TaskStatus.SUCCEEDED
    assert tasks["transcribe"].status is TaskStatus.FAILED
    assert "未配置语音模型" in tasks["transcribe"].error
    assert store.answer(finished.id) is None


def test_without_a_supervisor_the_job_stops_at_the_answer(store, workspace, sample_video):
    pipeline, _ = build(store, workspace, answering=None)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert finished.status is JobStatus.PARTIAL
    assert tasks["transcribe"].status is TaskStatus.SUCCEEDED
    assert tasks["answer"].status is TaskStatus.FAILED
    assert "未配置对话模型" in tasks["answer"].error


def test_a_silent_video_without_pictures_has_nothing_to_answer_from(
    store, workspace, answerer, silent_video
):
    pipeline, _ = build(store, workspace, answerer)

    finished = pipeline.run(pipeline.submit(silent_video, "讲了什么"))

    assert finished.status is JobStatus.PARTIAL
    assert "没有可用的转录或画面描述" in finished.error
    assert store.transcript(finished.video_id) is None
    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert tasks["transcribe"].status is TaskStatus.SUCCEEDED  # nothing to transcribe
    assert tasks["answer"].status is TaskStatus.FAILED

class _Vision:
    """A describer that records the call instead of looking at pixels."""

    def __init__(self) -> None:
        self.builds = 0

    def current(self, video) -> bool:
        return False

    def build(self, video, report, *, cancelled=None) -> VisionStats:
        self.builds += 1
        report(1.0, "场景")
        return VisionStats(scenes=1, described=1)


def test_vision_and_transcription_hang_from_the_probe_as_separate_branches(
    store, workspace, answerer, sample_video
):
    vision = _Vision()
    memory = MemoryBuilder(store, _Chat(), _Embedder(), model="stub")
    pipeline, _ = build(store, workspace, answerer, memory=memory, vision=vision)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.SUCCEEDED, finished.error
    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert tasks["transcribe"].depends_on == ("audio",)
    assert tasks["memory"].depends_on == ("transcribe",)
    assert tasks["vision"].depends_on == ("probe",)
    assert tasks["answer_preliminary"].depends_on == ("memory", "vision")
    assert tasks["answer_preliminary"].dependency_mode is DependencyMode.ANY
    assert tasks["answer"].depends_on == ("memory", "vision", "answer_preliminary")
    assert tasks["answer"].dependency_mode is DependencyMode.PARTIAL
    assert vision.builds == 1


class _Events:
    """An extractor with no events yet: like the real one before any shot exists."""

    def __init__(self) -> None:
        self.builds = 0

    def current(self, video) -> bool:
        return self.builds > 0

    def build(self, video, report, *, cancelled=None) -> EventStats:
        self.builds += 1
        report(1.0, "事件")
        return EventStats(events=2)


def test_events_hang_from_vision_alone(store, workspace, answerer, sample_video):
    vision, events = _Vision(), _Events()
    pipeline, _ = build(store, workspace, answerer, vision=vision, events=events)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.SUCCEEDED, finished.error
    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert tasks["events"].depends_on == ("vision",)
    assert tasks["answer_preliminary"].depends_on == ("transcribe", "events")
    assert tasks["answer_preliminary"].dependency_mode is DependencyMode.ANY
    assert tasks["answer"].depends_on == ("transcribe", "events", "answer_preliminary")
    assert events.builds == 1


class _BrokenVision:
    """A describer that fails before writing a single scene."""

    def __init__(self) -> None:
        self.builds = 0

    def current(self, video) -> bool:
        return False

    def build(self, video, report, *, cancelled=None) -> VisionStats:
        self.builds += 1
        raise RuntimeError("视觉模型不可用")


class _BrokenSpeech:
    """A transcriber that fails after the audio was prepared."""

    def __call__(self, audio: Path) -> list[Segment]:
        raise RuntimeError("转写服务不可用")


class _SceneWriter:
    """A describer that writes one scene, so a picture-only answer has evidence."""

    def __init__(self, store) -> None:
        self._store = store

    def current(self, video) -> bool:
        return False

    def build(self, video, report, *, cancelled=None) -> VisionStats:
        self._store.save_scene(
            video.id,
            "scene-key",
            0,
            Scene(0, 900, "画面里有人在写代码", "/tmp/0.jpg"),
        )
        report(1.0, "场景")
        return VisionStats(scenes=1, described=1)


def test_a_vision_failure_still_answers_from_the_transcript(
    store, workspace, answerer, sample_video
):
    pipeline, _ = build(
        store, workspace, answerer, vision=_BrokenVision(), events=_Events()
    )

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert finished.status is JobStatus.PARTIAL
    assert tasks["transcribe"].status is TaskStatus.SUCCEEDED
    assert tasks["vision"].status is TaskStatus.FAILED
    assert tasks["events"].status is TaskStatus.SKIPPED
    assert tasks["answer"].status is TaskStatus.SUCCEEDED
    assert store.answer(finished.id) is not None


def test_a_transcript_failure_still_answers_from_the_pictures(
    store, workspace, answerer, sample_video
):
    vision = _SceneWriter(store)
    memory = MemoryBuilder(store, _Chat(), _Embedder(), model="stub")
    pipeline, _ = build(
        store,
        workspace,
        answerer,
        transcribe=_BrokenSpeech(),
        memory=memory,
        vision=vision,
    )

    finished = pipeline.run(pipeline.submit(sample_video, "画面里在做什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert finished.status is JobStatus.PARTIAL
    assert tasks["transcribe"].status is TaskStatus.FAILED
    assert tasks["memory"].status is TaskStatus.SKIPPED
    assert tasks["vision"].status is TaskStatus.SUCCEEDED
    assert tasks["answer"].status is TaskStatus.SUCCEEDED
    answer = store.answer(finished.id)
    assert answer is not None
    assert answer.citations[0].kind == "scene"
    assert answer.citations[0].quote == "画面里有人在写代码"


class _GatedVision:
    """A describer that holds the branch open until the test releases it."""

    def __init__(self, store) -> None:
        self._store = store
        self.release = threading.Event()

    def current(self, video) -> bool:
        return False

    def build(self, video, report, *, cancelled=None) -> VisionStats:
        if not self.release.wait(timeout=10):
            raise RuntimeError("视觉分支没有被释放")
        self._store.save_scene(
            video.id,
            "scene-key",
            0,
            Scene(0, 900, "画面里有人在写代码", "/tmp/0.jpg"),
        )
        report(1.0, "场景")
        return VisionStats(scenes=1, described=1)


def _wait_for_answer(store, job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        answer = store.answer(job_id)
        if answer is not None:
            return answer
        time.sleep(0.02)
    return None


def test_a_preliminary_answer_is_stored_before_the_slow_branch_finishes(
    store, workspace, answerer, sample_video
):
    vision = _GatedVision(store)
    pipeline, _ = build(store, workspace, answerer, vision=vision)
    job = pipeline.submit(sample_video, "讲了什么")
    worker = threading.Thread(target=pipeline.run, args=(job,))
    worker.start()
    try:
        partial = _wait_for_answer(store, job.id)
        assert partial is not None, "初步结论没有先落库"
        assert partial.stage == "partial"
        assert partial.revision == 1
        assert store.job(job.id).status is JobStatus.RUNNING
    finally:
        vision.release.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    final = store.answer(job.id)
    assert final is not None
    assert final.stage == "final"
    assert final.revision == 2


class _FlakyPreview:
    """A supervisor whose preview fails without affecting the final pass."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def answer(self, job, report, *, cancelled=None, preliminary=False):
        if preliminary:
            raise RuntimeError("预览失败")
        return self._inner.answer(job, report, cancelled=cancelled)


def test_a_failed_preview_does_not_taint_the_job(store, workspace, answerer, sample_video):
    pipeline, _ = build(
        store, workspace, _FlakyPreview(answerer), vision=_Vision()
    )

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert finished.status is JobStatus.SUCCEEDED, finished.error
    assert tasks["answer_preliminary"].status is TaskStatus.SUCCEEDED
    assert tasks["answer"].status is TaskStatus.SUCCEEDED
    assert store.answer(finished.id).stage == "final"


def test_both_failed_branches_leave_no_conclusion(store, workspace, answerer, sample_video):
    pipeline, _ = build(
        store,
        workspace,
        answerer,
        transcribe=_BrokenSpeech(),
        vision=_BrokenVision(),
    )

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert tasks["transcribe"].status is TaskStatus.FAILED
    assert tasks["vision"].status is TaskStatus.FAILED
    assert tasks["answer_preliminary"].status is TaskStatus.SKIPPED
    assert tasks["answer"].status is TaskStatus.FAILED
    assert store.answer(finished.id) is None


def test_a_running_job_can_be_cancelled(store, workspace, answerer, sample_video):
    gate = threading.Event()

    def slow(audio: Path) -> list[Segment]:
        gate.wait(timeout=5)
        return spoken(audio)

    pipeline, _ = build(store, workspace, answerer, transcribe=slow)
    job = pipeline.submit(sample_video, "讲了什么")
    worker = threading.Thread(target=pipeline.run, args=(job,))
    worker.start()
    for _ in range(100):
        if store.job(job.id).status is JobStatus.RUNNING:
            break
        time.sleep(0.02)

    assert pipeline.cancel(job.id) is True
    gate.set()
    worker.join(timeout=10)

    finished = store.job(job.id)
    assert finished.status is JobStatus.CANCELED
    tasks = {task.id: task for task in store.tasks_for_job(job.id)}
    assert tasks["answer"].status is TaskStatus.CANCELED
    assert store.answer(job.id) is None


def test_cancelling_an_unknown_job_is_refused(store, workspace, answerer):
    pipeline, _ = build(store, workspace, answerer)
    assert pipeline.cancel("job_missing") is False
