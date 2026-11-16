import queue
from pathlib import Path

import pytest

from kvedio_agent1.core.asr import AsrStage
from kvedio_agent1.core.config.schema import RuntimeLimits
from kvedio_agent1.core.domain import Event, EventKind, JobStatus, Segment, TaskStatus
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.memory import MemoryBuilder
from kvedio_agent1.core.pipeline import Pipeline, Stages
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus
from kvedio_agent1.core.vision import VisionStats
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


def build(store, workspace, answering, transcribe=spoken, memory=None, vision=None, **runtime):
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


def test_a_video_without_audio_cannot_be_answered(store, workspace, answerer, silent_video):
    pipeline, _ = build(store, workspace, answerer)

    finished = pipeline.run(pipeline.submit(silent_video, "讲了什么"))

    assert finished.status is JobStatus.PARTIAL
    assert "没有音轨" in finished.error
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

    def build(self, video, report) -> VisionStats:
        self.builds += 1
        report(1.0, "场景")
        return VisionStats(scenes=1, described=1)


def test_vision_runs_after_memory(store, workspace, answerer, sample_video):
    vision = _Vision()
    pipeline, _ = build(store, workspace, answerer, vision=vision)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.SUCCEEDED, finished.error
    assert [task.id for task in store.tasks_for_job(finished.id)] == [
        "probe",
        "audio",
        "transcribe",
        "vision",
        "answer",
    ]
    assert vision.builds == 1
