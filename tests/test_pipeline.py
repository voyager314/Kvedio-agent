import queue
from pathlib import Path

import pytest

from kvedio_agent1.core.asr import AsrStage
from kvedio_agent1.core.config.schema import RuntimeLimits
from kvedio_agent1.core.domain import Event, EventKind, JobStatus, Segment, TaskStatus
from kvedio_agent1.core.pipeline import Pipeline
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus
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


def build(store: Storage, workspace: Workspace, transcribe=spoken, **runtime):
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
    return Pipeline(store, bus, RuntimeLimits(**runtime), workspace, asr), bus


def drain(channel: queue.Queue[Event]) -> list[Event]:
    collected: list[Event] = []
    while not channel.empty():
        collected.append(channel.get_nowait())
    return collected


def test_referencing_a_missing_file_leaves_no_trace(store, workspace, tmp_path):
    pipeline, _ = build(store, workspace)
    with pytest.raises(OSError):
        pipeline.submit(tmp_path / "gone.mp4", "讲了什么")
    assert store.videos() == []


def test_a_reference_is_probed_transcribed_and_succeeds(store, workspace, sample_video):
    pipeline, bus = build(store, workspace)
    channel = bus.subscribe()
    job = pipeline.submit(sample_video, "讲了什么")

    finished = pipeline.run(job)

    assert finished.status is JobStatus.SUCCEEDED
    video = store.video(job.video_id)
    assert video.media is not None
    assert video.media.video_codec == "h264"
    assert workspace.audio_path(video.id).exists()
    transcript = store.transcript(video.id)
    assert transcript is not None
    assert transcript.ready is True
    assert transcript.segment_count == 1
    assert all(
        task.status is TaskStatus.SUCCEEDED for task in store.tasks_for_job(job.id)
    )
    kinds = [event.kind for event in drain(channel)]
    assert kinds[0] is EventKind.JOB_QUEUED
    assert EventKind.VIDEO_REGISTERED in kinds
    assert kinds[-1] is EventKind.JOB_SUCCEEDED


def test_a_second_reference_reuses_the_video_row(store, workspace, sample_video):
    pipeline, _ = build(store, workspace)
    first = pipeline.submit(sample_video, "第一个问题")
    second = pipeline.submit(sample_video, "第二个问题")
    assert first.video_id == second.video_id
    assert len(store.videos()) == 1
    assert len(store.jobs()) == 2


def test_a_file_that_is_not_media_fails_the_job(store, workspace, tmp_path):
    impostor = tmp_path / "notes.mp4"
    impostor.write_text("这不是视频", encoding="utf-8")
    pipeline, _ = build(store, workspace)

    finished = pipeline.run(pipeline.submit(impostor, "讲了什么"))

    assert finished.status is JobStatus.FAILED
    assert "无法读取媒体文件" in finished.error


def test_a_recording_over_the_ceiling_is_refused_before_the_record_is_written(
    store, workspace, sample_video
):
    pipeline, _ = build(store, workspace, max_video_duration_seconds=1)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.FAILED
    assert "超过上限" in finished.error
    assert store.video(finished.video_id).media is None


def test_without_an_asr_model_the_job_stops_before_transcription(store, workspace, sample_video):
    pipeline, _ = build(store, workspace, transcribe=None)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    tasks = {task.id: task for task in store.tasks_for_job(finished.id)}
    assert finished.status is JobStatus.PARTIAL
    assert tasks["audio"].status is TaskStatus.SUCCEEDED
    assert tasks["transcribe"].status is TaskStatus.FAILED
    assert "未配置语音模型" in tasks["transcribe"].error


def test_a_video_without_audio_finishes_without_a_transcript(store, workspace, silent_video):
    pipeline, bus = build(store, workspace)
    channel = bus.subscribe()

    finished = pipeline.run(pipeline.submit(silent_video, "讲了什么"))

    assert finished.status is JobStatus.SUCCEEDED
    assert store.transcript(finished.video_id) is None
    messages = [
        event.payload.get("message")
        for event in drain(channel)
        if event.kind is EventKind.LOG_LINE
    ]
    assert "没有音轨，跳过转录" in messages
