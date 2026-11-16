import queue

import pytest

from kvedio_agent1.core.config.schema import RuntimeLimits
from kvedio_agent1.core.domain import Event, EventKind, JobStatus, TaskStatus
from kvedio_agent1.core.pipeline import Pipeline
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import EventBus


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def build(store: Storage, **runtime) -> tuple[Pipeline, EventBus]:
    bus = EventBus()
    return Pipeline(store, bus, RuntimeLimits(**runtime)), bus


def drain(channel: queue.Queue[Event]) -> list[Event]:
    collected: list[Event] = []
    while not channel.empty():
        collected.append(channel.get_nowait())
    return collected


def test_referencing_a_missing_file_leaves_no_trace(store, tmp_path):
    pipeline, _ = build(store)
    with pytest.raises(OSError):
        pipeline.submit(tmp_path / "gone.mp4", "讲了什么")
    assert store.videos() == []


def test_a_reference_is_probed_and_the_job_succeeds(store, sample_video):
    pipeline, bus = build(store)
    channel = bus.subscribe()
    job = pipeline.submit(sample_video, "讲了什么")

    finished = pipeline.run(job)

    assert finished.status is JobStatus.SUCCEEDED
    video = store.video(job.video_id)
    assert video.media is not None
    assert video.media.video_codec == "h264"
    assert store.tasks_for_job(job.id)[0].status is TaskStatus.SUCCEEDED
    kinds = [event.kind for event in drain(channel)]
    assert kinds[0] is EventKind.JOB_QUEUED
    assert EventKind.VIDEO_REGISTERED in kinds
    assert kinds[-1] is EventKind.JOB_SUCCEEDED


def test_a_second_reference_reuses_the_video_row(store, sample_video):
    pipeline, _ = build(store)
    first = pipeline.submit(sample_video, "第一个问题")
    second = pipeline.submit(sample_video, "第二个问题")
    assert first.video_id == second.video_id
    assert len(store.videos()) == 1
    assert len(store.jobs()) == 2


def test_a_file_that_is_not_media_fails_the_job(store, tmp_path):
    impostor = tmp_path / "notes.mp4"
    impostor.write_text("这不是视频", encoding="utf-8")
    pipeline, _ = build(store)

    finished = pipeline.run(pipeline.submit(impostor, "讲了什么"))

    assert finished.status is JobStatus.FAILED
    assert "无法读取媒体文件" in finished.error


def test_a_recording_over_the_ceiling_is_refused_before_the_record_is_written(
    store, sample_video
):
    pipeline, _ = build(store, max_video_duration_seconds=1)

    finished = pipeline.run(pipeline.submit(sample_video, "讲了什么"))

    assert finished.status is JobStatus.FAILED
    assert "超过上限" in finished.error
    assert store.video(finished.video_id).media is None
