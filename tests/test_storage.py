import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from kvedio_agent1.core.domain import (
    Chunk,
    Citation,
    Coverage,
    DependencyMode,
    JobStatus,
    MediaInfo,
    NewTask,
    Segment,
    TaskStatus,
)
from kvedio_agent1.core.storage import Storage, StorageError
from kvedio_agent1.core.storage.schema import SCHEMA_VERSION
from kvedio_agent1.core.storage.store import INTERRUPTED


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def reference(store: Storage, tmp_path: Path, fingerprint: str = "fp-1"):
    return store.register_video(
        tmp_path / "video.mp4", size_bytes=1024, mtime_ns=17, fingerprint=fingerprint
    )


def test_a_repeated_fingerprint_reuses_the_record(store, tmp_path):
    first = reference(store, tmp_path)
    second = store.register_video(
        tmp_path / "moved.mp4", size_bytes=1024, mtime_ns=17, fingerprint="fp-1"
    )
    assert second.id == first.id
    assert second.media is None
    assert len(store.videos()) == 1


def test_probing_fills_the_media_block(store, tmp_path):
    video = reference(store, tmp_path)
    media = MediaInfo(
        duration_ms=90_000,
        container="mp4",
        video_codec="h264",
        width=1920,
        height=1080,
        fps=29.97,
        audio_codec="aac",
        sample_rate=48_000,
        channels=2,
    )
    updated = store.record_media(video.id, media)
    assert updated.media == media
    assert store.video(video.id).media.duration_ms == 90_000


def test_recording_media_for_an_unknown_video_fails(store):
    media = MediaInfo(1, None, None, None, None, None, None, None, None)
    with pytest.raises(StorageError, match="视频不存在"):
        store.record_media(999, media)


def test_a_job_and_its_graph_round_trip(store, tmp_path):
    video = reference(store, tmp_path)
    job = store.create_job(
        video.id,
        "讲了什么",
        [
            NewTask(id="probe", agent="media", action="probe", max_attempts=2),
            NewTask(
                id="transcribe",
                agent="asr",
                action="transcribe",
                depends_on=("probe",),
                dependency_mode=DependencyMode.PARTIAL,
            ),
        ],
    )
    assert job.status is JobStatus.QUEUED
    assert job.question == "讲了什么"
    stored = {task.id: task for task in store.tasks_for_job(job.id)}
    assert stored["probe"].status is TaskStatus.PENDING
    assert stored["probe"].max_attempts == 2
    assert stored["transcribe"].depends_on == ("probe",)
    assert stored["transcribe"].dependency_mode is DependencyMode.PARTIAL


def test_a_repeated_task_id_is_refused(store, tmp_path):
    video = reference(store, tmp_path)
    with pytest.raises(StorageError, match="id 重复"):
        store.create_job(
            video.id,
            "q",
            [
                NewTask(id="a", agent="media", action="x"),
                NewTask(id="a", agent="media", action="y"),
            ],
        )


def test_a_dependency_outside_the_graph_is_refused(store, tmp_path):
    video = reference(store, tmp_path)
    with pytest.raises(StorageError, match="不存在"):
        store.create_job(
            video.id, "q", [NewTask(id="a", agent="media", action="x", depends_on=("ghost",))]
        )


def test_a_cyclic_graph_is_refused(store, tmp_path):
    video = reference(store, tmp_path)
    with pytest.raises(StorageError, match="成环"):
        store.create_job(
            video.id,
            "q",
            [
                NewTask(id="a", agent="media", action="x", depends_on=("b",)),
                NewTask(id="b", agent="media", action="y", depends_on=("a",)),
            ],
        )


def test_a_job_for_an_unknown_video_is_refused(store):
    with pytest.raises(StorageError, match="不存在"):
        store.create_job(999, "q", [NewTask(id="a", agent="media", action="x")])


def test_saving_a_task_persists_its_mutable_fields(store, tmp_path):
    video = reference(store, tmp_path)
    job = store.create_job(video.id, "q", [NewTask(id="a", agent="media", action="x")])
    task = store.tasks_for_job(job.id)[0]
    store.save_task(
        replace(
            task,
            status=TaskStatus.RUNNING,
            attempt=1,
            progress=0.4,
            started_at=datetime.now(UTC),
        )
    )
    reloaded = store.tasks_for_job(job.id)[0]
    assert reloaded.status is TaskStatus.RUNNING
    assert reloaded.attempt == 1
    assert reloaded.progress == pytest.approx(0.4)


def test_interrupted_work_is_failed_not_left_running(store, tmp_path):
    video = reference(store, tmp_path)
    job = store.create_job(video.id, "q", [NewTask(id="a", agent="media", action="x")])
    task = store.tasks_for_job(job.id)[0]
    store.save_task(replace(task, status=TaskStatus.RUNNING, attempt=1))
    store.save_job(replace(store.job(job.id), status=JobStatus.RUNNING))

    assert store.abandon_interrupted() == [job.id]
    assert store.job(job.id).status is JobStatus.FAILED
    assert store.job(job.id).error == INTERRUPTED
    assert store.tasks_for_job(job.id)[0].status is TaskStatus.FAILED
    assert store.abandon_interrupted() == []


def test_a_newer_schema_is_not_opened(tmp_path):
    path = tmp_path / "future.db"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(StorageError, match="版本"):
        Storage(path)


def test_a_transcript_is_created_once_per_key(store, tmp_path):
    video = reference(store, tmp_path)
    first = store.open_transcript(video.id, "key-1", "model", "transcribe")
    again = store.open_transcript(video.id, "key-1", "model", "transcribe")
    assert first.id == again.id
    assert first.ready is False


def test_a_different_key_replaces_the_previous_transcript(store, tmp_path):
    video = reference(store, tmp_path)
    transcript = store.open_transcript(video.id, "key-1", "model", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key-1",
        Chunk(index=0, start_ms=0, end_ms=1_000),
        [Segment(start_ms=0, end_ms=500, text="旧内容")],
    )
    store.finish_transcript(transcript.id)

    fresh = store.open_transcript(video.id, "key-2", "model", "transcribe")

    assert fresh.cache_key == "key-2"
    assert fresh.segment_count == 0
    assert fresh.ready is False
    assert store.completed_chunks(video.id, "transcribe", "key-1") == set()


def test_writing_the_same_chunk_twice_leaves_one_copy(store, tmp_path):
    video = reference(store, tmp_path)
    transcript = store.open_transcript(video.id, "key", "model", "transcribe")
    chunk = Chunk(index=0, start_ms=0, end_ms=2_000)

    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        chunk,
        [Segment(start_ms=0, end_ms=900, text="第一次")],
    )
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        chunk,
        [Segment(start_ms=0, end_ms=900, text="第二次")],
    )

    assert store.transcript(video.id).segment_count == 1
    assert store.completed_chunks(video.id, "transcribe", "key") == {0}


def test_a_transcript_summarises_its_segments(store, tmp_path):
    video = reference(store, tmp_path)
    transcript = store.open_transcript(video.id, "key", "model", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        Chunk(index=0, start_ms=0, end_ms=3_000),
        [
            Segment(start_ms=100, end_ms=500, text="一"),
            Segment(start_ms=1_500, end_ms=2_400, text="二"),
        ],
    )
    store.finish_transcript(transcript.id)

    summary = store.transcript(video.id)
    assert summary.segment_count == 2
    assert summary.first_ms == 100
    assert summary.last_ms == 2_400
    assert summary.ready is True


def test_a_transcript_can_be_read_back_in_timeline_order(store, tmp_path):
    video = reference(store, tmp_path)
    transcript = store.open_transcript(video.id, "key", "model", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        Chunk(index=0, start_ms=0, end_ms=3_000),
        [
            Segment(start_ms=1_500, end_ms=2_400, text="第二句"),
            Segment(start_ms=100, end_ms=900, text="第一句"),
        ],
    )
    store.finish_transcript(transcript.id)

    assert [segment.text for segment in store.transcript_segments(video.id)] == [
        "第一句",
        "第二句",
    ]


def test_search_finds_segments_by_keyword_and_escapes_wildcards(store, tmp_path):
    video = reference(store, tmp_path)
    transcript = store.open_transcript(video.id, "key", "model", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        Chunk(index=0, start_ms=0, end_ms=9_000),
        [
            Segment(start_ms=0, end_ms=1_000, text="预算还差二十万"),
            Segment(start_ms=1_000, end_ms=2_000, text="100% 完成"),
            Segment(start_ms=2_000, end_ms=3_000, text="人力缺口"),
        ],
    )
    store.finish_transcript(transcript.id)

    assert [item.text for item in store.search_segments(video.id, ["预算"], 5)] == [
        "预算还差二十万"
    ]
    assert [item.text for item in store.search_segments(video.id, ["100%"], 5)] == [
        "100% 完成"
    ]
    assert store.search_segments(video.id, ["不存在的词"], 5) == []


def test_an_answer_keeps_its_citations(store, tmp_path):
    video = reference(store, tmp_path)
    job = store.create_job(video.id, "q", [NewTask(id="answer", agent="supervisor", action="a")])

    stored = store.record_answer(
        job.id,
        video.id,
        "讲了什么",
        "结论 [00:10]",
        "chat-model",
        [
            Citation(start_ms=10_000, end_ms=11_000, quote="预算不足"),
            Citation(start_ms=20_000, end_ms=25_000, quote="补两个人"),
        ],
    )

    assert stored.text == "结论 [00:10]"
    assert stored.model == "chat-model"
    assert [item.quote for item in stored.citations] == ["预算不足", "补两个人"]
    assert store.answer(job.id).citations[0].start_ms == 10_000


def test_recording_a_second_answer_replaces_the_first(store, tmp_path):
    video = reference(store, tmp_path)
    job = store.create_job(video.id, "q", [NewTask(id="answer", agent="supervisor", action="a")])
    store.record_answer(job.id, video.id, "q", "第一版", "m", [Citation(0, 1_000, "一")])

    replaced = store.record_answer(job.id, video.id, "q", "第二版", "m", [])

    assert replaced.text == "第二版"
    assert replaced.citations == ()
    assert store.answer(job.id).text == "第二版"


def test_an_unknown_job_has_no_answer(store):
    assert store.answer("job_missing") is None

def test_a_coverage_map_is_written_and_read_in_order(store, tmp_path):
    video = reference(store, tmp_path)
    store.save_coverage(video.id, "ck", 0, Coverage(0, 60_000, "开场与议程"))
    store.save_coverage(video.id, "ck", 1, Coverage(60_000, 120_000, "预算讨论"))

    assert [block.gist for block in store.coverage(video.id)] == ["开场与议程", "预算讨论"]
    assert store.coverage_positions(video.id, "ck") == {0, 1}
    assert store.coverage_count(video.id, "ck") == 2

    store.prune_coverage(video.id, "another-key")
    assert store.coverage(video.id) == []


def test_vectors_round_trip_with_their_segments(store, tmp_path):
    video = reference(store, tmp_path)
    transcript = store.open_transcript(video.id, "tkey", "model", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "tkey",
        Chunk(index=0, start_ms=0, end_ms=2_000),
        [
            Segment(start_ms=0, end_ms=1_000, text="预算不足"),
            Segment(start_ms=1_000, end_ms=2_000, text="补人"),
        ],
    )
    store.finish_transcript(transcript.id)
    indexed = store.indexed_segments(video.id)
    assert len(indexed) == 2

    vector = np.asarray([0.5, 0.25], dtype=np.float32).tobytes()
    store.save_vectors(video.id, "vk", [(indexed[0][0], indexed[0][1].start_ms, vector)])

    rows = store.vectorised(video.id)
    assert [segment.text for segment, _ in rows] == ["预算不足"]
    assert np.frombuffer(rows[0][1], dtype=np.float32).tolist() == [0.5, 0.25]
    assert store.vectorised_ids(video.id, "vk") == {indexed[0][0]}
    assert store.vector_count(video.id, "vk") == 1

    store.prune_vectors(video.id, "another-key")
    assert store.vectorised(video.id) == []


def test_memory_summary_counts_both_halves(store, tmp_path):
    video = reference(store, tmp_path)
    assert store.memory_summary(video.id) == (0, 0)
    store.save_coverage(video.id, "ck", 0, Coverage(0, 60_000, "开场"))
    assert store.memory_summary(video.id)[0] == 1
