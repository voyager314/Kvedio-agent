from pathlib import Path

import pytest

from kvedio_agent1.core.domain import Chunk, Coverage, MediaInfo, Scene, Segment
from kvedio_agent1.core.memory import VideoMemory
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.tools import ToolBox


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def prepare(store: Storage, segments: list[Segment]):
    video = store.register_video(
        Path("talk.mp4"), size_bytes=1, mtime_ns=1, fingerprint="fp-tools"
    )
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=120_000,
            container="mp4",
            video_codec="h264",
            width=320,
            height=180,
            fps=25.0,
            audio_codec="aac",
            sample_rate=48_000,
            channels=2,
        ),
    )
    transcript = store.open_transcript(video.id, "key", "asr", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "key",
        Chunk(index=0, start_ms=0, end_ms=120_000),
        segments,
    )
    store.finish_transcript(transcript.id)
    return video


def box(store, video, **options):
    """A toolbox over one video, with the memory reader it needs."""
    return ToolBox(store, video, VideoMemory(store, video), **options)


SEGMENTS = [
    Segment(start_ms=0, end_ms=5_000, text="今天我们先过一遍项目进度"),
    Segment(start_ms=5_000, end_ms=12_000, text="预算方面还有二十万的缺口"),
    Segment(start_ms=12_000, end_ms=20_000, text="人力上需要再补两个后端"),
]


def test_overview_samples_the_transcript(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("overview", {})
    assert len(result.evidence) == 3
    assert result.note.startswith("overview")
    assert result.evidence[0].start_ms == 0


def test_search_returns_the_matching_segments(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("search", {"query": "预算"})
    assert [item.text for item in result.evidence] == ["预算方面还有二十万的缺口"]
    assert result.note == "search(预算)：1 段 · 关键词检索"


def test_search_falls_back_to_character_windows_for_a_long_phrase(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("search", {"query": "这次会议的预算怎么安排"})
    assert "预算" in result.note
    assert result.evidence


def test_search_without_a_query_returns_a_note(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("search", {"query": "  "})
    assert result.evidence == ()
    assert "缺少关键词" in result.note


def test_read_range_keeps_segments_that_straddle_the_boundary(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("read_range", {"start": 4, "end": 13})
    assert [item.text for item in result.evidence] == [
        "今天我们先过一遍项目进度",
        "预算方面还有二十万的缺口",
        "人力上需要再补两个后端",
    ]


def test_read_range_rejects_a_backwards_span(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("read_range", {"start": 9, "end": 3})
    assert result.evidence == ()
    assert "start < end" in result.note


def test_an_unknown_tool_answers_instead_of_raising(store):
    video = prepare(store, SEGMENTS)
    result = box(store, video).call("download", {})
    assert result.evidence == ()
    assert "没有这个工具" in result.note


def test_the_budget_stops_a_call_and_says_so(store):
    video = prepare(store, SEGMENTS)
    limited = box(store, video, budget=12)
    result = limited.call("overview", {})
    assert len(result.evidence) < len(SEGMENTS)
    assert "预算" in result.note

    spent = limited.call("search", {"query": "预算"})
    assert spent.evidence == ()
    assert "预算已用尽" in spent.note


def test_long_segments_are_clipped_for_the_model(store):
    video = prepare(
        store, [Segment(start_ms=0, end_ms=10_000, text="很长的句子" * 100)]
    )
    result = box(store, video).call("overview", {})
    assert result.evidence[0].text.endswith("…")
    assert len(result.evidence[0].text) <= 160

def test_the_scenes_tool_returns_labelled_evidence(store):
    video = prepare(store, SEGMENTS)
    store.save_scene(video.id, "sk", 0, Scene(0, 5_000, "白板上写着预算", "/tmp/frame.jpg"))
    result = box(store, video).call("scenes", {})
    assert [item.label for item in result.evidence] == ["scene"]
    assert result.evidence[0].text == "白板上写着预算"
    assert result.note == "scenes：1 个场景"


def test_the_map_lines_are_labelled(store):
    video = prepare(store, SEGMENTS)
    store.save_coverage(video.id, "ck", 0, Coverage(0, 20_000, "整段都在谈预算"))
    result = box(store, video).call("overview", {})
    assert any(item.label == "map" for item in result.evidence)
