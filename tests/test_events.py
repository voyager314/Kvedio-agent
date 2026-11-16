import json
from pathlib import Path

import pytest

from kvedio_agent1.core.domain import Chunk, MediaInfo, Scene, Segment
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.vision import EventBuilder


class ScriptedChat:
    """Plays back replies in order and keeps every prompt it was given."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages, *, json_mode: bool = False) -> Reply:
        self.calls += 1
        return Reply(text=self._replies.pop(0), model="stub", prompt_tokens=0, completion_tokens=0)


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def prepare(store: Storage, path: Path, scenes: list[Scene] | None = None):
    video = store.register_video(path, size_bytes=1, mtime_ns=1, fingerprint="fp-events")
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=2_000,
            container="mp4",
            video_codec="h264",
            width=160,
            height=120,
            fps=20.0,
            audio_codec="aac",
            sample_rate=48_000,
            channels=2,
        ),
    )
    transcript = store.open_transcript(video.id, "tkey", "asr", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "tkey",
        Chunk(index=0, start_ms=0, end_ms=2_000),
        [Segment(start_ms=0, end_ms=2_000, text="第一块屏幕讲预算")],
    )
    store.finish_transcript(transcript.id)
    if scenes is not None:
        for position, scene in enumerate(scenes):
            store.save_scene(video.id, "scene-key", position, scene)
        store.save_scene_run(video.id, "scene-key", len(scenes))
    return video


SCENES = [
    Scene(0, 1_000, "第一块屏幕：白底黑字", "/tmp/0.jpg"),
    Scene(1_000, 2_000, "第二块屏幕：图表", "/tmp/1.jpg"),
]


def events_json(*items: tuple[int, int, str]) -> str:
    return json.dumps(
        {
            "events": [
                {"start": start, "end": end, "title": title, "summary": f"{title}的摘要"}
                for start, end, title in items
            ]
        }
    )


def test_the_events_are_stored_and_the_run_is_marked(store, two_shot_video):
    video = prepare(store, two_shot_video, SCENES)
    chat = ScriptedChat([events_json((0, 1, "开场"), (1, 2, "预算"))])
    builder = EventBuilder(store, chat, model="chat")

    stats = builder.build(video, lambda fraction, message: None)

    assert stats.events == 2
    assert [event.title for event in store.events(video.id)] == ["开场", "预算"]
    assert store.event_run(video.id) == (builder.key(video, "scene-key"), 2)
    assert builder.current(video) is True


def test_a_second_build_costs_nothing(store, two_shot_video):
    video = prepare(store, two_shot_video, SCENES)
    chat = ScriptedChat([events_json((0, 2, "整段"))])
    builder = EventBuilder(store, chat, model="chat")
    builder.build(video, lambda fraction, message: None)

    builder.build(video, lambda fraction, message: None)

    assert chat.calls == 1


def test_overlapping_events_are_folded(store, two_shot_video):
    video = prepare(store, two_shot_video, SCENES)
    chat = ScriptedChat([events_json((0, 1, "一"), (0, 2, "二"))])
    EventBuilder(store, chat, model="chat").build(video, lambda fraction, message: None)

    events = store.events(video.id)
    assert len(events) == 1
    assert (events[0].start_ms, events[0].end_ms) == (0, 2_000)


def test_an_event_cannot_run_past_the_recording(store, two_shot_video):
    video = prepare(store, two_shot_video, SCENES)
    chat = ScriptedChat([events_json((0, 99, "过长"))])
    EventBuilder(store, chat, model="chat").build(video, lambda fraction, message: None)

    event = store.events(video.id)[0]
    assert event.end_ms == 2_000


def test_an_unusable_reply_falls_back_to_grouping_the_shots(store, two_shot_video):
    video = prepare(store, two_shot_video, SCENES)
    chat = ScriptedChat(["我先看看这些场景。"])
    EventBuilder(store, chat, model="chat").build(video, lambda fraction, message: None)

    events = store.events(video.id)
    assert len(events) == 1
    assert events[0].title == "第一块屏幕：白底黑字"
    assert "图表" in events[0].summary


def test_without_shots_there_is_nothing_to_group(store, two_shot_video):
    video = prepare(store, two_shot_video, scenes=None)
    chat = ScriptedChat([])
    builder = EventBuilder(store, chat, model="chat")

    assert builder.current(video) is True
    assert builder.build(video, lambda fraction, message: None).events == 0
    assert chat.calls == 0
