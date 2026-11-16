from pathlib import Path

import pytest

from kvedio_agent1.core.domain import Chunk, MediaInfo, Scene, Segment
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.supervisor import Chatting


class ScriptedChat:
    """Plays back one answer and keeps the prompt it was given."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[list] = []

    def complete(self, messages, *, json_mode: bool = False) -> Reply:
        self.calls.append(list(messages))
        return Reply(
            text=self._text, model="stub", prompt_tokens=0, completion_tokens=0
        )


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def prepare(store: Storage):
    video = store.register_video(
        Path("talk.mp4"), size_bytes=1, mtime_ns=1, fingerprint="fp-chat"
    )
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=20_000,
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
        Chunk(index=0, start_ms=0, end_ms=20_000),
        [
            Segment(start_ms=0, end_ms=5_000, text="预算还差二十万"),
            Segment(start_ms=5_000, end_ms=12_000, text="人力要补两个人"),
        ],
    )
    store.finish_transcript(transcript.id)
    return video


def test_a_text_file_is_read_and_answered(store, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("会议决定先修预算。", encoding="utf-8")
    chat = ScriptedChat("文件里说先修预算。")

    reply = Chatting(store, chat).answer("总结", text_path=path)

    assert reply.text == "文件里说先修预算。"
    prompt = chat.calls[0][0].content
    assert "notes.md" in prompt
    assert "会议决定先修预算。" in prompt
    assert "Text file" in prompt


def test_the_video_evidence_is_retrieved_without_delegation(store):
    video = prepare(store)
    chat = ScriptedChat("预算还差二十万 [00:01]。")

    reply = Chatting(store, chat).answer("预算多少", video=video)

    prompt = chat.calls[0][0].content
    assert "From the analysed video" in prompt
    assert "预算还差二十万" in prompt
    assert "[00:00]" in prompt
    assert reply.notes


def test_a_visual_text_request_uses_scene_evidence(store):
    video = prepare(store)
    store.save_scene(
        video.id, "sk", 0, Scene(0, 4_000, "白板上写着预算", "/tmp/frame.jpg")
    )
    chat = ScriptedChat("白板上写着预算 [00:01]。")

    reply = Chatting(store, chat).answer("画面里写了什么", video=video)

    prompt = chat.calls[0][0].content
    assert "白板上写着预算" in prompt
    assert "(scene)" in prompt
    assert reply.text == "白板上写着预算 [00:01]。"


def test_a_general_request_has_no_material(store):
    chat = ScriptedChat("好的。")

    reply = Chatting(store, chat).answer("你好")

    prompt = chat.calls[0][0].content
    assert "No file or video was supplied" in prompt
    assert reply.text == "好的。"


def test_a_missing_text_file_is_refused(store, tmp_path):
    with pytest.raises(ValueError, match="文本文件"):
        Chatting(store, ScriptedChat("x")).answer(
            "总结", text_path=tmp_path / "gone.md"
        )
