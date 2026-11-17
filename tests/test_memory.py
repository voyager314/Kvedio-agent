from pathlib import Path

import pytest

from kvedio_agent1.core.domain import Chunk, MediaInfo, Segment
from kvedio_agent1.core.llm import Reply
from kvedio_agent1.core.memory import MemoryBuilder, VideoMemory
from kvedio_agent1.core.memory.builder import plan_blocks
from kvedio_agent1.core.storage import Storage


def segments(count: int, step_ms: int = 10_000) -> list[Segment]:
    return [
        Segment(
            start_ms=index * step_ms,
            end_ms=(index + 1) * step_ms,
            text=f"第{index}段：讨论预算与人力" if index % 3 else f"第{index}段：其他话题",
        )
        for index in range(count)
    ]


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


def prepare(store: Storage, items: list[Segment]):
    video = store.register_video(
        Path("talk.mp4"), size_bytes=1, mtime_ns=1, fingerprint="fp-memory"
    )
    store.record_media(
        video.id,
        MediaInfo(
            duration_ms=items[-1].end_ms,
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
    transcript = store.open_transcript(video.id, "tkey", "asr", "transcribe")
    store.save_chunk(
        transcript.id,
        video.id,
        "transcribe",
        "tkey",
        Chunk(index=0, start_ms=0, end_ms=items[-1].end_ms),
        items,
    )
    store.finish_transcript(transcript.id)
    return video


class FakeChat:
    """Answers each coverage call with a line that names the stretch it saw."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, json_mode: bool = False) -> Reply:
        self.calls += 1
        return Reply(
            text=f" 第 {self.calls} 块：预算与人力 ",
            model="stub",
            prompt_tokens=0,
            completion_tokens=0,
        )


class FakeEmbedder:
    """A two-axis embedding: money talk on one axis, everything else on the other."""

    model = "fake-embed"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [
            [1.0, 0.0] if ("预算" in text or "资金" in text) else [0.0, 1.0] for text in texts
        ]


def test_blocks_group_about_three_minutes(store):
    blocks = plan_blocks(segments(30, step_ms=10_000), 60_000)
    assert [block.end_ms - block.start_ms for block in blocks] == [60_000] * 5
    assert blocks[-1].end_ms == 300_000


def test_a_short_tail_is_folded_into_its_predecessor():
    blocks = plan_blocks(segments(7, step_ms=10_000), 60_000)
    assert len(blocks) == 1
    assert blocks[0].end_ms == 70_000


def test_building_writes_a_map_and_vectors(store):
    video = prepare(store, segments(46))
    chat, embedder = FakeChat(), FakeEmbedder()
    builder = MemoryBuilder(store, chat, embedder, model="chat", block_ms=60_000)
    reports: list[float] = []

    stats = builder.build(video, lambda fraction, message: reports.append(fraction))

    assert stats.blocks > 1
    assert stats.vectors == 46
    assert chat.calls == stats.blocks
    assert embedder.calls == 3  # 46 segments in batches of 16
    assert reports[-1] == pytest.approx(1.0)
    assert store.coverage(video.id)[0].gist.startswith("第 1 块")
    assert len(store.vectorised(video.id)) == 46
    assert builder.current(video) is True


def test_a_second_build_costs_nothing(store):
    video = prepare(store, segments(46))
    chat, embedder = FakeChat(), FakeEmbedder()
    builder = MemoryBuilder(store, chat, embedder, model="chat", block_ms=60_000)
    builder.build(video, lambda fraction, message: None)
    chat_calls, embed_calls = chat.calls, embedder.calls

    builder.build(video, lambda fraction, message: None)

    assert chat.calls == chat_calls
    assert embedder.calls == embed_calls


def test_an_interrupted_build_resumes_at_the_first_missing_batch(store):
    video = prepare(store, segments(46))

    class Breaking(FakeEmbedder):
        def embed(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("上游中断")
            return [[1.0, 0.0] if "预算" in text else [0.0, 1.0] for text in texts]

    with pytest.raises(RuntimeError):
        MemoryBuilder(store, None, Breaking(), model="chat", block_ms=60_000).build(
            video, lambda fraction, message: None
        )
    assert len(store.vectorised(video.id)) == 16

    resumed = FakeEmbedder()
    MemoryBuilder(store, None, resumed, model="chat", block_ms=60_000).build(
        video, lambda fraction, message: None
    )
    assert resumed.calls == 2  # only the missing batches are paid for
    assert len(store.vectorised(video.id)) == 46


def test_a_new_embedding_model_replaces_the_old_vectors(store):
    video = prepare(store, segments(46))
    MemoryBuilder(store, None, FakeEmbedder(), model="chat", block_ms=60_000).build(
        video, lambda fraction, message: None
    )

    class OtherEmbedder(FakeEmbedder):
        model = "other-embed"

    other = OtherEmbedder()
    MemoryBuilder(store, None, other, model="chat", block_ms=60_000).build(
        video, lambda fraction, message: None
    )

    assert other.calls == 3
    assert len(store.vectorised(video.id)) == 46  # not 92


def test_a_short_recording_gets_vectors_but_no_map(store):
    video = prepare(store, segments(5))
    chat = FakeChat()
    stats = MemoryBuilder(store, chat, FakeEmbedder(), model="chat").build(
        video, lambda fraction, message: None
    )
    assert stats.blocks == 0
    assert chat.calls == 0
    assert stats.vectors == 5


def test_keyword_only_memory_reads_the_transcript(store):
    video = prepare(store, segments(6))
    memory = VideoMemory(store, video)
    assert memory.mode == "关键词检索"
    found = memory.recall("预算", limit=3)
    assert found and all("预算" in item.text for item in found)


def test_hybrid_memory_finds_what_keywords_alone_miss(store):
    video = prepare(store, segments(6))
    MemoryBuilder(store, None, FakeEmbedder(), model="chat").build(
        video, lambda fraction, message: None
    )
    memory = VideoMemory(store, video, FakeEmbedder())

    assert memory.mode == "混合检索"
    # No segment says 资金; only the vectors can rank the money talk.
    found = memory.recall("资金", limit=3)
    assert found
    assert all("预算" in item.text for item in found)


def test_a_question_without_a_map_reads_coverage_as_empty(store):
    video = prepare(store, segments(5))
    assert VideoMemory(store, video).coverage() == []
