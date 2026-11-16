from pathlib import Path

import pytest

from kvedio_agent1.core.domain import Chunk, MediaInfo, Segment
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.vision import VisionBuilder
from kvedio_agent1.core.workspace import Workspace


class FakeVision:
    """Names the frame it was given; counts how many it was asked to look at."""

    model = "fake-vision"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, image: Path, prompt: str) -> str:
        self.calls += 1
        assert image.exists()
        return f"画面 {self.calls}：一块纯色画面"


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "workspace")


def prepare(store: Storage, path: Path):
    video = store.register_video(
        path, size_bytes=path.stat().st_size, mtime_ns=1, fingerprint="fp-vision"
    )
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
        [Segment(start_ms=0, end_ms=2_000, text="大家好，看一下这块屏幕")],
    )
    store.finish_transcript(transcript.id)
    return video


def builder(store, workspace, vision, **options) -> VisionBuilder:
    return VisionBuilder(
        store,
        vision,
        workspace,
        sample_ms=250,
        min_ms=400,
        threshold=0.12,
        **options,
    )


def test_every_shot_is_described_once(store, workspace, two_shot_video):
    video = prepare(store, two_shot_video)
    vision = FakeVision()
    reports: list[float] = []

    stats = builder(store, workspace, vision).build(
        video, lambda fraction, message: reports.append(fraction)
    )

    assert stats.scenes == 2
    assert stats.described == 2
    assert vision.calls == 2
    assert reports[-1] == pytest.approx(1.0)
    scenes = store.scenes(video.id)
    assert [scene.description for scene in scenes] == [
        "画面 1：一块纯色画面",
        "画面 2：一块纯色画面",
    ]
    assert all(Path(scene.frame_path).exists() for scene in scenes)


def test_a_second_build_looks_at_nothing(store, workspace, two_shot_video):
    video = prepare(store, two_shot_video)
    vision = FakeVision()
    first = builder(store, workspace, vision)
    first.build(video, lambda fraction, message: None)
    assert first.current(video) is True

    builder(store, workspace, vision).build(video, lambda fraction, message: None)

    assert vision.calls == 2


def test_a_new_model_redescribes_the_shots(store, workspace, two_shot_video):
    video = prepare(store, two_shot_video)
    builder(store, workspace, FakeVision()).build(video, lambda fraction, message: None)

    class OtherVision(FakeVision):
        model = "other-vision"

    other = OtherVision()
    builder(store, workspace, other).build(video, lambda fraction, message: None)

    assert other.calls == 2
    assert len(store.scenes(video.id)) == 2  # not four


def test_scenes_are_folded_to_fit_the_budget(store, workspace, two_shot_video):
    video = prepare(store, two_shot_video)
    vision = FakeVision()
    stats = builder(store, workspace, vision, max_scenes=1).build(
        video, lambda fraction, message: None
    )
    assert stats.scenes == 1
    scene = store.scenes(video.id)[0]
    assert (scene.start_ms, scene.end_ms) == (0, 2_000)


def test_without_a_vision_model_nothing_is_built(store, workspace, two_shot_video):
    video = prepare(store, two_shot_video)
    vision_builder = builder(store, workspace, None)
    assert vision_builder.current(video) is True

    stats = vision_builder.build(video, lambda fraction, message: None)

    assert stats.scenes == 0
    assert store.scenes(video.id) == []
