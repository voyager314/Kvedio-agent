import shutil
from pathlib import Path

import pytest

from kvedio_agent1.core.asr import AsrStage
from kvedio_agent1.core.domain import Segment, UsageKind
from kvedio_agent1.core.storage import Storage
from kvedio_agent1.core.tasks import Cancelled
from kvedio_agent1.core.workspace import Workspace

PATTERN = [(1_000, 8_000), (200, 0)] * 3  # three seconds of speech, three gaps


class FakeTranscriber:
    """Stands in for the endpoint; records how many chunks were actually sent."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on_call = fail_on_call

    def __call__(self, audio: Path) -> list[Segment]:
        self.calls.append(audio.name)
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise RuntimeError("上游中断")
        return [Segment(start_ms=0, end_ms=500, text=f"第{len(self.calls)}段")]


@pytest.fixture
def store(tmp_path):
    opened = Storage(tmp_path / "kvedio.db")
    yield opened
    opened.close()


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "workspace")


def prepare(store: Storage, workspace: Workspace, wav: Path):
    video = store.register_video(
        Path("video.mp4"), size_bytes=1, mtime_ns=1, fingerprint="fp-asr"
    )
    target = workspace.audio(video.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(wav, target)
    return video


def stage(
    store, workspace, transcribe, *, model="asr-model", chunk_ms=1_000, concurrency=1, usage=None
):
    return AsrStage(
        store,
        workspace,
        transcribe,
        model=model,
        address="https://asr.example.invalid/v1/audio/transcriptions",
        chunk_ms=chunk_ms,
        concurrency=concurrency,
        usage=usage,
    )


def test_every_chunk_is_transcribed_and_recorded(store, workspace, make_wav):
    video = prepare(store, workspace, make_wav(PATTERN))
    transcriber = FakeTranscriber()
    reports: list[float] = []

    count = stage(store, workspace, transcriber).run(
        video, lambda fraction, message: reports.append(fraction)
    )

    transcript = store.transcript(video.id)
    assert len(transcriber.calls) == 3
    assert count == 3
    assert transcript.ready is True
    assert transcript.segment_count == 3
    assert transcript.first_ms == 0
    assert reports[-1] == pytest.approx(1.0)


def test_a_finished_transcript_costs_nothing_on_a_second_run(store, workspace, make_wav):
    video = prepare(store, workspace, make_wav(PATTERN))
    transcriber = FakeTranscriber()

    stage(store, workspace, transcriber).run(video, lambda fraction, message: None)
    stage(store, workspace, transcriber).run(video, lambda fraction, message: None)

    assert len(transcriber.calls) == 3
    assert store.transcript(video.id).segment_count == 3


def test_a_crashed_run_keeps_the_chunks_it_finished(store, workspace, make_wav):
    video = prepare(store, workspace, make_wav(PATTERN))
    failing = FakeTranscriber(fail_on_call=3)

    with pytest.raises(RuntimeError):
        stage(store, workspace, failing).run(video, lambda fraction, message: None)

    assert store.transcript(video.id).ready is False

    resumed = FakeTranscriber()
    count = stage(store, workspace, resumed).run(video, lambda fraction, message: None)

    assert len(resumed.calls) == 1
    assert count == 3
    assert store.transcript(video.id).ready is True


def test_changing_the_model_rebuilds_instead_of_mixing(store, workspace, make_wav):
    video = prepare(store, workspace, make_wav(PATTERN))
    first = FakeTranscriber()
    stage(store, workspace, first, model="model-a").run(video, lambda f, m: None)

    second = FakeTranscriber()
    stage(store, workspace, second, model="model-b").run(video, lambda f, m: None)

    transcript = store.transcript(video.id)
    assert len(second.calls) == 3
    assert transcript.model == "model-b"
    assert transcript.segment_count == 3


def test_a_video_without_prepared_audio_is_reported(store, workspace):
    video = store.register_video(
        Path("video.mp4"), size_bytes=1, mtime_ns=1, fingerprint="fp-none"
    )
    with pytest.raises(FileNotFoundError, match="音轨"):
        stage(store, workspace, FakeTranscriber()).run(video, lambda f, m: None)

def test_a_cancelled_run_writes_nothing(store, workspace, make_wav):
    video = prepare(store, workspace, make_wav(PATTERN))
    transcriber = FakeTranscriber()

    with pytest.raises(Cancelled):
        stage(store, workspace, transcriber).run(
            video, lambda fraction, message: None, cancelled=lambda: True
        )

    assert transcriber.calls == []
    assert store.transcript(video.id) is None

def test_every_chunk_reports_its_audio_seconds(store, workspace, make_wav):
    video = prepare(store, workspace, make_wav(PATTERN))
    seen = []

    stage(store, workspace, FakeTranscriber(), usage=seen.append).run(
        video, lambda fraction, message: None
    )

    assert len(seen) == 3
    assert all(item.kind is UsageKind.TRANSCRIPTION for item in seen)
    assert all(item.model == "asr-model" for item in seen)
    assert sum(item.seconds for item in seen) > 2
