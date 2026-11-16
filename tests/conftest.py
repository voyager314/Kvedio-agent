import wave
from pathlib import Path

import av
import numpy as np
import pytest

from kvedio_agent1.core.domain import Citation

RATE = 16_000


class StubAnswerer:
    """Records a fixed conclusion so pipeline tests need no chat endpoint."""

    def __init__(
        self,
        store,
        text: str = "开场白之后进入正题 [00:00]",
        quote: str = "开场白",
    ) -> None:
        self._store = store
        self._text = text
        self._quote = quote

    def answer(self, job, report, *, cancelled=None):
        transcript = self._store.transcript(job.video_id)
        if transcript is None or not transcript.ready:
            raise ValueError("没有可用的转录，无法作答")
        report(0.5, "撰写结论")
        return self._store.record_answer(
            job.id,
            job.video_id,
            job.question,
            self._text,
            "stub",
            (Citation(start_ms=0, end_ms=600, quote=self._quote),),
        )


@pytest.fixture
def make_answerer():
    """The supervisor stand-in class, for tests that own their store."""
    return StubAnswerer


@pytest.fixture
def answerer(store):
    """A supervisor stand-in bound to the store under test."""
    return StubAnswerer(store)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated configuration home, so no test reads or writes the user's own directory."""
    target = tmp_path / "kvedio_home"
    monkeypatch.setenv("KVEDIO_AGENT_HOME", str(target))
    return target


@pytest.fixture
def make_wav(tmp_path):
    """Build mono 16 kHz WAV files from ``(duration_ms, amplitude)`` pairs."""

    def factory(pattern: list[tuple[int, float]], name: str = "audio.wav") -> Path:
        target = tmp_path / name
        with wave.open(str(target), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            for duration_ms, amplitude in pattern:
                count = RATE * duration_ms // 1000
                if amplitude:
                    samples = np.sin(np.arange(count) / 20) * amplitude
                else:
                    samples = np.zeros(count)
                out.writeframes(samples.astype(np.int16).tobytes())
        return target

    return factory


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """A one-second mp4 with one video and one audio track, encoded by the decoder itself.

    Encoding rather than shipping a fixture file keeps the repository free of
    binary payloads and makes the expected stream layout explicit in code.
    """
    target = tmp_path_factory.mktemp("media") / "sample.mp4"
    _write_video(target, with_audio=True)
    return target


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> Path:
    """A one-second mp4 with no audio track: the path a stage must treat as finished, not broken."""
    target = tmp_path_factory.mktemp("media") / "silent.mp4"
    _write_video(target, with_audio=False)
    return target


def _write_video(target: Path, *, with_audio: bool) -> None:
    container = av.open(str(target), mode="w")
    video = container.add_stream("libx264", rate=25)
    video.width, video.height, video.pix_fmt = 160, 120, "yuv420p"
    audio = None
    if with_audio:
        audio = container.add_stream("aac", rate=48_000)
        audio.layout = "stereo"

    for index in range(25):
        frame = av.VideoFrame(160, 120, "yuv420p")
        for plane in frame.planes:
            shade = bytes([16 if offset % 3 == 0 else 128 for offset in range(plane.buffer_size)])
            plane.update(shade)
        frame.pts = index
        for packet in video.encode(frame):
            container.mux(packet)

    if audio is not None:
        for index in range(47):
            frame = av.AudioFrame(format="s16", layout="stereo", samples=1024)
            frame.sample_rate = 48_000
            frame.pts = index * 1024
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in audio.encode(frame):
                container.mux(packet)

    trailing = video.encode(None) if audio is None else (*video.encode(None), *audio.encode(None))
    for packet in trailing:
        container.mux(packet)
    container.close()

@pytest.fixture(scope="session")
def two_shot_video(tmp_path_factory) -> Path:
    """Two seconds, two solid colours: one cut at 1 s, for scene detection."""
    target = tmp_path_factory.mktemp("media") / "two-shot.mp4"
    container = av.open(str(target), "w")
    stream = container.add_stream("libx264", rate=20)
    stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
    for index in range(40):
        frame = av.VideoFrame(160, 120, "yuv420p")
        shade = 60 if index < 20 else 200
        for plane in frame.planes:
            plane.update(bytes([shade]) * plane.buffer_size)
        frame.pts = index
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return target
