from pathlib import Path

import av
import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated configuration home, so no test reads or writes the user's own directory."""
    target = tmp_path / "kvedio_home"
    monkeypatch.setenv("KVEDIO_AGENT_HOME", str(target))
    return target


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """A one-second mp4 with one video and one audio track, encoded by the decoder itself.

    Encoding rather than shipping a fixture file keeps the repository free of
    binary payloads and makes the expected stream layout explicit in code.
    """
    target = tmp_path_factory.mktemp("media") / "sample.mp4"
    container = av.open(str(target), mode="w")
    video = container.add_stream("libx264", rate=25)
    video.width, video.height, video.pix_fmt = 160, 120, "yuv420p"
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

    for index in range(47):
        frame = av.AudioFrame(format="s16", layout="stereo", samples=1024)
        frame.sample_rate = 48_000
        frame.pts = index * 1024
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in audio.encode(frame):
            container.mux(packet)

    for packet in (*video.encode(None), *audio.encode(None)):
        container.mux(packet)
    container.close()
    return target
