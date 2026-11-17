import os
import wave

import av
import numpy as np
import pytest

from kvedio_agent1.core.media import (
    NoAudioTrack,
    ProbeError,
    extract_audio,
    identify,
    needs_level_fix,
    probe,
)


def test_audio_is_extracted_to_a_sixteen_kilohertz_mono_wav(sample_video, tmp_path):
    target = tmp_path / "audio.wav"
    fractions: list[float] = []

    track_ms = extract_audio(
        sample_video, target, expected_ms=1_000, progress=fractions.append
    )

    assert target.exists()
    assert track_ms == pytest.approx(1_000, abs=60)
    with wave.open(str(target), "rb") as written:
        assert written.getnchannels() == 1
        assert written.getsampwidth() == 2
        assert written.getframerate() == 16_000
    assert fractions and fractions == sorted(fractions)


def test_a_quiet_track_is_lifted_to_a_usable_level(tmp_path):
    """A recording made from across a room must not reach the recogniser tens of decibels down."""
    source = _tone_video(tmp_path, amplitude=1_200)
    target = tmp_path / "audio.wav"

    extract_audio(source, target, expected_ms=1_000)

    assert 24_000 < _peak(target) < 32_000
    assert needs_level_fix(target) is False


def test_a_loud_track_keeps_its_level(tmp_path):
    """A track already at a usable level is not touched, so its samples stay as recorded."""
    source = _tone_video(tmp_path, amplitude=20_000)
    target = tmp_path / "audio.wav"

    extract_audio(source, target, expected_ms=1_000)

    assert 12_000 < _peak(target) < 26_000


def test_a_silent_track_is_not_amplified(tmp_path):
    """Silence carries no words; lifting it would only raise the noise floor."""
    source = _tone_video(tmp_path, amplitude=0)
    target = tmp_path / "audio.wav"

    extract_audio(source, target, expected_ms=1_000)

    assert _peak(target) == 0


def _tone_video(directory, *, amplitude: int):
    """A one-second mp4 carrying a 440 Hz tone at the requested peak amplitude."""
    target = directory / f"tone-{amplitude}.mp4"
    container = av.open(str(target), mode="w")
    stream = container.add_stream("aac", rate=48_000)
    stream.layout = "stereo"
    time = np.arange(1024 * 47, dtype=np.float32) / 48_000
    tone = (amplitude * np.sin(2 * np.pi * 440 * time)).astype(np.int16)
    for index in range(47):
        block = tone[index * 1024 : (index + 1) * 1024]
        frame = av.AudioFrame(format="s16", layout="stereo", samples=1024)
        frame.sample_rate = 48_000
        frame.pts = index * 1024
        stereo = np.empty(1024 * 2, dtype=np.int16)
        stereo[0::2] = block
        stereo[1::2] = block
        frame.planes[0].update(stereo.tobytes())
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return target


def _peak(path) -> int:
    with wave.open(str(path), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return int(np.abs(samples).max(initial=0))


def test_an_interrupted_extraction_leaves_no_file(silent_video, tmp_path):
    target = tmp_path / "audio.wav"
    with pytest.raises(NoAudioTrack):
        extract_audio(silent_video, target, expected_ms=1_000)
    assert not target.exists()
    assert not target.with_name("audio.wav.part").exists()


def test_probing_reports_both_tracks(sample_video):
    media = probe(sample_video)
    assert 950 <= media.duration_ms <= 1_100
    assert media.video_codec == "h264"
    assert (media.width, media.height) == (160, 120)
    assert media.fps == pytest.approx(25.0)
    assert media.audio_codec == "aac"
    assert media.sample_rate == 48_000
    assert media.channels == 2


def test_a_file_that_is_not_media_is_reported_as_such(tmp_path):
    target = tmp_path / "notes.mp4"
    target.write_text("这不是视频", encoding="utf-8")
    with pytest.raises(ProbeError, match="无法读取媒体文件"):
        probe(target)


def test_identity_is_stable_until_the_file_changes(sample_video):
    first = identify(sample_video)
    assert identify(sample_video).fingerprint == first.fingerprint
    assert first.size_bytes == sample_video.stat().st_size


def test_a_new_modification_time_changes_identity(sample_video, tmp_path):
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(sample_video.read_bytes())
    first = identify(copy)
    later = first.mtime_ns + 1_000_000
    os.utime(copy, ns=(later, later))
    assert identify(copy).fingerprint != first.fingerprint


def test_different_content_of_the_same_length_changes_identity(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    os.utime(second, ns=(first.stat().st_mtime_ns, first.stat().st_mtime_ns))
    assert identify(first).fingerprint != identify(second).fingerprint
