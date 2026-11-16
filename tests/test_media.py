import os
import wave

import pytest

from kvedio_agent1.core.media import (
    NoAudioTrack,
    ProbeError,
    extract_audio,
    identify,
    probe,
)


def test_audio_is_extracted_to_a_sixteen_kilohertz_mono_wav(sample_video, tmp_path):
    target = tmp_path / "audio.wav"
    fractions: list[float] = []

    track = extract_audio(
        sample_video, target, expected_ms=1_000, progress=fractions.append
    )

    assert target.exists()
    assert track.sample_rate == 16_000
    assert track.duration_ms == pytest.approx(1_000, abs=60)
    with wave.open(str(target), "rb") as written:
        assert written.getnchannels() == 1
        assert written.getsampwidth() == 2
        assert written.getframerate() == 16_000
    assert fractions and fractions == sorted(fractions)


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
