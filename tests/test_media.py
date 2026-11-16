import os

import pytest

from kvedio_agent1.core.media import ProbeError, identify, probe


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
