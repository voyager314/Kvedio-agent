from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import pytest

from kvedio_agent.pipeline.media import MediaProcessor, fingerprint, probe, validate_resume
from kvedio_agent.platform.config import Config
from kvedio_agent.platform.storage import Store


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=2)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for index in range(4):
        frame = av.VideoFrame(64, 48, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes([index * 50]) * plane.buffer_size)
        frame.pts = index
        frame.time_base = Fraction(1, 2)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return path


def test_probe_reads_stream_metadata_and_limits_duration(sample_video: Path) -> None:
    result = probe(sample_video, Config(max_video_duration_seconds=10))
    assert result.duration_ms == 2000
    assert result.video_streams == 1
    assert result.width == 64
    with pytest.raises(ValueError, match="duration limit"):
        probe(sample_video, Config(max_video_duration_seconds=1))


def test_source_fingerprint_detects_mutation(sample_video: Path) -> None:
    before = fingerprint(sample_video)
    assert validate_resume(sample_video, before)
    sample_video.write_bytes(sample_video.read_bytes() + b"changed")
    assert not validate_resume(sample_video, before)


def test_preprocess_creates_playable_clips_and_resumes(
    sample_video: Path, tmp_path: Path
) -> None:
    store = Store(tmp_path / "workspace")
    config = Config(max_segment_duration_seconds=1, segment_overlap_seconds=0)
    processor = MediaProcessor(store, config)
    source_id, job_id, media = processor.preprocess(sample_video)
    manifest_path = processor.manifest_path(fingerprint(sample_video).digest)
    assert media.duration_ms == 2000
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = Path(manifest["segments"][0]["clip"])
    assert clip.is_file() and clip.suffix == ".mp4"
    with av.open(str(clip)) as container:
        assert container.streams.video
    _, resumed_job_id, _ = processor.preprocess(sample_video)
    reopened = Store(tmp_path / "workspace")
    assert resumed_job_id == job_id
    assert reopened.job_history()[0]["id"] == job_id
    assert reopened.recoverable_jobs()[0]["id"] == job_id
    assert source_id


def test_clip_size_follows_settings_rather_than_the_source(
    sample_video: Path, tmp_path: Path
) -> None:
    store = Store(tmp_path / "workspace")
    processor = MediaProcessor(store, Config(
        max_segment_duration_seconds=2, segment_overlap_seconds=0,
        clip_frames_per_second=1, clip_max_width=32,
    ))
    _, _, _ = processor.preprocess(sample_video)
    manifest_path = processor.manifest_path(fingerprint(sample_video).digest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with av.open(str(manifest["segments"][0]["clip"])) as container:
        stream = container.streams.video[0]
        assert stream.width == 32
        # The source runs at 2 fps over 2 seconds; one frame per second keeps half.
        assert sum(1 for _ in container.decode(stream)) == 2


def test_provider_identity_changes_job_but_reuses_preprocessed_media(
    sample_video: Path, tmp_path: Path
) -> None:
    store = Store(tmp_path / "workspace")
    processor = MediaProcessor(
        store, Config(max_segment_duration_seconds=1, segment_overlap_seconds=0)
    )
    _, first_job, _ = processor.preprocess(sample_video, "objective", {"asr": "model-a"})
    _, second_job, _ = processor.preprocess(sample_video, "objective", {"asr": "model-b"})
    assert first_job != second_job
    assert processor.manifest_path(fingerprint(sample_video).digest).exists()


def test_processing_settings_have_distinct_manifest_paths(
    sample_video: Path, tmp_path: Path
) -> None:
    store = Store(tmp_path / "workspace")
    first = MediaProcessor(store, Config(max_segment_duration_seconds=30))
    second = MediaProcessor(store, Config(max_segment_duration_seconds=60))
    digest = fingerprint(sample_video).digest
    assert first.manifest_path(digest) != second.manifest_path(digest)


def test_probe_rejects_invalid_media(tmp_path: Path) -> None:
    path = tmp_path / "invalid.mp4"
    path.write_bytes(b"not media")
    with pytest.raises(av.error.FFmpegError):
        probe(path, Config())
