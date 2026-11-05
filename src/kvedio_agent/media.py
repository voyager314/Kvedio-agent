"""PyAV-backed media probing and resumable time-window preprocessing."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import av

from kvedio_agent.config import Config
from kvedio_agent.storage import Store


@dataclass(frozen=True)
class MediaProbe:
    """Stable technical properties observed from one local source file."""

    duration_ms: int
    video_streams: int
    audio_streams: int
    width: int | None
    height: int | None
    frame_rate: float | None


@dataclass(frozen=True)
class SourceFingerprint:
    """File identity used to reject unsafe resume after source mutation."""

    size_bytes: int
    modified_ns: int
    digest: str


def fingerprint(path: Path, block_size: int = 4 * 1024 * 1024) -> SourceFingerprint:
    """Hash a source file in bounded reads and capture its filesystem identity."""

    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    stat = resolved.stat()
    return SourceFingerprint(stat.st_size, stat.st_mtime_ns, digest.hexdigest())


def probe(path: Path, config: Config) -> MediaProbe:
    """Inspect streams with PyAV and enforce the configured duration ceiling."""

    with av.open(str(path), mode="r") as container:
        duration_ms = int(container.duration or 0) // 1000
        if duration_ms <= 0:
            raise ValueError("media duration is missing or invalid")
        if duration_ms > config.max_video_duration_seconds * 1000:
            raise ValueError("video exceeds the configured duration limit")
        video = list(container.streams.video)
        audio = list(container.streams.audio)
        primary = video[0] if video else None
        rate = float(primary.average_rate) if primary and primary.average_rate else None
        return MediaProbe(
            duration_ms=duration_ms,
            video_streams=len(video),
            audio_streams=len(audio),
            width=primary.codec_context.width if primary else None,
            height=primary.codec_context.height if primary else None,
            frame_rate=rate,
        )


class MediaProcessor:
    """Create bounded audio and frame artifacts without changing source media."""

    def __init__(self, store: Store, config: Config) -> None:
        self.store = store
        self.config = config

    def preprocess(
        self, path: Path, objective: str = "media preprocessing"
    ) -> tuple[str, str, MediaProbe]:
        """Register a source and produce deterministic, checkpointed media windows."""

        source = path.expanduser().resolve(strict=True)
        identity = fingerprint(source)
        media = probe(source, self.config)
        source_id = self.store.find_source(source, identity.digest) or self.store.register_source(
            source, identity.size_bytes, identity.modified_ns, identity.digest, media.__dict__
        )
        job_id = self.store.find_job(source_id, objective) or self.store.create_job(
            source_id, objective, self.config.model_dump()
        )
        output_dir = self.store.root / "artifacts" / identity.digest
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {"source_digest": identity.digest, "duration_ms": media.duration_ms,
                            "segments": []}
        else:
            manifest = {"source_digest": identity.digest, "duration_ms": media.duration_ms,
                        "segments": []}
        manifest["segments"] = []
        self.store.set_job_status(job_id, "running", "preprocess")
        segment_ms = self.config.max_segment_duration_seconds * 1000
        overlap_ms = self.config.segment_overlap_seconds * 1000
        start = 0
        while start < media.duration_ms:
            end = min(start + segment_ms, media.duration_ms)
            segment_id = self.store.add_segment(job_id, "media", start, end)
            audio_path = output_dir / f"{start:012d}-{end:012d}.wav"
            frame_path = output_dir / f"{start:012d}-{end:012d}.jpg"
            if not audio_path.exists() and media.audio_streams:
                self._write_audio(source, start, end, audio_path)
            if not frame_path.exists() and media.video_streams:
                self._write_keyframe(source, start, end, frame_path)
            artifact_id = None
            for kind, artifact_path in (("audio", audio_path), ("keyframe", frame_path)):
                if artifact_path.exists():
                    artifact_id = self.store.add_artifact(
                        job_id, kind, artifact_path, _file_digest(artifact_path),
                        f"pyav-{av.__version__}", self.config.retention_days, segment_id,
                    )
            self.store.set_segment_status(segment_id, "completed", artifact_id)
            manifest["segments"].append(
                {
                    "id": segment_id,
                    "start_ms": start,
                    "end_ms": end,
                    "audio": str(audio_path) if audio_path.exists() else None,
                    "keyframe": str(frame_path) if frame_path.exists() else None,
                }
            )
            temporary_manifest = manifest_path.with_suffix(".tmp")
            temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary_manifest.replace(manifest_path)
            if end == media.duration_ms:
                break
            start = end - overlap_ms
        self.store.upsert_stage(job_id, "preprocess", "completed", progress=1)
        return source_id, job_id, media

    def _write_audio(self, source: Path, start_ms: int, end_ms: int, target: Path) -> None:
        with av.open(str(source), mode="r") as container:
            stream = container.streams.audio[0]
            if stream.time_base is None:
                raise ValueError("audio stream has no time base")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
            chunks: list[bytes] = []
            seek_pts = int(start_ms * av.time_base / 1000 / stream.time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp_ms = int(frame.pts * (frame.time_base or stream.time_base) * 1000)
                if timestamp_ms >= end_ms:
                    break
                if timestamp_ms + int(frame.samples * 1000 / frame.sample_rate) <= start_ms:
                    continue
                for output in resampler.resample(frame):
                    chunks.append(output.to_ndarray().tobytes())
            with wave.open(str(target), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(b"".join(chunks))

    def _write_keyframe(self, source: Path, start_ms: int, end_ms: int, target: Path) -> None:
        with av.open(str(source), mode="r") as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise ValueError("video stream has no time base")
            seek_ms = start_ms + (end_ms - start_ms) // 2
            seek_pts = int(seek_ms * av.time_base / 1000 / stream.time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp_ms = int(frame.pts * (frame.time_base or stream.time_base) * 1000)
                if timestamp_ms < start_ms:
                    continue
                if timestamp_ms >= end_ms:
                    break
                rgb = frame.reformat(format="yuvj420p")
                with av.open(str(target), mode="w", format="image2") as output:
                    image_stream = cast(av.video.stream.VideoStream, output.add_stream("mjpeg"))
                    image_stream.width = rgb.width
                    image_stream.height = rgb.height
                    image_stream.pix_fmt = "yuvj420p"
                    for packet in image_stream.encode(rgb):
                        output.mux(packet)
                    for packet in image_stream.encode(None):
                        output.mux(packet)
                return


def validate_resume(path: Path, expected: SourceFingerprint) -> bool:
    """Confirm that the referenced source still matches its persisted identity."""

    try:
        return fingerprint(path) == expected
    except OSError:
        return False


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as generated:
        while block := generated.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
