"""PyAV-backed media probing and resumable time-window preprocessing."""

from __future__ import annotations

import hashlib
import json
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

import av

from kvedio_agent.platform.config import Config
from kvedio_agent.platform.storage import Store


@dataclass(frozen=True)
class MediaProbe:
    """Stable technical properties observed from one local source file."""

    duration_ms: int  # Container duration in milliseconds; positive.
    video_streams: int  # Number of video streams; zero when absent.
    audio_streams: int  # Number of audio streams; zero when absent.
    width: int | None  # Primary video width in pixels, or None without video.
    height: int | None  # Primary video height in pixels, or None without video.
    frame_rate: float | None  # Primary video frames per second, or None when unknown.


@dataclass(frozen=True)
class SourceFingerprint:
    """File identity used to reject unsafe resume after source mutation."""

    size_bytes: int  # Source size in bytes; non-negative.
    modified_ns: int  # Source modification time in filesystem nanoseconds.
    digest: str  # Lowercase SHA-256 hex digest of the complete source file.


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

    def manifest_path(self, source_digest: str) -> Path:
        """Return a manifest path namespaced by source and processing settings."""

        processing_key = hashlib.sha256(
            json.dumps(self.config.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.store.root / "artifacts" / source_digest / processing_key / "manifest.json"

    def preprocess(
        self,
        path: Path,
        objective: str = "media preprocessing",
        provider_identity: dict[str, str] | None = None,
    ) -> tuple[str, str, MediaProbe]:
        """Register a source and produce deterministic, checkpointed media windows."""

        source = path.expanduser().resolve(strict=True)
        identity = fingerprint(source)
        media = probe(source, self.config)
        config_snapshot = {
            "runtime": self.config.model_dump(),
            "providers": provider_identity or {},
        }
        source_id = self.store.find_source(source, identity.digest) or self.store.register_source(
            source, identity.size_bytes, identity.modified_ns, identity.digest, media.__dict__
        )
        job_id = self.store.find_job(source_id, objective, config_snapshot)
        if job_id is None:
            job_id = self.store.create_job(source_id, objective, config_snapshot)
        manifest_path = self.manifest_path(identity.digest)
        output_dir = manifest_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
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
        specs: list[tuple[str, int, int, Path, Path]] = []
        start = 0
        while start < media.duration_ms:
            end = min(start + segment_ms, media.duration_ms)
            segment_id = self.store.add_segment(job_id, "media", start, end)
            specs.append((
                segment_id, start, end,
                output_dir / f"{start:012d}-{end:012d}.wav",
                output_dir / f"{start:012d}-{end:012d}.mp4",
            ))
            if end == media.duration_ms:
                break
            start = end - overlap_ms

        def generate(spec: tuple[str, int, int, Path, Path]) -> None:
            _, start_ms, end_ms, audio_path, clip_path = spec
            if not audio_path.exists() and media.audio_streams:
                self._write_audio(source, start_ms, end_ms, audio_path)
            if not clip_path.exists() and media.video_streams:
                self.write_clip(source, start_ms, end_ms, clip_path)

        with ThreadPoolExecutor(max_workers=self.config.max_media_workers) as pool:
            list(pool.map(generate, specs))

        for segment_id, start, end, audio_path, clip_path in specs:
            artifact_id = None
            for kind, artifact_path in (("audio", audio_path), ("clip", clip_path)):
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
                    "clip": str(clip_path) if clip_path.exists() else None,
                }
            )
            temporary_manifest = manifest_path.with_suffix(".tmp")
            temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary_manifest.replace(manifest_path)
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
                    if output.pts is None or output.time_base is None:
                        continue
                    output_start_ms = int(output.pts * output.time_base * 1000)
                    first = max(0, int((start_ms - output_start_ms) * 16_000 / 1000))
                    last = min(
                        output.samples,
                        int((end_ms - output_start_ms) * 16_000 / 1000),
                    )
                    if last > first:
                        chunks.append(output.to_ndarray()[:, first:last].tobytes())
            with wave.open(str(target), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(b"".join(chunks))

    def write_clip(self, source: Path, start_ms: int, end_ms: int, target: Path) -> int:
        """Re-encode one source window as an mp4 clip and return the frames it holds.

        The window is resampled to the configured perception frame rate and scaled
        under the configured width, so the size of a perception request follows the
        settings rather than the source resolution. Returns zero and writes nothing
        when the window decodes to no frames.
        """

        interval_ms = 1000 / self.config.clip_frames_per_second
        sampled: list[av.VideoFrame] = []
        with av.open(str(source), mode="r") as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise ValueError("video stream has no time base")
            container.seek(
                int(start_ms * av.time_base / 1000 / stream.time_base),
                stream=stream, backward=True, any_frame=False,
            )
            next_sample_ms = float(start_ms)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp_ms = int(frame.pts * (frame.time_base or stream.time_base) * 1000)
                if timestamp_ms >= end_ms:
                    break
                if timestamp_ms < start_ms or timestamp_ms < next_sample_ms:
                    continue
                sampled.append(frame)
                next_sample_ms += interval_ms
        if not sampled:
            return 0
        width, height = _even_scale(
            sampled[0].width, sampled[0].height, self.config.clip_max_width
        )
        with av.open(str(target), mode="w") as output:
            clip_stream = cast(
                av.video.stream.VideoStream,
                output.add_stream("libx264", rate=Fraction(self.config.clip_frames_per_second)),
            )
            clip_stream.width = width
            clip_stream.height = height
            clip_stream.pix_fmt = "yuv420p"
            for index, frame in enumerate(sampled):
                scaled = frame.reformat(width=width, height=height, format="yuv420p")
                scaled.pts = index
                scaled.time_base = Fraction(1, 1) / Fraction(
                    self.config.clip_frames_per_second
                )
                for packet in clip_stream.encode(scaled):
                    output.mux(packet)
            for packet in clip_stream.encode(None):
                output.mux(packet)
        return len(sampled)


def validate_resume(path: Path, expected: SourceFingerprint) -> bool:
    """Confirm that the referenced source still matches its persisted identity."""

    try:
        return fingerprint(path) == expected
    except OSError:
        return False


def _even_scale(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Fit a frame under a width ceiling, keeping aspect and even dimensions.

    Both results are even because the H.264 encoders reject odd dimensions in
    4:2:0, and both are at least two pixels.
    """

    if width <= max_width:
        return max(2, width - width % 2), max(2, height - height % 2)
    scaled_height = max(2, round(height * max_width / width))
    return max(2, max_width - max_width % 2), scaled_height - scaled_height % 2


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as generated:
        while block := generated.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
