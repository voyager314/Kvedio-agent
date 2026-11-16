"""What the system knows about a referenced video file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    """What a decode probe reports about one file.

    Every duration is milliseconds from the start of the file. A field the
    container does not describe is ``None`` rather than a default, so a missing
    audio track stays distinguishable from a silent one.
    """

    duration_ms: int
    container: str | None  # short format name, for example "mp4"
    video_codec: str | None
    width: int | None
    height: int | None
    fps: float | None  # average frame rate; the declared value for variable-rate files
    audio_codec: str | None
    sample_rate: int | None  # Hz
    channels: int | None


@dataclass(frozen=True)
class FileIdentity:
    """What identifies a file on disk, computed without reading it whole.

    ``fingerprint`` covers the size, the modification time and two samples, so a
    rewritten or truncated source is a different identity rather than the same
    one carrying artifacts that no longer match it.
    """

    path: Path
    size_bytes: int
    mtime_ns: int  # modification time in nanoseconds
    fingerprint: str  # hex sha256 over size, mtime and the head and tail samples


@dataclass(frozen=True)
class VideoRecord:
    """A referenced file: where it is, what identifies it, what probing reported.

    ``path`` is a locator only. Identity is ``fingerprint``, which covers the
    bytes and the modification time, so a rewritten source is a different record
    instead of the same one carrying artifacts that no longer match it.
    """

    id: int
    path: Path
    size_bytes: int
    mtime_ns: int  # modification time in nanoseconds, part of the fingerprint
    fingerprint: str  # hex sha256 over size, mtime and the head and tail samples
    media: MediaInfo | None  # None until probing has recorded the stream layout
    created_at: datetime
    updated_at: datetime
