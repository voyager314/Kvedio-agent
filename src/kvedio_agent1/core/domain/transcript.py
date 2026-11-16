"""What is said in a video, positioned on its timeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Segment:
    """One spoken stretch, positioned in the source video rather than in a chunk.

    ``text`` is kept as the transcriber returned it apart from whitespace
    normalization: every later citation quotes it, so rewriting it would make the
    quote untraceable.
    """

    start_ms: int  # milliseconds from the start of the video
    end_ms: int  # exclusive; always greater than start_ms
    text: str


@dataclass(frozen=True)
class Chunk:
    """A slice of the prepared audio that one transcription call covers.

    ``index`` is the position in the plan and keys the checkpoint, so a resumed run
    recognizes the slices it already paid for.
    """

    index: int
    start_ms: int  # milliseconds from the start of the video
    end_ms: int  # exclusive


@dataclass(frozen=True)
class Transcript:
    """What the store knows about one video's transcript.

    ``segment_count``, ``first_ms`` and ``last_ms`` are aggregates over the
    segments, so a reader can describe coverage without loading them all.
    ``completed_at`` is ``None`` while the transcript is still being written.
    """

    id: int
    video_id: int
    cache_key: str  # fingerprint, model, address and chunk length, hashed
    model: str
    segment_count: int
    first_ms: int | None  # start of the earliest segment; None when empty
    last_ms: int | None  # end of the latest segment; None when empty
    created_at: datetime
    completed_at: datetime | None

    @property
    def ready(self) -> bool:
        """Whether every planned chunk is written; a transcript under construction is not cited."""
        return self.completed_at is not None


def format_moment(milliseconds: int) -> str:
    """A position inside a recording as ``mm:ss``.

    Minutes are not wrapped at 60: a two-hour recording reads ``125:03`` rather
    than pretending to be an hour and five minutes. This is the form the model is
    told to cite in, so parsing and rendering must agree exactly.
    """
    total = max(0, milliseconds) // 1000
    return f"{total // 60:02d}:{total % 60:02d}"
