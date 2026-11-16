"""What the system concluded about a question, and the moments it rests on."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Citation:
    """One stretch of the recording that a conclusion points at.

    ``quote`` is the transcript text at that moment, carried along so a reader can
    check the conclusion without running the retrieval again.
    """

    start_ms: int
    end_ms: int  # exclusive
    quote: str


@dataclass(frozen=True)
class Answer:
    """One conclusion for one job, with the evidence it cites.

    An answer with no citations is only produced when nothing was retrieved; a
    conclusion drawn from evidence always names the moments it drew on.
    """

    id: int
    job_id: str
    video_id: int
    question: str  # the user's own wording, never rewritten
    text: str
    model: str  # the chat model that wrote it
    citations: tuple[Citation, ...]
    created_at: datetime
