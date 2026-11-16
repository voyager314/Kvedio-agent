"""What the system concluded about a question, and the moments it rests on."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Citation:
    """One stretch of the recording that a conclusion points at.

    ``quote`` is the text at that moment, carried along so a reader can check the
    conclusion without running the retrieval again. ``kind`` says where it came
    from: ``transcript`` for words someone said, ``scene`` for what the picture
    showed when nobody was talking.
    """

    start_ms: int
    end_ms: int  # exclusive
    quote: str
    kind: str = "transcript"


@dataclass(frozen=True)
class Answer:
    """One conclusion for one job, with the evidence it cites.

    An answer with no citations is only produced when nothing was retrieved; a
    conclusion drawn from evidence always names the moments it drew on.

    A job answers more than once when its two evidence branches finish apart: the
    first stored text is ``partial``, written from the branch that arrived first,
    and the final text replaces it with ``stage`` ``final`` and a higher
    ``revision``. Only the latest row is kept; earlier revisions survive in the
    event stream the interface already rendered.
    """

    id: int
    job_id: str
    video_id: int
    question: str  # the user's own wording, never rewritten
    text: str
    model: str  # the chat model that wrote it
    citations: tuple[Citation, ...]
    created_at: datetime
    stage: str = "final"  # permitted values: partial, final
    revision: int = 1  # 1 for the first stored text, incremented on each replacement
