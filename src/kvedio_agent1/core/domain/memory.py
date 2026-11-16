"""What a long recording looks like before anyone asks a question about it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Coverage:
    """One stretch of the recording, and the line that says what it is about.

    A gist is a summary, not a quote: it can orient a reader or a planner, but it
    cannot be cited, because the words in the recording at that moment are the
    segment's, not this line's.
    """

    start_ms: int
    end_ms: int
    gist: str
