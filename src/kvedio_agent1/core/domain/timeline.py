"""One happening in the recording, named."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEvent:
    """A stretch of the recording with a name.

    Named apart from the bus ``Event`` on purpose: one is a state change the
    interface renders, the other is a piece of the recording a reader looks for.

    An event is a grouping, not a quotation: it says "this is the part where they
    reviewed the budget". The shots and the speech inside it are the evidence; the
    title is not, so a conclusion never cites an event itself.
    """

    start_ms: int
    end_ms: int
    title: str
    summary: str
