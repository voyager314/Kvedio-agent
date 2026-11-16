"""Reading and checking the moments a conclusion points at."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ...tools import Evidence
from ..domain import Citation, format_moment

PATTERN = re.compile(r"\[(\d{1,3}):(\d{2})(?:\s*[–—~-]\s*(\d{1,3}):(\d{2}))?\]")


@dataclass(frozen=True)
class Span:
    """A stretch of the recording named by the conclusion.

    A single moment becomes a one-second span, so a point and a range can be
    stored the same way.
    """

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Verdict:
    """What the check found: the citations that may be kept, and what was wrong.

    ``missing`` holds the moments the answer cited that were not in the evidence,
    already formatted the way they appeared. ``cited`` counts every moment the
    answer named, valid or not, so "cited nothing" stays distinguishable from
    "cited only the wrong thing" — the correction says different things for each.
    """

    citations: tuple[Citation, ...]
    missing: tuple[str, ...]
    cited: int

    @property
    def ok(self) -> bool:
        """Whether the answer cites at least one moment and invents none."""
        return self.cited > 0 and not self.missing


def find(text: str) -> tuple[Span, ...]:
    """Every ``[mm:ss]`` or ``[mm:ss-mm:ss]`` the text contains, in reading order."""
    spans: list[Span] = []
    for match in PATTERN.finditer(text):
        start = int(match.group(1)) * 60_000 + int(match.group(2)) * 1_000
        if match.group(3) is None:
            spans.append(Span(start, start + 1_000))
            continue
        end = int(match.group(3)) * 60_000 + int(match.group(4)) * 1_000
        if end > start:
            spans.append(Span(start, end))
    return tuple(spans)


def check(text: str, evidence: Sequence[Evidence]) -> Verdict:
    """Keep the citations the evidence supports and name the ones it does not.

    A citation is supported when its start falls inside a given segment; a range
    additionally has to end inside one, so a conclusion cannot stretch from a
    moment it saw to a moment it did not. The quote stored with each citation is
    the transcript itself, never the model's paraphrase.
    """
    spans = find(text)
    accepted: list[Citation] = []
    refused: list[str] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        holder = _holder(span, evidence)
        if holder is None:
            refused.append(f"[{format_moment(span.start_ms)}]")
            continue
        key = (span.start_ms, span.end_ms)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            Citation(start_ms=span.start_ms, end_ms=span.end_ms, quote=holder.text)
        )
    return Verdict(citations=tuple(accepted), missing=tuple(refused), cited=len(spans))


def _holder(span: Span, evidence: Sequence[Evidence]) -> Evidence | None:
    anchored = next(
        (
            item
            for item in evidence
            if item.verbatim and item.start_ms <= span.start_ms < item.end_ms
        ),
        None,
    )
    if anchored is None:
        return None
    if span.end_ms - span.start_ms > 1_000 and not any(
        item.verbatim and item.start_ms < span.end_ms <= item.end_ms for item in evidence
    ):
        return None
    return anchored
