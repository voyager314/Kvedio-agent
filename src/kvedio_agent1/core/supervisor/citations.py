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


def check(text: str, evidence: Sequence[Evidence]) -> tuple[tuple[Citation, ...], str | None]:
    """Keep the citations the evidence supports and name the ones it does not.

    A citation is supported when its start falls inside a given segment; a range
    additionally has to end inside one, so a conclusion cannot stretch from a
    moment it saw to a moment it did not. The quote stored with each citation is
    the transcript itself, never the model's paraphrase.
    """
    spans = find(text)
    if not spans:
        return (), "结论没有引用任何时间点"
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
    if refused:
        return tuple(accepted), "这些时间点不在给定片段里：" + "、".join(sorted(set(refused)))
    return tuple(accepted), None


def _holder(span: Span, evidence: Sequence[Evidence]) -> Evidence | None:
    anchored = next(
        (item for item in evidence if item.start_ms <= span.start_ms < item.end_ms), None
    )
    if anchored is None:
        return None
    if span.end_ms - span.start_ms > 1_000 and not any(
        item.start_ms < span.end_ms <= item.end_ms for item in evidence
    ):
        return None
    return anchored
