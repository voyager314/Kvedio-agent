"""Read-only querying of one video's transcript, shaped for citation.

Each tool returns evidence — a stretch of transcript with the position it
occupies — plus a line saying what it did. The box owns the budget: a call that
would run past it returns what fits and says so, so the model narrows next time
instead of quietly receiving a truncated transcript.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.domain import Segment, VideoRecord, format_moment
from ..core.storage import Storage

MAX_SPAN_CHARS = 160
MAX_EVIDENCE_CHARS = 6_000  # the single-response evidence ceiling from the design
MAX_SEARCH_RESULTS = 10
MAX_RANGE_SEGMENTS = 40
SAMPLE_SEGMENTS = 24

TOOL_NAMES = frozenset({"overview", "search", "read_range"})


@dataclass(frozen=True)
class Evidence:
    """One stretch of transcript handed to the model, with the moment it occupies."""

    start_ms: int
    end_ms: int
    text: str

    def render(self) -> str:
        """One line the model can cite: the position in brackets, then the words."""
        return f"[{format_moment(self.start_ms)}] {self.text}"


@dataclass(frozen=True)
class ToolResult:
    """What one call produced: the evidence, and a line for the reader."""

    evidence: tuple[Evidence, ...]
    note: str


class ToolBox:
    """The tools one question may use, with a shared evidence budget."""

    def __init__(
        self, store: Storage, video: VideoRecord, *, budget: int = MAX_EVIDENCE_CHARS
    ) -> None:
        self._store = store
        self._video = video
        self._remaining = max(0, budget)

    @property
    def names(self) -> frozenset[str]:
        return TOOL_NAMES

    def catalogue(self) -> str:
        """One line per tool, written for the planning model."""
        return "\n".join(
            (
                "overview：给出录像里均匀抽样的片段，用来判断话题分布，无参数。",
                'search：按关键词找片段，参数 {"query": "关键词"}。',
                'read_range：取某段时间内的片段，参数 {"start": 起始秒, "end": 结束秒}。',
            )
        )

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Run one tool; an unknown name answers with a note instead of raising."""
        if name == "overview":
            return self._overview()
        if name == "search":
            return self._search(str(arguments.get("query", "")))
        if name == "read_range":
            return self._range(arguments)
        return ToolResult((), f"{name}：没有这个工具")

    def _overview(self) -> ToolResult:
        segments = self._store.transcript_segments(self._video.id)
        if not segments:
            return ToolResult((), "overview：转录为空")
        step = max(1, len(segments) // SAMPLE_SEGMENTS)
        sampled = segments[::step][:SAMPLE_SEGMENTS]
        return self._pack(sampled, f"overview：{len(segments)} 段，抽样 {len(sampled)} 段")

    def _search(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult((), "search：缺少关键词")
        terms = query.split()
        found = self._store.search_segments(self._video.id, terms, MAX_SEARCH_RESULTS)
        note = f"search({query})：{len(found)} 段"
        if not found and len(terms) == 1 and len(query.strip()) > 6:
            found = self._store.search_segments(
                self._video.id, _bigrams(query), MAX_SEARCH_RESULTS
            )
            note = f"search({query})：整句无命中，按 2 字片段找到 {len(found)} 段"
        return self._pack(found, note)

    def _range(self, arguments: Mapping[str, Any]) -> ToolResult:
        start = _seconds(arguments.get("start"))
        end = _seconds(arguments.get("end"))
        if start is None or end is None or end <= start:
            return ToolResult((), "read_range：需要 start < end 的秒数")
        segments = [
            segment
            for segment in self._store.transcript_segments(self._video.id)
            if segment.end_ms > start * 1000 and segment.start_ms < end * 1000
        ][:MAX_RANGE_SEGMENTS]
        return self._pack(segments, f"read_range({start}-{end}s)：{len(segments)} 段")

    def _pack(self, segments: Sequence[Segment], note: str) -> ToolResult:
        """Fit the segments into the remaining budget, naming any shortfall."""
        if self._remaining <= 0:
            return ToolResult((), f"{note}；证据预算已用尽")
        evidence: list[Evidence] = []
        used = 0
        for segment in segments:
            text = _clip(segment.text)
            if used + len(text) > self._remaining:
                note = f"{note}；证据预算不足，只取前 {len(evidence)} 段"
                break
            used += len(text)
            evidence.append(Evidence(segment.start_ms, segment.end_ms, text))
        self._remaining -= used
        return ToolResult(tuple(evidence), note)


def _clip(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= MAX_SPAN_CHARS else collapsed[: MAX_SPAN_CHARS - 1] + "…"


def _bigrams(text: str) -> list[str]:
    """Two-character windows, deduplicated: a stand-in for a Chinese tokenizer.

    A sentence searched as one substring matches nothing, because the transcript
    has no whitespace to split on. Overlapping windows give the database something
    to match without pretending to know where words begin.
    """
    cleaned = "".join(
        character
        for character in text
        if not character.isspace() and character not in "，。！？、,.!?：:；;\"'“”‘’（）()"
    )
    return list(dict.fromkeys(cleaned[index : index + 2] for index in range(len(cleaned) - 1)))


def _seconds(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None
