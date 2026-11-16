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

from ..core.domain import Segment, VideoRecord
from ..core.memory import VideoMemory
from ..core.storage import Storage

MAX_SPAN_CHARS = 160
MAX_EVIDENCE_CHARS = 6_000  # the single-response evidence ceiling from the design
MAX_SEARCH_RESULTS = 10
MAX_RANGE_SEGMENTS = 40
SAMPLE_SEGMENTS = 24
OVERVIEW_ANCHORS = 6  # real segments kept beside the map, so a claim can be cited


@dataclass(frozen=True)
class Tool:
    """One capability, described for the planner that chooses it.

    ``summary`` and ``arguments`` are English because the model reads them; the
    note a call returns is Chinese because the user reads that.
    """

    name: str
    summary: str
    arguments: Mapping[str, str]  # argument name -> what it means


TOOLS: tuple[Tool, ...] = (
    Tool(
        name="overview",
        summary="Sample the transcript evenly across the recording, to see which topics it covers.",
        arguments={},
    ),
    Tool(
        name="search",
        summary="Find the stretches of transcript that mention a keyword or phrase.",
        arguments={"query": "the words to look for, as they would be said"},
    ),
    Tool(
        name="scenes",
        summary="List the recording's shots with a description of what is visible in each.",
        arguments={},
    ),
    Tool(
        name="events",
        summary="List the recording's events, each with what was on screen and what was said.",
        arguments={},
    ),
    Tool(
        name="read_range",
        summary="Read everything said between two moments of the recording.",
        arguments={
            "start": "first second, as a whole number",
            "end": "last second, as a whole number; must be greater than start",
        },
    ),
)


@dataclass(frozen=True)
class Evidence:
    """One stretch of transcript handed to the model, with the moment it occupies.

    ``label`` is empty for transcript lines. A map line is labelled ``map``: it
    summarises a stretch instead of quoting it, and the citation checker refuses
    to anchor a claim on words nobody said. A scene line is labelled ``scene``:
    it describes what the picture showed, which is an observation of its own.
    """

    start_ms: int
    end_ms: int
    text: str
    label: str = ""


@dataclass(frozen=True)
class ToolResult:
    """What one call produced: the evidence, and a line for the reader."""

    evidence: tuple[Evidence, ...]
    note: str


class ToolBox:
    """The tools one question may use, with a shared evidence budget."""

    def __init__(
        self,
        store: Storage,
        video: VideoRecord,
        memory: VideoMemory,
        *,
        budget: int = MAX_EVIDENCE_CHARS,
    ) -> None:
        self._store = store
        self._video = video
        self._memory = memory
        self._remaining = max(0, budget)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The catalogue a planner chooses from; the same list this box can dispatch."""
        return TOOLS

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Run one tool; an unknown name answers with a note instead of raising."""
        if name == "overview":
            return self._overview()
        if name == "search":
            return self._search(str(arguments.get("query", "")))
        if name == "scenes":
            return self._scenes()
        if name == "events":
            return self._events()
        if name == "read_range":
            return self._range(arguments)
        return ToolResult((), f"{name}：没有这个工具")

    def _overview(self) -> ToolResult:
        segments = self._store.transcript_segments(self._video.id)
        if not segments:
            return ToolResult((), "overview：转录为空")
        blocks = self._memory.coverage()
        if not blocks:
            step = max(1, len(segments) // SAMPLE_SEGMENTS)
            sampled = segments[::step][:SAMPLE_SEGMENTS]
            return self._pack(sampled, f"overview：{len(segments)} 段，抽样 {len(sampled)} 段")
        step = max(1, len(segments) // OVERVIEW_ANCHORS)
        anchors = segments[::step][:OVERVIEW_ANCHORS]
        evidence = [
            Evidence(block.start_ms, block.end_ms, block.gist, label="map") for block in blocks
        ]
        evidence += [Evidence(item.start_ms, item.end_ms, _clip(item.text)) for item in anchors]
        return self._pack_evidence(
            evidence, f"overview：地图 {len(blocks)} 块 + 抽样 {len(anchors)} 段"
        )

    def _search(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult((), "search：缺少关键词")
        found = self._memory.recall(query, limit=MAX_SEARCH_RESULTS)
        return self._pack(found, f"search({query})：{len(found)} 段 · {self._memory.mode}")

    def _scenes(self) -> ToolResult:
        scenes = self._store.scenes(self._video.id)
        if not scenes:
            return ToolResult((), "scenes：没有场景描述")
        evidence = [
            Evidence(item.start_ms, item.end_ms, item.description, label="scene")
            for item in scenes
        ]
        return self._pack_evidence(evidence, f"scenes：{len(scenes)} 个场景")

    def _events(self) -> ToolResult:
        events = self._store.events(self._video.id)
        if not events:
            return ToolResult((), "events：没有事件")
        scenes = self._store.scenes(self._video.id)
        segments = self._store.transcript_segments(self._video.id)
        evidence = []
        for event in events:
            seeing = [
                scene.description for scene in scenes if _overlaps(scene, event)
            ][:2]
            saying = [segment.text for segment in segments if _overlaps(segment, event)][:3]
            parts = [f"{event.title}：{event.summary}"]
            if seeing:
                parts.append("画面：" + "；".join(seeing))
            if saying:
                parts.append("同期声：" + "；".join(saying))
            evidence.append(
                Evidence(
                    event.start_ms, event.end_ms, _clip("｜".join(parts)), label="event"
                )
            )
        return self._pack_evidence(
            evidence, f"events：{len(events)} 个事件，已对齐画面与同期声"
        )

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
        """Fit whole segments into the remaining budget."""
        return self._pack_evidence(
            [Evidence(item.start_ms, item.end_ms, _clip(item.text)) for item in segments], note
        )

    def _pack_evidence(self, evidence: Sequence[Evidence], note: str) -> ToolResult:
        """Fit evidence into the remaining budget, naming any shortfall."""
        if self._remaining <= 0:
            return ToolResult((), f"{note}；证据预算已用尽")
        kept: list[Evidence] = []
        used = 0
        for item in evidence:
            if used + len(item.text) > self._remaining:
                note = f"{note}；证据预算不足，只取前 {len(kept)} 条"
                break
            used += len(item.text)
            kept.append(item)
        self._remaining -= used
        return ToolResult(tuple(kept), note)


def _overlaps(left, right) -> bool:
    """Whether two timed things share any moment of the recording."""
    return left.start_ms < right.end_ms and left.end_ms > right.start_ms


def _clip(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= MAX_SPAN_CHARS else collapsed[: MAX_SPAN_CHARS - 1] + "…"


def _seconds(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None
