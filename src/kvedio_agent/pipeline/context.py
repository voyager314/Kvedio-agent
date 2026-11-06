"""Persistent scratchpad, budgeted evidence retrieval, and model context assembly."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from kvedio_agent.platform.storage import Store

# Character ranges billed at roughly one token each by current providers.
_WIDE_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana and katakana
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xFF00, 0xFFEF),  # Halfwidth and fullwidth forms
)


@dataclass
class Scratchpad:
    """Compact analysis checkpoint containing references rather than raw media."""

    objective: str
    current_stage: str = "queued"
    completed_ranges: list[tuple[int, int]] = field(default_factory=list)
    pending_ranges: list[tuple[int, int]] = field(default_factory=list)
    failed_ranges: list[tuple[int, int, str]] = field(default_factory=list)
    transcript_facts: list[dict[str, Any]] = field(default_factory=list)
    visual_facts: list[dict[str, Any]] = field(default_factory=list)
    cross_modal_facts: list[dict[str, Any]] = field(default_factory=list)
    timeline_facts: list[dict[str, Any]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a compact checkpoint suitable for SQLite persistence."""

        return {
            "objective": self.objective,
            "current_stage": self.current_stage,
            "completed_ranges": self.completed_ranges,
            "pending_ranges": self.pending_ranges,
            "failed_ranges": self.failed_ranges,
            "transcript_facts": self.transcript_facts,
            "visual_facts": self.visual_facts,
            "cross_modal_facts": self.cross_modal_facts,
            "timeline_facts": self.timeline_facts,
            "entities": self.entities,
            "uncertainties": self.uncertainties,
            "contradictions": self.contradictions,
            "quality_flags": self.quality_flags,
            "evidence_refs": self.evidence_refs,
            "next_actions": self.next_actions,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Scratchpad:
        """Restore a checkpoint with empty defaults for omitted collections."""

        restored = dict(value)
        for field_name in ("completed_ranges", "pending_ranges"):
            restored[field_name] = [tuple(item) for item in restored.get(field_name, [])]
        restored["failed_ranges"] = [
            (int(item[0]), int(item[1]), str(item[2]))
            for item in restored.get("failed_ranges", [])
        ]
        return cls(**restored)

    def save(self, store: Store, job_id: str) -> None:
        """Persist the current analysis checkpoint."""

        store.save_checkpoint(job_id, self.to_dict())

    def record_segment(
        self, start_ms: int, end_ms: int, evidence_ids: list[str],
        transcript_facts: list[dict[str, Any]], visual_facts: list[dict[str, Any]],
        entities: list[str], uncertainty: str | None,
    ) -> None:
        """Merge one completed segment's references and facts into the checkpoint."""

        self.completed_ranges = [
            item for item in self.completed_ranges if item != (start_ms, end_ms)
        ]
        self.completed_ranges.append((start_ms, end_ms))
        self.completed_ranges.sort()
        self.pending_ranges = [item for item in self.pending_ranges if item != (start_ms, end_ms)]
        self.evidence_refs = list(dict.fromkeys([*self.evidence_refs, *evidence_ids]))
        self.transcript_facts.extend(transcript_facts)
        self.visual_facts.extend(visual_facts)
        self.entities = list(dict.fromkeys([*self.entities, *entities]))
        if uncertainty:
            self.uncertainties.append(uncertainty)


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens one string occupies, biased high.

    Counts one token per ideograph, kana, or fullwidth character and one per four
    remaining characters. No tokenizer is consulted, so the same text measures the
    same size whichever model a run is bound to, and a budget computed here cannot
    drift when the binding changes.
    """

    wide = sum(1 for char in text if _is_wide(char))
    return wide + (len(text) - wide + 3) // 4


@dataclass(frozen=True)
class ContextBudget:
    """Token ceilings for one assembled request, applied per section in priority order."""

    total_tokens: int = 32_000  # Hard ceiling on the whole assembled block.
    chapter_tokens: int = 2_000  # Ceiling on the chapter map section.
    history_tokens: int = 2_000  # Ceiling on the conversation section.
    evidence_tokens: int = 16_000  # Ceiling on the retrieved evidence section.
    tool_tokens: int = 8_000  # Ceiling on tool results folded into a later turn.
    max_evidence_items: int = 64  # Upper bound on evidence rows admitted, whatever their size.
    verbatim_turns: int = 2  # Newest turns kept in full; older turns compress to one line each.


@dataclass(frozen=True)
class AssembledContext:
    """One budgeted context block and the evidence the model was actually shown."""

    text: str  # Rendered block; sections appear in fixed priority order.
    evidence_ids: tuple[str, ...]  # IDs present in the block; the only citable set.
    notes: tuple[str, ...]  # Degradations applied; also stated inside text.
    tokens: int  # Estimated size of text; never above the configured total.


class ContextAssembler:
    """Build every model context block from sections under a single token ceiling.

    - A section is filled only from what higher-priority sections left unused, so a
      long conversation cannot displace the chapter map and evidence cannot displace
      either.
    - Evidence and tool rows are admitted whole or not at all; no value is cut at a
      byte offset, because half a transcript line reads as a complete one.
    - Every omission is stated inside the block, so a trimmed context cannot be
      mistaken for the full record.
    """

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()

    def assemble(
        self,
        chapters: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> AssembledContext:
        """Assemble one block, spending the ceiling on higher-priority sections first.

        chapters: chapter rows in source-time order; summaries are dropped before
        titles, and titles before whole chapters, because the map of the recording is
        worth more than any one chapter's detail.
        history: conversation messages oldest first; the newest turns stay verbatim
        and older ones compress to one line carrying their cited evidence IDs.
        evidence: retrieved rows in retrieval order, so dropping the tail drops the
        least relevant rows.
        tool_results: rows produced by tools during this turn; lowest priority.
        Returns the block, the evidence IDs it contains, and every degradation applied.
        """

        remaining = self.budget.total_tokens
        notes: list[str] = []
        sections: list[str] = []
        admitted: list[str] = []

        chapter_lines, used, note = self._chapter_section(
            chapters or [], min(self.budget.chapter_tokens, max(remaining, 0))
        )
        if chapter_lines:
            sections.append("Chapters:\n" + "\n".join(chapter_lines))
            remaining -= used
        if note:
            notes.append(note)

        history_lines, used, note = self._history_section(
            history or [], min(self.budget.history_tokens, max(remaining, 0))
        )
        if history_lines:
            sections.append("Conversation:\n" + "\n".join(history_lines))
            remaining -= used
        if note:
            notes.append(note)

        rows = (evidence or [])[: self.budget.max_evidence_items]
        if len(evidence or []) > len(rows):
            notes.append(f"{len(evidence or []) - len(rows)} evidence rows exceeded the item cap.")
        lines = [_evidence_line(row) for row in rows]
        taken, used = _fit(lines, min(self.budget.evidence_tokens, max(remaining, 0)))
        if taken:
            sections.append("Evidence:\n" + "\n".join(taken))
            remaining -= used
        admitted.extend(str(row["id"]) for row in rows[: len(taken)])
        if len(taken) < len(lines):
            notes.append(f"{len(lines) - len(taken)} evidence rows did not fit the budget.")

        tool_rows = tool_results or []
        lines = [_tool_line(row) for row in tool_rows]
        taken, used = _fit(lines, min(self.budget.tool_tokens, max(remaining, 0)))
        if taken:
            sections.append("Tool results:\n" + "\n".join(taken))
            remaining -= used
        if len(taken) < len(lines):
            notes.append(f"{len(lines) - len(taken)} tool results did not fit the budget.")

        if notes:
            sections.append("Record completeness:\n" + "\n".join(f"- {note}" for note in notes))
        text = "\n\n".join(sections)
        return AssembledContext(text, tuple(admitted), tuple(notes), estimate_tokens(text))

    def _chapter_section(
        self, chapters: list[dict[str, Any]], limit: int
    ) -> tuple[list[str], int, str | None]:
        full = [_chapter_line(row, with_summary=True) for row in chapters]
        taken, used = _fit(full, limit)
        if len(taken) == len(full):
            return taken, used, None
        titles = [_chapter_line(row, with_summary=False) for row in chapters]
        taken, used = _fit(titles, limit)
        if len(taken) == len(titles):
            return taken, used, "Chapter summaries were reduced to titles."
        omitted = len(titles) - len(taken)
        return taken, used, f"Chapter summaries were reduced to titles and {omitted} were omitted."

    def _history_section(
        self, history: list[dict[str, Any]], limit: int
    ) -> tuple[list[str], int, str | None]:
        verbatim = self.budget.verbatim_turns
        recent = history[-verbatim:] if verbatim else []
        older = history[: len(history) - len(recent)]
        lines = [_compressed_turn(message) for message in older]
        lines.extend(_verbatim_turn(message) for message in recent)
        taken, used = _fit(list(reversed(lines)), limit)
        taken.reverse()
        note = None
        if older:
            note = f"{len(older)} earlier turns were compressed to one line each."
        if len(taken) < len(lines):
            dropped = len(lines) - len(taken)
            note = f"{note or ''} {dropped} earlier turns were omitted.".strip()
        return taken, used, note


class EvidenceIndex:
    """Find timestamped evidence and hand back whole rows in relevance order."""

    def __init__(self, store: Store, budget: ContextBudget | None = None) -> None:
        self.store = store
        self.budget = budget or ContextBudget()

    def retrieve(
        self, job_id: str, query: str, time_range: tuple[int, int] | None = None
    ) -> list[dict[str, Any]]:
        """Return evidence rows under the item cap, ranked by the store's relevance order.

        job_id: the job whose evidence may be read; no other job is searched.
        query: free text matched against evidence content; ignored when time_range is given.
        time_range: inclusive source-relative bounds in milliseconds, or None to search
        by relevance instead.
        Rows are returned whole and already ranked, so a caller that keeps a prefix
        keeps the most relevant rows.
        """

        if time_range is not None:
            return self.store.evidence_for_range(job_id, *time_range)
        return self.store.search_evidence(
            job_id, _query_terms(query), self.budget.max_evidence_items
        )


def _evidence_line(item: dict[str, Any]) -> str:
    """Render one evidence row as the single line every prompt cites it by."""

    return (
        f"[{item['id']}] {item['start_ms']}..{item['end_ms']} ms "
        f"{item['modality']}: {item.get('content') or ''}"
    )


def _is_wide(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _WIDE_RANGES)


def _fit(lines: list[str], limit: int) -> tuple[list[str], int]:
    """Take whole lines in order while the running estimate stays within limit."""

    taken: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > limit:
            break
        taken.append(line)
        used += cost
    return taken, used


def _chapter_line(row: dict[str, Any], with_summary: bool) -> str:
    chapter = _loads(str(row.get("content") or ""))
    title = str(chapter.get("title") or "untitled")
    line = f"[{row['id']}] {row['start_ms']}..{row['end_ms']} ms {title}"
    summary = str(chapter.get("summary") or "")
    return f"{line}: {summary}" if with_summary and summary else line


def _verbatim_turn(message: dict[str, Any]) -> str:
    return f"{message.get('role', 'user')}: {str(message.get('content') or '').strip()}"


def _compressed_turn(message: dict[str, Any]) -> str:
    """Reduce one stored turn to a single line that keeps its cited evidence IDs."""

    role = str(message.get("role", "user"))
    content = str(message.get("content") or "").strip()
    claims = _loads(content).get("claims")
    if not isinstance(claims, list) or not claims:
        return f"{role}: {_one_line(content)}"
    gist = "; ".join(str(claim.get("text", "")) for claim in claims[:2] if isinstance(claim, dict))
    cited = dict.fromkeys(
        str(evidence_id)
        for claim in claims
        if isinstance(claim, dict)
        for evidence_id in claim.get("evidence_ids", [])
    )
    return f"{role}: {_one_line(gist)} [cited: {', '.join(list(cited)[:6])}]"


def _tool_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)


def _one_line(text: str, limit: int = 200) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _loads(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[\w\u4e00-\u9fff]{2,}", query)))
