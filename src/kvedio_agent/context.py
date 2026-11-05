"""Persistent multimodal scratchpad and budgeted evidence retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from kvedio_agent.storage import Store


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


@dataclass(frozen=True)
class ContextBudget:
    """Maximum evidence count and serialized bytes for one model request."""

    max_items: int = 12
    max_bytes: int = 24_000


class EvidenceIndex:
    """Find timestamped evidence and enforce bounded model context."""

    def __init__(self, store: Store, budget: ContextBudget | None = None) -> None:
        self.store = store
        self.budget = budget or ContextBudget()

    def retrieve(
        self, job_id: str, query: str, time_range: tuple[int, int] | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant evidence under configured item and byte budgets."""

        if time_range is not None:
            candidates = self.store.evidence_for_range(job_id, *time_range)
        else:
            candidates = self.store.search_evidence(
                job_id, _query_terms(query), self.budget.max_items * 4
            )
        candidates.sort(key=lambda item: _relevance(item, query))
        selected: list[dict[str, Any]] = []
        used_bytes = 0
        for item in candidates:
            encoded_size = len(str(item.get("content") or "").encode("utf-8"))
            if len(selected) >= self.budget.max_items:
                break
            if used_bytes + encoded_size > self.budget.max_bytes:
                content = str(item.get("content") or "")
                remaining = self.budget.max_bytes - used_bytes
                if remaining <= 0:
                    break
                item = {
                    **item,
                    "content": content.encode("utf-8")[:remaining].decode("utf-8", "ignore"),
                }
                encoded_size = len(item["content"].encode("utf-8"))
            selected.append(item)
            used_bytes += encoded_size
        return sorted(selected, key=lambda item: (item["start_ms"], item["end_ms"]))

    def context_text(self, job_id: str, query: str) -> str:
        """Format retrieved evidence with source-relative timestamps and IDs."""

        items = self.retrieve(job_id, query)
        return "\n".join(
            f"[{item['id']}] {item['start_ms']}..{item['end_ms']} ms "
            f"({item['modality']}): {item.get('content') or ''}"
            for item in items
        )


def _query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[\w\u4e00-\u9fff]{2,}", query)))


def _relevance(item: dict[str, Any], query: str) -> int:
    content = str(item.get("content") or "").casefold()
    return -sum(content.count(term.casefold()) for term in _query_terms(query))
