"""Bounded evidence analysis and persistent follow-up conversations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kvedio_agent.context import EvidenceIndex
from kvedio_agent.providers import ProviderError
from kvedio_agent.storage import Store


class TextModel(Protocol):
    """Small text generation surface required by the analysis workflow."""

    def complete(self, system: str, prompt: str) -> str: ...


class ConclusionClaim(BaseModel):
    """One factual conclusion and the evidence IDs that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class AnalysisResult(BaseModel):
    """Structured result with plan, trace, evidence, uncertainty, and quality."""

    model_config = ConfigDict(extra="forbid")

    objective: str
    plan: list[str]
    claims: list[ConclusionClaim]
    uncertainties: list[str]
    quality_assessment: str
    stage_trace: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SegmentFact:
    """A concise segment summary linked to transcript and visual evidence."""

    start_ms: int
    end_ms: int
    summary: str
    evidence_ids: tuple[str, ...]
    entities: tuple[str, ...] = ()
    uncertainty: str | None = None


class TimelineChapter(BaseModel):
    """One chronological chapter linked to segment evidence."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    title: str
    summary: str
    evidence_ids: list[str] = Field(min_length=1)

    def valid_range(self) -> bool:
        return self.end_ms > self.start_ms


class AnalysisService:
    """Create segment facts, timeline summaries, answers, and durable follow-ups."""

    def __init__(self, store: Store, index: EvidenceIndex, model: TextModel) -> None:
        self.store = store
        self.index = index
        self.model = model

    def understand_segment(
        self, job_id: str, start_ms: int, end_ms: int, retention_days: int = 30
    ) -> SegmentFact:
        """Summarize only evidence for one bounded source-time segment."""

        evidence = self.store.evidence_for_range(job_id, start_ms, end_ms)
        if not evidence:
            raise ValueError("segment has no evidence")
        prompt = _json_prompt(
            "Extract key facts, entities, and uncertainty from this bounded evidence.", evidence
        )
        data = _parse_json(self.model.complete(_SEGMENT_SYSTEM, prompt))
        fact = SegmentFact(
            start_ms=start_ms,
            end_ms=end_ms,
            summary=str(data.get("summary", "")),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            entities=tuple(data.get("entities", [])),
            uncertainty=data.get("uncertainty"),
        )
        allowed = {item["id"] for item in evidence}
        if not fact.summary or not fact.evidence_ids or not set(fact.evidence_ids) <= allowed:
            raise ProviderError("analysis_invalid_citation", retryable=False)
        self.store.add_artifact_summary(
            job_id, "segment_summary", start_ms, end_ms,
            json.dumps(data, ensure_ascii=False), "segment-agent", retention_days,
            evidence_ids=list(fact.evidence_ids),
        )
        return fact

    def segment_fact_from_summary(
        self, summary: dict[str, Any], provenance: dict[str, Any]
    ) -> SegmentFact:
        """Restore a segment fact from a persisted summary and its evidence references."""

        data = _parse_json(str(summary.get("content") or "{}"))
        return SegmentFact(
            int(summary["start_ms"]),
            int(summary["end_ms"]),
            str(data.get("summary", "")),
            tuple(str(item) for item in provenance.get("evidence_ids", [])),
            tuple(str(item) for item in data.get("entities", [])),
            data.get("uncertainty"),
        )

    def synthesize_timeline(self, job_id: str, retention_days: int = 30) -> list[TimelineChapter]:
        """Reduce stored segment summaries into ordered video chapters."""

        summaries = self.store.evidence_for_modality(job_id, "segment_summary")
        if not summaries:
            return []
        response = self.model.complete(
            _TIMELINE_SYSTEM,
            _json_prompt("Create chronological chapters and key moments.", summaries),
        )
        data = _parse_json(response)
        raw_chapters = data.get("chapters", [])
        if not isinstance(raw_chapters, list):
            raise ProviderError("analysis_invalid_response", retryable=False)
        available = {item["id"] for item in summaries}
        chapters: list[TimelineChapter] = []
        try:
            for raw in raw_chapters:
                chapter = TimelineChapter.model_validate(raw)
                if not chapter.valid_range() or not set(chapter.evidence_ids) <= available:
                    raise ProviderError("analysis_invalid_citation", retryable=False)
                self.store.add_artifact_summary(
                    job_id, "timeline_summary", chapter.start_ms, chapter.end_ms,
                    json.dumps(chapter.model_dump(), ensure_ascii=False),
                    "timeline-agent", retention_days,
                    evidence_ids=chapter.evidence_ids,
                )
                chapters.append(chapter)
        except ValidationError:
            raise ProviderError("analysis_invalid_response", retryable=False) from None
        return chapters

    def analyze(self, job_id: str, objective: str) -> AnalysisResult:
        """Answer an objective from bounded evidence and validate citations."""

        context = self.index.retrieve(job_id, objective)
        context.extend(self.store.evidence_for_modality(job_id, "timeline_summary"))
        context = list({item["id"]: item for item in context}.values())
        context.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
        if not context:
            raise ValueError("no matching evidence is available")
        response = self.model.complete(
            _ANSWER_SYSTEM,
            _json_prompt(objective, context),
        )
        try:
            result = AnalysisResult.model_validate(_parse_json(response))
        except (ValidationError, ValueError):
            raise ProviderError("analysis_invalid_response", retryable=False) from None
        available = {item["id"] for item in context}
        if any(not set(claim.evidence_ids).issubset(available) for claim in result.claims):
            raise ProviderError("analysis_invalid_citation", retryable=False)
        return result

    def follow_up(
        self, job_id: str, conversation_id: str, question: str, history_limit: int = 12
    ) -> AnalysisResult:
        """Persist a user question and cited answer in a recoverable conversation."""

        messages = self.store.conversation_messages(conversation_id)[-history_limit:]
        self.store.add_message(conversation_id, "user", question)
        context = self.index.retrieve(job_id, question)
        if not context:
            context = self.store.recent_evidence(job_id, self.index.budget.max_items)
        history = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = f"Conversation so far:\n{history}\n\nNew question: {question}\n\nEvidence:\n"
        prompt += _format_evidence(context)
        response = self.model.complete(_ANSWER_SYSTEM, prompt)
        try:
            result = AnalysisResult.model_validate(_parse_json(response))
        except (ValidationError, ValueError):
            raise ProviderError("analysis_invalid_response", retryable=False) from None
        evidence_ids = [
            evidence_id for claim in result.claims for evidence_id in claim.evidence_ids
        ]
        available = {item["id"] for item in context}
        if not set(evidence_ids) <= available:
            raise ProviderError("analysis_invalid_citation", retryable=False)
        self.store.add_message(conversation_id, "assistant", result.model_dump_json(), evidence_ids)
        return result


_SEGMENT_SYSTEM = (
    "Analyze only the supplied timestamped evidence. Return JSON with summary, evidence_ids, "
    "entities, and uncertainty. Do not invent facts or evidence references."
)
_TIMELINE_SYSTEM = (
    "Synthesize ordered video chapters from supplied segment summaries. Return JSON with "
    "chapters containing start_ms, end_ms, title, summary, and evidence_ids."
)
_ANSWER_SYSTEM = (
    "Answer using only supplied evidence. Return JSON with objective, plan, claims, "
    "uncertainties, and quality_assessment. Each claim must cite supplied evidence IDs."
)


def _format_evidence(items: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{item['id']}] {item['start_ms']}..{item['end_ms']} ms "
        f"{item['modality']}: {item.get('content') or ''}"
        for item in items
    )


def _json_prompt(instruction: str, items: list[dict[str, Any]]) -> str:
    return f"{instruction}\n\nEvidence:\n{_format_evidence(items)}"


def _parse_json(response: str) -> dict[str, Any]:
    """Parse a JSON object from a model response without exposing its contents."""

    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise ProviderError("analysis_invalid_response", retryable=False) from None
    if not isinstance(data, dict):
        raise ProviderError("analysis_invalid_response", retryable=False)
    return data
