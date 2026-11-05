"""Bounded evidence analysis and persistent follow-up conversations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kvedio_agent.pipeline.context import EvidenceIndex
from kvedio_agent.pipeline.tools import JobToolRegistry, ToolCallError
from kvedio_agent.platform.costs import ProviderGate
from kvedio_agent.platform.prompting import PromptRenderer
from kvedio_agent.platform.providers import Completion, ProviderError
from kvedio_agent.platform.storage import Store


class TextModel(Protocol):
    """Small text generation surface required by the analysis workflow."""

    def complete(
        self, system: str, prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


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


class LocatedEvidence(BaseModel):
    """One evidence item overlapping a requested source-relative time range."""

    evidence_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    modality: str
    content: str


class ContentLocation(BaseModel):
    """Timestamped evidence returned for one point or range lookup."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    matches: list[LocatedEvidence]
    uncertainty: str | None = None


class AnalysisService:
    """Create segment facts, timeline summaries, answers, and durable follow-ups."""

    def __init__(
        self, store: Store, index: EvidenceIndex, model: TextModel,
        gate: ProviderGate | None = None, job_id: str | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.model = model
        self.gate = gate
        self.job_id = job_id
        self.prompts = PromptRenderer()

    def understand_segment(
        self, job_id: str, start_ms: int, end_ms: int, retention_days: int = 30
    ) -> SegmentFact:
        """Summarize only evidence for one bounded source-time segment."""

        evidence = self.store.evidence_for_range(job_id, start_ms, end_ms)
        if not evidence:
            raise ValueError("segment has no evidence")
        prompt = self.prompts.render(
            "segment_extraction", evidence=_format_evidence(evidence)
        )
        data = _parse_json(self._complete("segment", prompt))
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
        prompt = self.prompts.render(
            "timeline_reduction", evidence=_format_evidence(summaries)
        )
        response = self._complete("timeline", prompt)
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
        prompt = self.prompts.render(
            "final_answer", objective=objective, evidence=_format_evidence(context)
        )
        response = self._complete("answer", prompt)
        try:
            result = AnalysisResult.model_validate(_parse_json(response))
        except (ValidationError, ValueError):
            raise ProviderError("analysis_invalid_response", retryable=False) from None
        available = {item["id"] for item in context}
        if any(not set(claim.evidence_ids).issubset(available) for claim in result.claims):
            raise ProviderError("analysis_invalid_citation", retryable=False)
        return result

    def locate(self, job_id: str, start_ms: int, end_ms: int) -> ContentLocation:
        """Return original evidence overlapping the requested source-relative range."""

        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("location range must be positive and ordered")
        evidence = self.store.evidence_for_range(job_id, start_ms, end_ms)
        matches = [
            LocatedEvidence(
                evidence_id=item["id"],
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                modality=item["modality"],
                content=str(item.get("content") or ""),
            )
            for item in evidence
            if item["modality"] in {"transcript", "visual"}
        ]
        return ContentLocation(
            start_ms=start_ms,
            end_ms=end_ms,
            matches=matches,
            uncertainty=(
                None if matches else "No transcript or visual evidence overlaps this range."
            ),
        )

    def summarize_milestone(self, job_id: str, start_ms: int, end_ms: int) -> str:
        """Summarize recent evidence at a key processing milestone with its source times."""

        evidence = self.store.evidence_for_range(job_id, start_ms, end_ms)
        if not evidence:
            return "No evidence was produced for this interval."
        prompt = self.prompts.render(
            "milestone_summary", evidence=_format_evidence(evidence)
        )
        data = _parse_json(self._complete("milestone", prompt))
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ProviderError("analysis_invalid_response", retryable=False)
        return summary.strip()

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
        registry = JobToolRegistry(self.store, job_id)
        tool_instruction = self.prompts.render("tool_selection")
        prompt = self.prompts.render(
            "follow_up", question=question, history=history,
            evidence=_format_evidence(context), tool_instruction=tool_instruction,
        )
        completion = self._complete_result("follow_up", prompt, registry.schemas())
        for _ in range(3):
            if not completion.tool_calls:
                break
            tool_results = []
            for call in completion.tool_calls:
                try:
                    arguments = json.loads(call["arguments"])
                    tool_result = registry.call(call["name"], arguments)
                except (json.JSONDecodeError, ToolCallError):
                    raise ProviderError("analysis_invalid_tool_call", retryable=False) from None
                for item in tool_result.get("evidence", []):
                    row = self.store.evidence_by_id(item["evidence_id"])
                    if row is not None and row["id"] not in {entry["id"] for entry in context}:
                        context.append(row)
                tool_results.append(json.dumps(tool_result, ensure_ascii=False))
            prompt += "\n\nTool results:\n" + "\n".join(tool_results)
            completion = self._complete_result("follow_up", prompt, registry.schemas())
        if completion.tool_calls:
            raise ProviderError("analysis_tool_call_limit", retryable=False)
        response = completion.content
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

    def _complete(self, stage: str, prompt: str) -> str:
        return self._complete_result(stage, prompt).content

    def _complete_result(
        self, stage: str, prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        if self.gate is None or self.job_id is None:
            return (
                self.model.complete("", prompt, tools)
                if tools is not None
                else self.model.complete("", prompt)
            )
        model_name = getattr(self.model, "model", "chat")
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        reservation = self.gate.reserve(
            self.job_id,
            stage,
            f"{stage}:{prompt_digest}:{model_name}",
            self.gate.estimate_chat(prompt, output_tokens=2048),
        )
        try:
            response = (
                self.model.complete("", prompt, tools)
                if tools is not None
                else self.model.complete("", prompt)
            )
        except Exception:
            self.gate.settle(reservation, None, failed=True)
            raise
        self.gate.settle(
            reservation, response.actual_cost_usd,
            response.input_tokens, response.output_tokens,
        )
        return response


def _format_evidence(items: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{item['id']}] {item['start_ms']}..{item['end_ms']} ms "
        f"{item['modality']}: {item.get('content') or ''}"
        for item in items
    )


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
