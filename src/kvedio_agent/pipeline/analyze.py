"""Bounded evidence analysis and persistent follow-up conversations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kvedio_agent.pipeline.context import ContextAssembler, EvidenceIndex
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
    """Create segment facts, timeline summaries, answers, and durable follow-ups.

    Every request this service sends is built by one assembler under one token
    ceiling. No method appends to a prompt a previous call produced, so neither a
    long conversation nor a long tool round can push a request past that ceiling.
    """

    def __init__(
        self, store: Store, index: EvidenceIndex, model: TextModel,
        gate: ProviderGate | None = None, job_id: str | None = None,
        assembler: ContextAssembler | None = None,
        clip_observer: Callable[[int, int], list[dict[str, Any]]] | None = None,
        max_clip_ms: int = 0,
    ) -> None:
        self.store = store
        self.index = index
        self.model = model
        self.gate = gate
        self.job_id = job_id
        self.prompts = PromptRenderer()
        self.assembler = assembler or ContextAssembler(index.budget)
        self.clip_observer = clip_observer
        self.max_clip_ms = max_clip_ms

    def understand_segment(
        self, job_id: str, start_ms: int, end_ms: int, retention_days: int = 30
    ) -> SegmentFact:
        """Summarize only evidence for one bounded source-time segment."""

        evidence = self.store.evidence_for_range(job_id, start_ms, end_ms)
        if not evidence:
            raise ValueError("segment has no evidence")
        assembled = self.assembler.assemble(evidence=evidence)
        prompt = self.prompts.render("segment_extraction", context=assembled.text)
        data = _parse_json(self._complete("segment", prompt))
        fact = SegmentFact(
            start_ms=start_ms,
            end_ms=end_ms,
            summary=str(data.get("summary", "")),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            entities=tuple(data.get("entities", [])),
            uncertainty=data.get("uncertainty"),
        )
        allowed = set(assembled.evidence_ids)
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
        """Reduce stored segment summaries into ordered video chapters.

        Summaries are reduced in as many passes as the context budget requires, so a
        recording longer than one request can hold yields chapters for its whole
        length instead of chapters for its opening.
        """

        pending = self.store.evidence_for_modality(job_id, "segment_summary")
        chapters: list[TimelineChapter] = []
        while pending:
            assembled = self.assembler.assemble(evidence=pending)
            if not assembled.evidence_ids:
                raise ProviderError("analysis_context_budget_too_small", retryable=False)
            prompt = self.prompts.render("timeline_reduction", context=assembled.text)
            data = _parse_json(self._complete("timeline", prompt))
            raw_chapters = data.get("chapters", [])
            if not isinstance(raw_chapters, list):
                raise ProviderError("analysis_invalid_response", retryable=False)
            available = set(assembled.evidence_ids)
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
            pending = pending[len(assembled.evidence_ids):]
        return chapters

    def analyze(self, job_id: str, objective: str) -> AnalysisResult:
        """Answer an objective from retrieved evidence and the chapter map.

        The returned stage trace is the durable stage record, not what the model
        claims it did: the record exists independently of the answer and is the only
        account of the run a caller can check.
        """

        evidence = self.index.retrieve(job_id, objective)
        chapters = self.store.evidence_for_modality(job_id, "timeline_summary")
        if not evidence and not chapters:
            raise ValueError("no matching evidence is available")
        result = self._answer_with_tools(
            "answer", job_id, "final_answer", {"objective": objective}, evidence, chapters
        )
        return result.model_copy(update={"stage_trace": self._recorded_stages(job_id)})

    def _recorded_stages(self, job_id: str) -> list[str]:
        """Render the durable stage record as one line per stage, in execution order."""

        return [
            f"{stage['name']}: {stage['status']}"
            f" (attempt {stage.get('attempt') or 0},"
            f" progress {float(stage.get('progress') or 0):.0%}"
            + (f", {stage['error_category']}" if stage.get("error_category") else "")
            + ")"
            for stage in self.store.job_stages(job_id)
        ]

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
        assembled = self.assembler.assemble(evidence=evidence)
        prompt = self.prompts.render("milestone_summary", context=assembled.text)
        data = _parse_json(self._complete("milestone", prompt))
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ProviderError("analysis_invalid_response", retryable=False)
        return summary.strip()

    def follow_up(
        self, job_id: str, conversation_id: str, question: str, history_limit: int = 12
    ) -> AnalysisResult:
        """Persist a user question and cited answer in a recoverable conversation.

        history_limit bounds how many stored turns are considered; the assembler then
        decides how many of those survive verbatim, so conversation length changes what
        a request contains but never how large it is.
        """

        messages = self.store.conversation_messages(conversation_id)[-history_limit:]
        self.store.add_message(conversation_id, "user", question)
        evidence = self.index.retrieve(job_id, question)
        if not evidence:
            evidence = self.store.recent_evidence(job_id, self.index.budget.max_evidence_items)
        chapters = self.store.evidence_for_modality(job_id, "timeline_summary")
        result = self._answer_with_tools(
            "follow_up", job_id, "follow_up",
            {"tool_instruction": self.prompts.render("tool_selection"), "question": question},
            evidence, chapters, messages,
            clip_observer=self.clip_observer,
        )
        evidence_ids = [
            evidence_id for claim in result.claims for evidence_id in claim.evidence_ids
        ]
        self.store.add_message(conversation_id, "assistant", result.model_dump_json(), evidence_ids)
        return result

    def _answer_with_tools(
        self, stage: str, job_id: str, template: str, variables: dict[str, Any],
        evidence: list[dict[str, Any]], chapters: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None, max_rounds: int = 3,
        clip_observer: Callable[[int, int], list[dict[str, Any]]] | None = None,
    ) -> AnalysisResult:
        """Answer one question under the budget, folding tool output into a rebuilt context.

        Each round assembles the whole context again rather than appending to the
        previous prompt, so accumulated tool output cannot carry a request past the
        ceiling. Evidence a tool discovers joins the citable set for later rounds, and
        a claim may cite only what the assembled context actually contained.
        stage: usage-accounting label for the calls this answer makes.
        template: prompt template name; it receives the assembled block as `context`.
        variables: remaining template variables, none of which may carry evidence.
        evidence: retrieved rows; extended in place when a tool discovers more.
        clip_observer: rewatch hook offered as a tool, or None when the answer may not
        ask for footage again.
        Raises a provider error when the rounds run out with a tool call outstanding.
        """

        registry = JobToolRegistry(
            self.store, job_id, clip_observer, self.max_clip_ms
        )
        tool_results: list[dict[str, Any]] = []
        for _ in range(max_rounds + 1):
            assembled = self.assembler.assemble(
                chapters=chapters, history=history or [],
                evidence=evidence, tool_results=tool_results,
            )
            prompt = self.prompts.render(template, context=assembled.text, **variables)
            completion = self._complete_result(stage, prompt, registry.schemas())
            if not completion.tool_calls:
                return _validated(completion.content, set(assembled.evidence_ids))
            tool_results.extend(self._run_tools(completion.tool_calls, registry, evidence))
        raise ProviderError("analysis_tool_call_limit", retryable=False)

    def _run_tools(
        self, calls: tuple[dict[str, Any], ...], registry: JobToolRegistry,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute one round of tool calls and admit the evidence they discovered.

        evidence is extended in place with returned rows that are not already present,
        which is what makes a tool-discovered row citable in the next round.
        Raises a provider error when a call is malformed or names an unknown tool.
        """

        results: list[dict[str, Any]] = []
        known = {item["id"] for item in evidence}
        for call in calls:
            try:
                result = registry.call(call["name"], json.loads(call["arguments"]))
            except (json.JSONDecodeError, ToolCallError):
                raise ProviderError("analysis_invalid_tool_call", retryable=False) from None
            for item in result.get("evidence", []):
                row = self.store.evidence_by_id(item["evidence_id"])
                if row is not None and row["id"] not in known:
                    evidence.append(row)
                    known.add(row["id"])
            results.append(result)
        return results

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


def _validated(response: str, available: set[str]) -> AnalysisResult:
    """Parse one analysis result, rejecting any claim that cites unsupplied evidence."""

    try:
        result = AnalysisResult.model_validate(_parse_json(response))
    except (ValidationError, ValueError):
        raise ProviderError("analysis_invalid_response", retryable=False) from None
    if any(not set(claim.evidence_ids) <= available for claim in result.claims):
        raise ProviderError("analysis_invalid_citation", retryable=False)
    return result


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
