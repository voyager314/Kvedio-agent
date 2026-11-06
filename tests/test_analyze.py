from __future__ import annotations

import json
from pathlib import Path

from kvedio_agent.pipeline.analyze import AnalysisService
from kvedio_agent.pipeline.context import EvidenceIndex
from kvedio_agent.platform.providers import Completion
from kvedio_agent.platform.storage import Store


class FakeModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(
        self, system: str, prompt: str,
        tools: list[dict[str, object]] | None = None,
    ) -> Completion:
        self.prompts.append(prompt)
        return Completion(json.dumps(self.response))


def _job(store: Store, tmp_path: Path) -> tuple[str, str, str]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    job_id = store.create_job(source_id, "explain launch", {})
    evidence_id = store.add_evidence(job_id, "transcript", 1000, 2000, {}, "launch approved")
    conversation_id = store.create_conversation(job_id, "Analysis")
    return job_id, evidence_id, conversation_id


def test_analysis_validates_claim_citations(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id, evidence_id, _ = _job(store, tmp_path)
    model = FakeModel({
        "objective": "explain launch",
        "plan": ["inspect evidence"],
        "claims": [{"text": "Launch approved", "evidence_ids": [evidence_id], "confidence": 0.9}],
        "uncertainties": [],
        "quality_assessment": "One transcript item supports the claim.",
    })
    result = AnalysisService(store, EvidenceIndex(store), model).analyze(job_id, "launch")
    assert result.claims[0].evidence_ids == [evidence_id]
    assert "launch approved" in model.prompts[0]


def test_initial_analysis_can_read_referenced_local_file(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id, evidence_id, _ = _job(store, tmp_path)
    reference = tmp_path / "context notes.txt"
    reference.write_text("The project name is Aurora.", encoding="utf-8")

    class FileModel:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def complete(
            self, system: str, prompt: str,
            tools: list[dict[str, object]] | None = None,
        ) -> Completion:
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                return Completion(
                    "",
                    tool_calls=(
                        {
                            "id": "read-1", "name": "read_file",
                            "arguments": json.dumps({"path": str(reference)}),
                        },
                    ),
                )
            return Completion(json.dumps({
                "objective": "Use context notes",
                "plan": ["read notes", "check video evidence"],
                "claims": [{
                    "text": "The transcript records approval.",
                    "evidence_ids": [evidence_id], "confidence": 0.9,
                }],
                "uncertainties": [],
                "quality_assessment": "Video evidence supports the claim.",
            }))

    model = FileModel()
    result = AnalysisService(store, EvidenceIndex(store), model).analyze(
        job_id, "Use context notes to explain launch approval"
    )

    assert result.claims[0].evidence_ids == [evidence_id]
    assert "The project name is Aurora." in model.prompts[1]


def test_locate_returns_original_timestamped_evidence(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id, evidence_id, _ = _job(store, tmp_path)
    result = AnalysisService(store, EvidenceIndex(store), FakeModel({})).locate(
        job_id, 1500, 1600
    )
    assert result.matches[0].evidence_id == evidence_id
    assert result.matches[0].start_ms == 1000


def test_follow_up_persists_turns_for_restart(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id, evidence_id, conversation_id = _job(store, tmp_path)
    model = FakeModel({
        "objective": "What next?",
        "plan": [],
        "claims": [
            {"text": "No next step stated", "evidence_ids": [evidence_id], "confidence": 0.6}
        ],
        "uncertainties": ["No later evidence was supplied."],
        "quality_assessment": "Limited context.",
    })
    service = AnalysisService(store, EvidenceIndex(store), model)
    service.follow_up(job_id, conversation_id, "What next?")
    messages = Store(tmp_path / "workspace").conversation_messages(conversation_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_follow_up_executes_bounded_read_only_tool_call(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id, evidence_id, conversation_id = _job(store, tmp_path)

    class ToolModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, system: str, prompt: str,
            tools: list[dict[str, object]] | None = None,
        ) -> Completion:
            self.calls += 1
            if self.calls == 1:
                return Completion(
                    "",
                    tool_calls=(
                        {
                            "id": "call-1", "name": "search_evidence",
                            "arguments": '{"query":"launch"}',
                        },
                    ),
                )
            return Completion(json.dumps({
                "objective": "What next?",
                "plan": [],
                "claims": [{
                    "text": "Launch approved", "evidence_ids": [evidence_id],
                    "confidence": 0.8,
                }],
                "uncertainties": [],
                "quality_assessment": "Supported.",
            }))

    result = AnalysisService(store, EvidenceIndex(store), ToolModel()).follow_up(
        job_id, conversation_id, "What next?"
    )
    assert result.claims[0].evidence_ids == [evidence_id]


def test_segment_and_timeline_agents_persist_only_valid_evidence_links(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "workspace")
    job_id, evidence_id, _ = _job(store, tmp_path)

    class SegmentTimelineModel:
        def __init__(self) -> None:
            self.responses = [
                {"summary": "Launch was approved", "evidence_ids": [evidence_id],
                 "entities": ["Project A"], "uncertainty": None},
                {"chapters": [{"start_ms": 1000, "end_ms": 2000, "title": "Approval",
                               "summary": "The launch was approved.",
                               "evidence_ids": ["summary-ref"]}]},
            ]

        def complete(
            self, system: str, prompt: str,
            tools: list[dict[str, object]] | None = None,
        ) -> Completion:
            response = self.responses.pop(0)
            if "chapters" in response:
                summary = store.evidence_for_modality(job_id, "segment_summary")[0]
                response["chapters"][0]["evidence_ids"] = [summary["id"]]
            return Completion(json.dumps(response))

    service = AnalysisService(store, EvidenceIndex(store), SegmentTimelineModel())
    fact = service.understand_segment(job_id, 1000, 2000)
    chapters = service.synthesize_timeline(job_id)
    assert fact.evidence_ids == (evidence_id,)
    assert chapters[0].title == "Approval"
    assert store.evidence_for_modality(job_id, "timeline_summary")
