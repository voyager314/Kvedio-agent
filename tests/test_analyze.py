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

    def complete(self, system: str, prompt: str) -> Completion:
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

        def complete(self, system: str, prompt: str) -> Completion:
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
