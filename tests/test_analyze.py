from __future__ import annotations

import json
from pathlib import Path

from kvedio_agent.analyze import AnalysisService
from kvedio_agent.context import EvidenceIndex
from kvedio_agent.storage import Store


class FakeModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.response)


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
