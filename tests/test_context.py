from __future__ import annotations

import json
from pathlib import Path

from kvedio_agent.pipeline.context import (
    ContextAssembler,
    ContextBudget,
    EvidenceIndex,
    Scratchpad,
    estimate_tokens,
)
from kvedio_agent.platform.storage import Store


def _job(store: Store, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    return store.create_job(source_id, "explain the launch", {})


def _evidence(identifier: str, start_ms: int, end_ms: int, content: str) -> dict[str, object]:
    return {
        "id": identifier, "start_ms": start_ms, "end_ms": end_ms,
        "modality": "transcript", "content": content,
    }


def _chapter(identifier: str, start_ms: int, title: str, summary: str) -> dict[str, object]:
    return {
        "id": identifier, "start_ms": start_ms, "end_ms": start_ms + 1000,
        "modality": "timeline_summary",
        "content": json.dumps({"title": title, "summary": summary}),
    }


def _answer_turn(text: str, evidence_ids: list[str]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": json.dumps({
            "objective": "explain the launch",
            "plan": ["retrieve", "answer"],
            "claims": [{"text": text, "evidence_ids": evidence_ids, "confidence": 0.9}],
            "uncertainties": ["none recorded"],
            "quality_assessment": "evidence covers the requested range",
            "stage_trace": ["transcribe", "answer"],
        }),
    }


def test_scratchpad_roundtrips_as_compact_checkpoint(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    pad = Scratchpad(objective="explain the launch", current_stage="timeline")
    pad.completed_ranges.append((0, 60_000))
    pad.transcript_facts.append({"text": "Launch approved", "evidence_id": "e1"})
    pad.visual_facts.append({"text": "Green status slide", "evidence_id": "e2"})
    pad.cross_modal_facts.append({"text": "Speech matches slide", "evidence_ids": ["e1", "e2"]})
    pad.save(store, job_id)
    restored = Scratchpad.from_dict(store.checkpoint(job_id) or {})
    assert restored.completed_ranges == [(0, 60_000)]
    assert restored.cross_modal_facts[0]["evidence_ids"] == ["e1", "e2"]


def test_token_estimate_charges_wide_characters_more_than_ascii() -> None:
    assert estimate_tokens("发射窗口确认") == 6
    assert estimate_tokens("launch") == 2
    assert estimate_tokens("") == 0


def test_evidence_retrieval_keeps_store_ranking_and_whole_rows(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    first = store.add_evidence(job_id, "transcript", 0, 1000, {}, "launch date announced")
    second = store.add_evidence(job_id, "visual", 900, 1500, {}, "launch date shown on slide")
    store.add_evidence(job_id, "transcript", 3000, 4000, {}, "unrelated closing remarks")
    index = EvidenceIndex(store, ContextBudget(max_evidence_items=2))
    result = index.retrieve(job_id, "launch date")
    assert len(result) == 2
    assert result[0]["content"] == "launch date announced"
    assert [item["id"] for item in index.retrieve(job_id, "", (800, 1100))] == [first, second]


def test_assembled_evidence_rows_are_admitted_whole_or_not_at_all() -> None:
    rows = [_evidence(f"e{index}", index * 1000, index * 1000 + 999, "launch " * 40)
            for index in range(6)]
    assembled = ContextAssembler(ContextBudget(total_tokens=200, evidence_tokens=200)).assemble(
        evidence=rows
    )
    assert assembled.tokens <= 200
    assert 0 < len(assembled.evidence_ids) < len(rows)
    for identifier in assembled.evidence_ids:
        assert f"[{identifier}]" in assembled.text
    for row in rows:
        if row["id"] in assembled.evidence_ids:
            assert str(row["content"]) in assembled.text
    assert any("did not fit" in note for note in assembled.notes)
    assert "Record completeness" in assembled.text


def test_chapter_map_outranks_evidence_and_degrades_to_titles() -> None:
    chapters = [_chapter(f"c{index}", index * 1000, f"Chapter {index}", "detail " * 30)
                for index in range(4)]
    evidence = [_evidence("e1", 0, 999, "launch " * 40)]
    assembled = ContextAssembler(
        ContextBudget(total_tokens=90, chapter_tokens=90, evidence_tokens=90)
    ).assemble(chapters=chapters, evidence=evidence)
    assert "Chapters:" in assembled.text
    assert "Chapter 0" in assembled.text
    assert "detail detail" not in assembled.text
    assert assembled.evidence_ids == ()
    assert any("titles" in note for note in assembled.notes)


def test_conversation_compresses_older_turns_and_keeps_their_citations() -> None:
    history: list[dict[str, object]] = []
    for index in range(20):
        history.append({"role": "user", "content": f"question {index}"})
        history.append(_answer_turn(f"conclusion {index}", [f"e{index}"]))
    budget = ContextBudget(total_tokens=600, history_tokens=600, verbatim_turns=2)
    assembled = ContextAssembler(budget).assemble(history=history)
    assert assembled.tokens <= 600
    assert "conclusion 19" in assembled.text
    assert "cited: e0" in assembled.text or "earlier turns were omitted" in str(assembled.notes)
    assert assembled.text.count("stage_trace") <= budget.verbatim_turns
    assert any("compressed to one line" in note for note in assembled.notes)


def test_assembled_size_does_not_grow_with_conversation_length() -> None:
    budget = ContextBudget(total_tokens=800, history_tokens=400)
    assembler = ContextAssembler(budget)
    long = [_answer_turn("conclusion", ["e1"]) for _ in range(400)]
    saturated = assembler.assemble(history=long).tokens
    assert saturated <= budget.total_tokens
    # Quadrupling the conversation may only widen the digits of the omission count.
    assert assembler.assemble(history=long * 4).tokens - saturated <= 4


def test_tool_results_yield_the_budget_to_evidence() -> None:
    evidence = [_evidence("e1", 0, 999, "launch window confirmed")]
    tool_results = [{"evidence": [{"evidence_id": "e9", "content": "spare " * 80}]}]
    assembled = ContextAssembler(
        ContextBudget(total_tokens=60, evidence_tokens=60, tool_tokens=60)
    ).assemble(evidence=evidence, tool_results=tool_results)
    assert assembled.evidence_ids == ("e1",)
    assert "Tool results:" not in assembled.text
    assert any("tool results did not fit" in note for note in assembled.notes)
