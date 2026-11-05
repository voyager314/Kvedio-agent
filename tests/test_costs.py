from __future__ import annotations

from pathlib import Path

import pytest

from kvedio_agent.costs import ProviderGate
from kvedio_agent.settings import ModelEntry
from kvedio_agent.storage import Store


def _job(store: Store, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    return store.create_job(source_id, "objective", {})


def test_unknown_cost_is_rejected_when_budget_is_enabled(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    model = ModelEntry(model="provider/model", roles=["asr"], api_key_env="TOKEN")
    gate = ProviderGate(store, model, max_job_cost_usd=1)
    with pytest.raises(ValueError, match="unknown"):
        gate.reserve(_job(store, tmp_path), "asr", "segment-1", None)


def test_cost_reservation_and_settlement_are_idempotent(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    model = ModelEntry(
        model="provider/model", roles=["chat"], api_key_env="TOKEN",
        input_cost_per_million=1, output_cost_per_million=2,
    )
    gate = ProviderGate(store, model, max_job_cost_usd=1)
    job_id = _job(store, tmp_path)
    estimate = gate.estimate_chat(4000, 1000)
    reservation = gate.reserve(job_id, "chat", "question-1", estimate)
    gate.settle(reservation, 0.002, input_tokens=1000, output_tokens=500)
    with pytest.raises(ValueError, match="already complete"):
        gate.reserve(job_id, "chat", "question-1", estimate)
    summary = store.usage_summary(job_id)
    assert summary["calls"] == 1
    assert summary["known_cost_usd"] == 0.002


def test_failed_call_remains_uncertain_and_blocks_duplicate_dispatch(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    model = ModelEntry(
        model="provider/model", roles=["chat"], api_key_env="TOKEN",
        input_cost_per_million=1, output_cost_per_million=2,
    )
    gate = ProviderGate(store, model, max_job_cost_usd=1)
    job_id = _job(store, tmp_path)
    reservation = gate.reserve(job_id, "chat", "failed-call", 0.1)
    gate.settle(reservation, None, failed=True)
    with pytest.raises(ValueError, match="uncertain"):
        gate.reserve(job_id, "chat", "failed-call", 0.1)
    assert store.usage_summary(job_id)["unknown_calls"] == 1


def test_retryable_failure_creates_a_new_attempt(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    model = ModelEntry(
        model="provider/model", roles=["chat"], api_key_env="TOKEN",
        input_cost_per_million=1, output_cost_per_million=2,
    )
    gate = ProviderGate(store, model, max_job_cost_usd=1)
    job_id = _job(store, tmp_path)
    first = gate.reserve(job_id, "chat", "retry-call", 0.1)
    gate.settle(first, None, failed=True, retryable=True)
    second = gate.reserve(job_id, "chat", "retry-call", 0.1)
    gate.settle(second, 0.1)
    assert store.usage_summary(job_id)["calls"] == 2
