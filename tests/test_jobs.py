from __future__ import annotations

import threading
import time
from pathlib import Path

from kvedio_agent.jobs import StageScheduler, WorkUnit
from kvedio_agent.providers import ProviderError
from kvedio_agent.storage import Store


def _job(store: Store, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_id = store.register_source(source, 6, source.stat().st_mtime_ns, "digest")
    return store.create_job(source_id, "objective", {})


def test_scheduler_limits_concurrency_and_returns_time_order(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    units = [WorkUnit(str(i), i * 1000, (i + 1) * 1000, i) for i in range(8)]
    for unit in units:
        store.add_segment(job_id, "asr", unit.start_ms, unit.end_ms)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker(unit: WorkUnit) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return int(unit.payload)

    completed: list[int] = []
    scheduler = StageScheduler(store, max_workers=3, seed=1)
    results = scheduler.run(
        job_id, "asr", units, worker, lambda unit, value: completed.append(value)
    )
    assert max_active == 3
    assert [result.unit_id for result in results] == [str(i) for i in range(8)]
    assert completed == list(range(8))


def test_scheduler_retries_only_retryable_provider_errors(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    segment = store.add_segment(job_id, "asr", 0, 1000)
    calls = 0

    def worker(unit: WorkUnit) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("provider_timeout", retryable=True)
        return "done"

    scheduler = StageScheduler(store, 1, max_attempts=3, base_delay_seconds=0, seed=1)
    result = scheduler.run(
        job_id, "asr", [WorkUnit(segment, 0, 1000, None)], worker, lambda unit, value: None
    )[0]
    assert calls == 2
    assert result.attempts == 2
    assert result.error_category is None


def test_scheduler_does_not_retry_permanent_failure(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    segment = store.add_segment(job_id, "vision", 0, 1000)
    calls = 0

    def worker(unit: WorkUnit) -> None:
        nonlocal calls
        calls += 1
        raise ProviderError("provider_request_rejected", retryable=False)

    scheduler = StageScheduler(store, 1, max_attempts=4)
    result = scheduler.run(
        job_id, "vision", [WorkUnit(segment, 0, 1000, None)], worker, lambda unit, value: None
    )[0]
    assert calls == 1
    assert result.error_category == "provider_request_rejected"


def test_segment_stays_retryable_when_output_commit_fails(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    job_id = _job(store, tmp_path)
    segment_id = store.add_segment(job_id, "asr", 0, 1000)
    scheduler = StageScheduler(store, 1, base_delay_seconds=0, seed=1)

    def fail_commit(unit: WorkUnit, value: str) -> None:
        raise OSError("simulated process interruption")

    result = scheduler.run(
        job_id,
        "transcribe",
        [WorkUnit(segment_id, 0, 1000, None)],
        lambda unit: "provider result",
        fail_commit,
    )

    assert result[0].error_category == "stage_output_commit_failed"
    assert store.pending_segments(job_id, "asr")[0]["status"] == "retryable_failed"
    assert store.job_events(job_id)[-1]["data_json"].find(
        "stage_output_commit_failed"
    ) >= 0
