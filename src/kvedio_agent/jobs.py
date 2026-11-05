"""Bounded, restart-safe scheduling for independent segment provider calls."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, TypeVar

from kvedio_agent.providers import ProviderError
from kvedio_agent.storage import Store

T = TypeVar("T")


@dataclass(frozen=True)
class WorkUnit:
    """One durable, independently retryable item of stage work."""

    id: str
    start_ms: int
    end_ms: int
    payload: Any


@dataclass(frozen=True)
class WorkResult:
    """Outcome persisted for one unit, including a safe failure category."""

    unit_id: str
    value: Any = None
    error_category: str | None = None
    attempts: int = 0


class StageScheduler:
    """Run independent blocking work with concurrency, retry, and cancellation bounds."""

    def __init__(
        self,
        store: Store,
        max_workers: int,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.25,
        max_delay_seconds: float = 8.0,
        seed: int | None = None,
    ) -> None:
        if max_workers < 1 or max_attempts < 1:
            raise ValueError("worker and attempt limits must be positive")
        self.store = store
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self._random = random.Random(seed)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Stop scheduling new work and signal retry loops to stop."""

        self._cancel.set()

    def reset_cancel(self) -> None:
        """Clear cancellation before an explicitly resumed stage."""

        self._cancel.clear()

    def run(
        self,
        job_id: str,
        stage_name: str,
        units: Iterable[WorkUnit],
        worker: Callable[[WorkUnit], T],
        on_success: Callable[[WorkUnit, T], None],
    ) -> list[WorkResult]:
        """Execute units concurrently and commit successful outputs via callback."""

        unit_list = list(units)
        if not unit_list:
            return []
        stage_id = self.store.upsert_stage(job_id, stage_name, "running")
        raw_results: list[WorkResult] = []
        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="kvedio-stage"
        ) as pool:
            pending: dict[Future[WorkResult], WorkUnit] = {}
            unit_iter = iter(unit_list)

            def submit(unit: WorkUnit) -> None:
                self.store.set_segment_status(unit.id, "running")
                pending[pool.submit(self._run_unit, unit, worker)] = unit

            for _ in range(min(self.max_workers, len(unit_list))):
                if self._cancel.is_set():
                    break
                unit = next(unit_iter)
                submit(unit)
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    unit = pending.pop(future)
                    result = future.result()
                    raw_results.append(result)
                    try:
                        next_unit = next(unit_iter)
                    except StopIteration:
                        continue
                    if not self._cancel.is_set():
                        submit(next_unit)
        start_by_id = {unit.id: unit.start_ms for unit in unit_list}
        ordered = sorted(raw_results, key=lambda result: start_by_id[result.unit_id])
        unit_by_id = {unit.id: unit for unit in unit_list}
        committed_results: list[WorkResult] = []
        committed = 0
        failures = 0
        for result in ordered:
            unit = unit_by_id[result.unit_id]
            error_category = result.error_category
            if error_category is None:
                try:
                    on_success(unit, result.value)
                except Exception:
                    error_category = "stage_output_commit_failed"
            if error_category is None:
                self.store.set_segment_status(unit.id, "completed")
                committed += 1
                self.store.add_event(
                    job_id, "segment_completed", "Segment completed",
                    {"segment_id": unit.id, "start_ms": unit.start_ms,
                     "end_ms": unit.end_ms}, stage_id,
                )
            else:
                status = "cancelled" if error_category == "cancelled" else "retryable_failed"
                self.store.set_segment_status(unit.id, status)
                failures += status != "cancelled"
                self.store.add_event(
                    job_id, "segment_failed", "Segment processing failed",
                    {"segment_id": unit.id, "error_category": error_category}, stage_id,
                )
            committed_results.append(
                WorkResult(
                    unit.id,
                    result.value if error_category is None else None,
                    error_category,
                    result.attempts,
                )
            )
            self.store.upsert_stage(
                job_id, stage_name, "running", attempt=result.attempts,
                progress=(committed + failures) / len(unit_list),
            )
        cancelled = self._cancel.is_set() or len(raw_results) < len(unit_list)
        failed = failures > 0
        stage_status = "interrupted" if cancelled else "retryable_failed" if failed else "completed"
        self.store.upsert_stage(
            job_id, stage_name, stage_status,
            attempt=max((result.attempts for result in ordered), default=0),
            progress=committed / len(unit_list),
        )
        return committed_results

    def _run_unit(self, unit: WorkUnit, worker: Callable[[WorkUnit], T]) -> WorkResult:
        attempts = 0
        while attempts < self.max_attempts and not self._cancel.is_set():
            attempts += 1
            try:
                return WorkResult(unit.id, worker(unit), attempts=attempts)
            except ProviderError as exc:
                if not exc.retryable:
                    return WorkResult(unit.id, error_category=exc.category, attempts=attempts)
                if attempts >= self.max_attempts:
                    return WorkResult(unit.id, error_category=exc.category, attempts=attempts)
            except Exception:
                return WorkResult(unit.id, error_category="stage_internal_error", attempts=attempts)
            delay = min(self.base_delay_seconds * (2 ** (attempts - 1)), self.max_delay_seconds)
            time.sleep(self._random.uniform(delay * 0.75, delay * 1.25))
        return WorkResult(unit.id, error_category="cancelled", attempts=attempts)
