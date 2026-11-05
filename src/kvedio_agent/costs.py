"""Provider pacing and durable per-job cost reservations."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

from kvedio_agent.settings import ModelEntry
from kvedio_agent.storage import Store


@dataclass(frozen=True)
class UsageReservation:
    """Durable reservation associated with one idempotent provider call key."""

    id: str
    reused: bool


class ProviderGate:
    """Pace one model and reserve estimated job cost before external dispatch."""

    _rate_locks: ClassVar[dict[str, tuple[threading.Lock, deque[float]]]] = {}
    _rate_registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, store: Store, model: ModelEntry, max_job_cost_usd: float | None) -> None:
        self.store = store
        self.model = model
        self.max_job_cost_usd = max_job_cost_usd
        with self._rate_registry_lock:
            if model.model not in self._rate_locks:
                self._rate_locks[model.model] = (threading.Lock(), deque())
            self._lock, self._calls = self._rate_locks[model.model]

    def reserve(
        self, job_id: str, stage: str, call_key: str, estimate: float | None
    ) -> UsageReservation:
        """Wait for the model rate window, then reserve estimated cost durably."""

        usage_id, existing_status, _ = self.store.reserve_usage(
            job_id,
            stage,
            self.model.model,
            call_key,
            estimate,
            self.max_job_cost_usd,
        )
        if existing_status == "completed":
            raise ValueError("provider call is already complete; expected cached output evidence")
        if self.model.requests_per_minute:
            self._wait_for_rate()
        return UsageReservation(usage_id, False)

    def settle(
        self,
        reservation: UsageReservation,
        actual_cost_usd: float | None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        audio_seconds: float | None = None,
        failed: bool = False,
        retryable: bool = False,
    ) -> None:
        """Persist reported usage or mark the reservation cost uncertain."""

        self.store.finish_usage(
            reservation.id,
            actual_cost_usd,
            input_tokens,
            output_tokens,
            audio_seconds,
            failed,
            retryable,
        )

    def estimate_audio(self, duration_seconds: float) -> float | None:
        if self.model.audio_cost_per_minute is None:
            return None
        return duration_seconds / 60 * self.model.audio_cost_per_minute

    def estimate_chat(self, input_bytes: int, output_tokens: int = 1024) -> float | None:
        if self.model.input_cost_per_million is None or self.model.output_cost_per_million is None:
            return None
        input_tokens = input_bytes // 4
        return (
            input_tokens * self.model.input_cost_per_million
            + output_tokens * self.model.output_cost_per_million
        ) / 1_000_000

    def _wait_for_rate(self) -> None:
        limit = self.model.requests_per_minute
        if limit is None:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) < limit:
                    self._calls.append(now)
                    return
                delay = max(0.01, 60 - (now - self._calls[0]))
            time.sleep(delay)


def call_key(stage: str, segment_id: str, model: ModelEntry) -> str:
    """Build a deterministic provider call key that excludes credentials."""

    return f"{stage}:{segment_id}:{model.model}"
