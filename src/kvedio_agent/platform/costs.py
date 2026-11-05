"""Provider pacing and durable per-job cost reservations."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import ClassVar
from weakref import WeakValueDictionary

from kvedio_agent.platform.settings import ModelEntry
from kvedio_agent.platform.storage import Store


class BudgetExceededError(ValueError):
    """Non-retryable reservation rejection caused by an enabled spend ceiling."""

    category = "budget_exceeded"


@dataclass
class _RateWindow:
    """Shared pacing state retained only while a gate uses the model."""

    lock: threading.Lock
    calls: deque[float]


@dataclass(frozen=True)
class UsageReservation:
    """Durable reservation associated with one idempotent provider call key."""

    id: str
    reused: bool


class ProviderGate:
    """Pace one model and reserve estimated job cost before external dispatch."""

    _rate_locks: ClassVar[WeakValueDictionary[str, _RateWindow]] = WeakValueDictionary()
    _rate_registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, store: Store, model: ModelEntry, max_job_cost_usd: float | None) -> None:
        self.store = store
        self.model = model
        self.max_job_cost_usd = max_job_cost_usd
        with self._rate_registry_lock:
            window = self._rate_locks.get(model.model)
            if window is None:
                window = _RateWindow(threading.Lock(), deque())
                self._rate_locks[model.model] = window
            self._rate_window = window

    def reserve(
        self, job_id: str, stage: str, call_key: str, estimate: float | None
    ) -> UsageReservation:
        """Wait for the model rate window, then reserve estimated cost durably."""

        try:
            usage_id, existing_status, _ = self.store.reserve_usage(
                job_id,
                stage,
                self.model.model,
                call_key,
                estimate,
                self.max_job_cost_usd,
            )
        except ValueError as exc:
            if "ceiling" in str(exc) or "cost is unknown" in str(exc):
                raise BudgetExceededError(str(exc)) from None
            raise
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

    def estimate_chat(self, text: str, output_tokens: int = 1024) -> float | None:
        if self.model.input_cost_per_million is None or self.model.output_cost_per_million is None:
            return None
        try:
            import litellm

            input_tokens = litellm.token_counter(model=self.model.model, text=text)
        except Exception:
            cjk = sum(
                "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"
                for char in text
            )
            input_tokens = math.ceil(cjk + (len(text) - cjk) / 3)
        input_tokens = math.ceil(input_tokens * 1.2)
        return (
            input_tokens * self.model.input_cost_per_million
            + output_tokens * self.model.output_cost_per_million
        ) / 1_000_000

    def estimate_vision(
        self, prompt: str, frame_count: int, output_tokens: int = 1024
    ) -> float | None:
        if frame_count < 0:
            raise ValueError("frame count cannot be negative")
        if self.model.input_cost_per_million is None or self.model.output_cost_per_million is None:
            return None
        text_cost = self.estimate_chat(prompt, output_tokens=0)
        image_tokens = math.ceil(frame_count * 1500 * 1.2)
        image_cost = image_tokens * self.model.input_cost_per_million / 1_000_000
        output_cost = output_tokens * self.model.output_cost_per_million / 1_000_000
        return (text_cost or 0) + image_cost + output_cost

    def _wait_for_rate(self) -> None:
        limit = self.model.requests_per_minute
        if limit is None:
            return
        while True:
            with self._rate_window.lock:
                now = time.monotonic()
                calls = self._rate_window.calls
                while calls and now - calls[0] >= 60:
                    calls.popleft()
                if len(calls) < limit:
                    calls.append(now)
                    return
                delay = max(0.01, 60 - (now - calls[0]))
            time.sleep(delay)


def call_key(stage: str, segment_id: str, model: ModelEntry) -> str:
    """Build a deterministic provider call key that excludes credentials."""

    return f"{stage}:{segment_id}:{model.model}"
