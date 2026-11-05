"""Deterministic synthetic benchmark for bounded stage scheduling."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured wall time and worker utilization for one scheduling mode."""

    mode: str  # "serial" or "concurrent".
    elapsed_seconds: float  # Monotonic wall time spent processing all units.
    units: int  # Number of synthetic units processed; positive.
    workers: int  # Worker limit used by the mode; positive.


def run_benchmark(
    units: int = 8, workers: int = 2, unit_seconds: float = 0.01,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Measure serial and bounded concurrent synthetic work without external services."""

    if units < 1 or workers < 1 or unit_seconds < 0:
        raise ValueError("units and workers must be positive; duration cannot be negative")

    def work(_: int) -> None:
        sleep(unit_seconds)

    serial_start = time.perf_counter()
    for index in range(units):
        work(index)
    serial = BenchmarkResult("serial", time.perf_counter() - serial_start, units, 1)
    concurrent_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(units)))
    concurrent = BenchmarkResult(
        "concurrent", time.perf_counter() - concurrent_start, units, workers
    )
    return serial, concurrent
