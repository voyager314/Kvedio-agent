"""Cutting prepared audio into slices a transcriber can take in one call.

A fixed interval would cut a sentence in half. Each boundary is moved to the
quietest window within a search radius of the target, which costs one pass over
the samples and no model call.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..domain import Chunk

WINDOW_MS = 100
MIN_TAIL_MS = 1_000


def plan_chunks(path: Path, chunk_ms: int, search_ms: int | None = None) -> list[Chunk]:
    """Plan contiguous slices covering the whole file.

    ``chunk_ms`` is the preferred length. ``search_ms`` is how far a boundary may
    move to land on silence, defaulting to a quarter of the chunk and never more
    than five seconds. The last slice is never shorter than ``MIN_TAIL_MS``: a
    sliver at the end is merged into its predecessor rather than sent as a
    request of its own.
    """
    if search_ms is None:
        search_ms = min(5_000, max(WINDOW_MS, chunk_ms // 4))
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        frames = audio.getnframes()
    total_ms = round(frames * 1000 / rate)
    if total_ms <= chunk_ms + MIN_TAIL_MS:
        return [Chunk(index=0, start_ms=0, end_ms=total_ms)]

    energy = _window_energy(path, rate)
    boundaries: list[int] = []
    target = chunk_ms
    while target + MIN_TAIL_MS < total_ms:
        boundary = _quietest(energy, target, search_ms)
        if boundaries:
            boundary = max(boundary, boundaries[-1] + MIN_TAIL_MS)
        if boundary >= total_ms - MIN_TAIL_MS:
            break
        boundaries.append(boundary)
        target = boundary + chunk_ms

    edges = [0, *boundaries, total_ms]
    return [
        Chunk(index=index, start_ms=edges[index], end_ms=edges[index + 1])
        for index in range(len(edges) - 1)
    ]


def slice_audio(source: Path, chunk: Chunk, target: Path) -> None:
    """Copy one slice's frames into its own WAV, without re-encoding."""
    with wave.open(str(source), "rb") as audio, wave.open(str(target), "wb") as out:
        rate = audio.getframerate()
        out.setnchannels(audio.getnchannels())
        out.setsampwidth(audio.getsampwidth())
        out.setframerate(rate)
        first = min(chunk.start_ms * rate // 1000, audio.getnframes())
        count = max(0, (chunk.end_ms - chunk.start_ms) * rate // 1000)
        audio.setpos(first)
        out.writeframes(audio.readframes(count))


def _window_energy(path: Path, rate: int) -> np.ndarray:
    """Root-mean-square level per WINDOW_MS, read block by block."""
    block = max(1, rate * WINDOW_MS // 1000)
    levels: list[float] = []
    with wave.open(str(path), "rb") as audio:
        while True:
            raw = audio.readframes(block)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            levels.append(float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0)
    return np.asarray(levels, dtype=np.float64)


def _quietest(energy: np.ndarray, target_ms: int, search_ms: int) -> int:
    """The end of the quietest window within the search radius, in milliseconds."""
    if energy.size == 0:
        return target_ms
    low = max(0, (target_ms - search_ms) // WINDOW_MS)
    high = min(energy.size - 1, (target_ms + search_ms) // WINDOW_MS)
    if high < low:
        return target_ms
    index = low + int(np.argmin(energy[low : high + 1]))
    return (index + 1) * WINDOW_MS
