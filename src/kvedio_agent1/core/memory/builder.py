"""Building what a long recording needs before it can be asked about at scale.

Two products, both keyed to the transcript they came from: a coverage map (one line
per few minutes) and a vector per segment. Both are written incrementally, so an
interrupted build resumes at the first missing piece instead of paying the model
again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..domain import Coverage, Segment, VideoRecord
from ..llm import Chat, Embedder, Message
from ..prompts import render
from ..storage import Storage
from ..tasks import stop_if

MIN_SEGMENTS_FOR_COVERAGE = 40  # below this the raw samples already cover the whole recording
DEFAULT_BLOCK_MS = 180_000  # three minutes of talk per map line
VECTOR_BATCH = 32
Report = Callable[[float, str | None], None]


@dataclass(frozen=True)
class MemoryStats:
    """What a build produced, or found already in place."""

    blocks: int
    vectors: int


@dataclass(frozen=True)
class _Block:
    start_ms: int
    end_ms: int
    segments: tuple[Segment, ...]


class MemoryBuilder:
    """Produces the map and the vectors for one video, skipping whatever exists.

    Cache identity is the transcript key plus the model that produced the output:
    a new transcript or a new model replaces the memory instead of mixing two.
    Neither model is required — a missing one simply means that half is not built.
    """

    def __init__(
        self,
        store: Storage,
        chat: Chat | None,
        embedder: Embedder | None,
        *,
        model: str = "",
        block_ms: int = DEFAULT_BLOCK_MS,
        batch: int = VECTOR_BATCH,
    ) -> None:
        self._store = store
        self._chat = chat
        self._embedder = embedder
        self._model = model
        self._block_ms = block_ms
        self._batch = max(1, batch)

    def coverage_key(self, transcript_key: str) -> str:
        return _digest(f"{transcript_key}|{self._model}|{self._block_ms}")

    def vector_key(self, transcript_key: str) -> str:
        model = self._embedder.model if self._embedder is not None else ""
        return _digest(f"{transcript_key}|{model}")

    def current(self, video: VideoRecord) -> bool:
        """Whether every piece this builder is responsible for is already in place."""
        transcript = self._store.transcript(video.id)
        if transcript is None or not transcript.ready:
            return True  # nothing to build; the answer stage reports the missing transcript
        indexed = self._store.indexed_segments(video.id)
        if self._chat is not None and len(indexed) > MIN_SEGMENTS_FOR_COVERAGE:
            key = self.coverage_key(transcript.cache_key)
            blocks = plan_blocks([segment for _, segment in indexed], self._block_ms)
            if self._store.coverage_count(video.id, key) < len(blocks):
                return False
        if self._embedder is not None:
            key = self.vector_key(transcript.cache_key)
            if self._store.vector_count(video.id, key) < len(indexed):
                return False
        return True

    def build(
        self, video: VideoRecord, report: Report, *, cancelled: Callable[[], bool] | None = None
    ) -> MemoryStats:
        """Build whichever halves are configured, reporting progress across both."""
        stop_if(cancelled)
        transcript = self._store.transcript(video.id)
        if transcript is None or not transcript.ready:
            raise ValueError("没有可用的转录，无法建立记忆")
        indexed = self._store.indexed_segments(video.id)
        segments = [segment for _, segment in indexed]
        make_map = self._chat is not None and len(segments) > MIN_SEGMENTS_FOR_COVERAGE
        make_vectors = self._embedder is not None and bool(indexed)
        share = 0.4 if make_map and make_vectors else 1.0

        blocks = 0
        if make_map:
            blocks = self._build_coverage(
                video, transcript.cache_key, segments, _scaled(report, share), cancelled
            )
        else:
            blocks = len(self._store.coverage(video.id))

        vectors = 0
        if make_vectors:
            vectors = self._build_vectors(
                video,
                transcript.cache_key,
                indexed,
                _shifted(report, share if make_map else 0.0),
                cancelled,
            )
        else:
            vectors = len(self._store.vectorised(video.id))
        return MemoryStats(blocks=blocks, vectors=vectors)

    def _build_coverage(
        self,
        video: VideoRecord,
        transcript_key: str,
        segments: Sequence[Segment],
        report: Report,
        cancelled: Callable[[], bool] | None,
    ) -> int:
        key = self.coverage_key(transcript_key)
        self._store.prune_coverage(video.id, key)
        blocks = plan_blocks(segments, self._block_ms)
        done = self._store.coverage_positions(video.id, key)
        for position, block in enumerate(blocks):
            if position in done:
                continue
            stop_if(cancelled)
            gist = self._gist(block)
            self._store.save_coverage(
                video.id, key, position, Coverage(block.start_ms, block.end_ms, gist)
            )
            report((position + 1) / len(blocks), f"覆盖图 {position + 1}/{len(blocks)} 块")
        return len(blocks)

    def _gist(self, block: _Block) -> str:
        message = render(
            "coverage",
            start_ms=block.start_ms,
            end_ms=block.end_ms,
            segments=block.segments,
        )
        reply = self._chat.complete([Message(role="system", content=message)])
        return " ".join(reply.text.split())

    def _build_vectors(
        self,
        video: VideoRecord,
        transcript_key: str,
        indexed: Sequence[tuple[int, Segment]],
        report: Report,
        cancelled: Callable[[], bool] | None,
    ) -> int:
        key = self.vector_key(transcript_key)
        self._store.prune_vectors(video.id, key)
        done = self._store.vectorised_ids(video.id, key)
        pending = [
            (segment_id, segment) for segment_id, segment in indexed if segment_id not in done
        ]
        for offset in range(0, len(pending), self._batch):
            stop_if(cancelled)
            batch = pending[offset : offset + self._batch]
            vectors = self._embedder.embed([segment.text for _, segment in batch])
            self._store.save_vectors(
                video.id,
                key,
                [
                    (segment_id, segment.start_ms, np.asarray(vector, dtype=np.float32).tobytes())
                    for (segment_id, segment), vector in zip(batch, vectors, strict=True)
                ],
            )
            report(
                min(1.0, (offset + len(batch)) / max(1, len(pending))),
                f"向量 {offset + len(batch)}/{len(pending)} 段",
            )
        return len(indexed)


def plan_blocks(segments: Sequence[Segment], block_ms: int) -> list[_Block]:
    """Group consecutive segments into blocks of about ``block_ms``.

    A trailing block shorter than a third of the target is folded into its
    predecessor: a map line for ten seconds of speech is noise, not orientation.
    """
    if not segments:
        return []
    blocks: list[_Block] = []
    current: list[Segment] = []
    start = segments[0].start_ms
    for segment in segments:
        current.append(segment)
        if segment.end_ms - start >= block_ms:
            blocks.append(_Block(start, segment.end_ms, tuple(current)))
            current = []
            start = segment.end_ms
    if current:
        tail = _Block(start, current[-1].end_ms, tuple(current))
        if blocks and tail.end_ms - tail.start_ms < block_ms // 3:
            last = blocks.pop()
            blocks.append(_Block(last.start_ms, tail.end_ms, last.segments + tail.segments))
        else:
            blocks.append(tail)
    return blocks


def _digest(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _scaled(report: Report, share: float) -> Report:
    return lambda fraction, message: report(fraction * share, message)


def _shifted(report: Report, offset: float) -> Report:
    span = max(0.0, 1.0 - offset)
    return lambda fraction, message: report(offset + fraction * span, message)
