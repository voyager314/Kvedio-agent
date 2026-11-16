"""Finding the stretches that answer a question, by wording and by meaning.

Two rankings are produced and merged rather than compared: a keyword match and a
cosine score live on different scales, and mixing them by value would let one drown
the other. Reciprocal rank fusion only uses order, which is what both can promise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..domain import Coverage, Segment, VideoRecord
from ..llm import Embedder
from ..storage import Storage

RRF_K = 60  # the usual damping constant; lower rewards the top hit more


class VideoMemory:
    """Read access to one video's memory: its map, and its segments by relevance."""

    def __init__(
        self, store: Storage, video: VideoRecord, embedder: Embedder | None = None
    ) -> None:
        self._store = store
        self._video = video
        self._embedder = embedder

    @property
    def mode(self) -> str:
        """Which rankings a recall will use, for the trace line."""
        if self._embedder is None:
            return "关键词检索"
        blocks, vectors = self._store.memory_summary(self._video.id)
        return "混合检索" if vectors else "关键词检索"

    def coverage(self) -> list[Coverage]:
        """The map built for this video, or an empty list when none was built."""
        return self._store.coverage(self._video.id)

    def recall(self, question: str, *, limit: int) -> list[Segment]:
        """The most relevant stretches for a question, best first."""
        keyword = self._keyword(question, limit * 2)
        if self._embedder is None:
            return keyword[:limit]
        return _fuse(keyword, self._vector(question, limit * 2))[:limit]

    def _keyword(self, question: str, limit: int) -> list[Segment]:
        terms = question.split()
        found = self._store.search_segments(self._video.id, terms, limit)
        if not found and len(terms) == 1 and len(question.strip()) > 6:
            found = self._store.search_segments(
                self._video.id, _bigrams(question), limit
            )
        return found

    def _vector(self, question: str, limit: int) -> list[Segment]:
        rows = self._store.vectorised(self._video.id)
        if not rows:
            return []
        query = np.asarray(self._embedder.embed([question])[0], dtype=np.float32)
        matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("向量维度与当前嵌入模型不符，请重建记忆")
        scores = matrix @ query / (
            np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query)) + 1e-8
        )
        order = np.argsort(-scores)[:limit]
        return [rows[int(index)][0] for index in order]


def _fuse(*rankings: Sequence[Segment]) -> list[Segment]:
    """Merge rankings by reciprocal rank, keeping the first copy of each stretch."""
    scores: dict[tuple[int, int], float] = {}
    segments: dict[tuple[int, int], Segment] = {}
    for ranking in rankings:
        for rank, segment in enumerate(ranking, start=1):
            key = (segment.start_ms, segment.end_ms)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            segments.setdefault(key, segment)
    ordered = sorted(scores, key=lambda key: -scores[key])
    return [segments[key] for key in ordered]


def _bigrams(text: str) -> list[str]:
    """Two-character windows, deduplicated: a stand-in for a Chinese tokenizer.

    A sentence searched as one substring matches nothing, because the transcript
    has no whitespace to split on. Overlapping windows give the database something
    to match without pretending to know where words begin.
    """
    cleaned = "".join(
        character
        for character in text
        if not character.isspace() and character not in "，。！？、,.!?：:；;\"'“”‘’（）()"
    )
    return list(dict.fromkeys(cleaned[index : index + 2] for index in range(len(cleaned) - 1)))
