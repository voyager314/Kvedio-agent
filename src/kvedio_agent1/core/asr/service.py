"""Transcription of one video's audio, resumable chunk by chunk.

Each chunk is transcribed on its own and written as soon as it returns, so a run
that stops halfway keeps every finished piece. The cache key names everything
that can change the words; a different key starts a new transcript rather than
mixing two.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..domain import Chunk, Segment, VideoRecord
from ..storage import Storage
from ..tasks import stop_if
from ..workspace import Workspace
from .chunking import plan_chunks, slice_audio

TRANSCRIBE_STAGE = "transcribe"
Transcribe = Callable[[Path], list[Segment]]
Report = Callable[[float, str | None], None]


class AsrStage:
    """Produces a video's transcript, reusing whatever a previous run finished.

    Cache identity is the video fingerprint plus the model, the address and the
    chunk length: a change in any of them starts a new transcript instead of
    mixing segments produced under different conditions. A second run over
    unchanged inputs costs no model call at all.

    Does not extract audio and does not render: the caller prepares the file and
    reads the result back from the store.
    """

    def __init__(
        self,
        store: Storage,
        workspace: Workspace,
        transcribe: Transcribe,
        *,
        model: str,
        address: str,
        chunk_ms: int,
        concurrency: int,
    ) -> None:
        self._store = store
        self._workspace = workspace
        self._transcribe_file = transcribe
        self._model = model
        self._address = address
        self._chunk_ms = chunk_ms
        self._concurrency = max(1, concurrency)

    def current(self, video: VideoRecord) -> bool:
        """Whether a finished transcript already exists under the key this stage would use."""
        transcript = self._store.transcript(video.id)
        return (
            transcript is not None
            and transcript.ready
            and transcript.cache_key == self._cache_key(video)
        )

    def run(
        self, video: VideoRecord, report: Report, *, cancelled: Callable[[], bool] | None = None
    ) -> int:
        """Transcribe every unfinished chunk and return the number of segments.

        ``report`` receives a fraction of the whole plan and a line naming what is
        happening, so a resumed run starts where the previous one stopped rather
        than at zero. ``cancelled`` is polled between waves; chunks already stored
        stay stored, so a cancelled run resumes where it stopped.
        """
        stop_if(cancelled)
        audio = self._workspace.audio(video.id)
        if not audio.exists():
            raise FileNotFoundError(f"缺少音轨文件：{audio}")
        chunks = plan_chunks(audio, self._chunk_ms)
        cache_key = self._cache_key(video)
        transcript = self._store.open_transcript(
            video.id, cache_key, self._model, TRANSCRIBE_STAGE
        )
        done = self._store.completed_chunks(video.id, TRANSCRIBE_STAGE, cache_key)
        pending = [chunk for chunk in chunks if chunk.index not in done]
        finished = len(chunks) - len(pending)
        if finished:
            report(finished / len(chunks), f"沿用已有 {finished}/{len(chunks)} 段")
        if pending:
            workers = min(self._concurrency, len(pending))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kvedio-asr") as pool:
                for start in range(0, len(pending), workers):
                    stop_if(cancelled)
                    wave = pending[start : start + workers]
                    futures = {
                        pool.submit(self._transcribe_chunk, audio, chunk, video.id): chunk
                        for chunk in wave
                    }
                    for future in as_completed(futures):
                        chunk = futures[future]
                        self._store.save_chunk(
                            transcript.id,
                            video.id,
                            TRANSCRIBE_STAGE,
                            cache_key,
                            chunk,
                            future.result(),
                        )
                        finished += 1
                        report(finished / len(chunks), f"转录 {finished}/{len(chunks)} 段")
        self._store.finish_transcript(transcript.id)
        stored = self._store.transcript(video.id)
        return stored.segment_count if stored is not None else 0

    def _transcribe_chunk(self, audio: Path, chunk: Chunk, video_id: int) -> list[Segment]:
        """Transcribe one slice and move its timestamps onto the video's timeline."""
        with NamedTemporaryFile(
            dir=self._workspace.scratch_dir(video_id), suffix=".wav", delete=False
        ) as handle:
            slice_path = Path(handle.name)
        try:
            slice_audio(audio, chunk, slice_path)
            reported = self._transcribe_file(slice_path)
        finally:
            slice_path.unlink(missing_ok=True)
        segments: list[Segment] = []
        for segment in reported:
            start = chunk.start_ms + segment.start_ms
            end = min(chunk.end_ms, chunk.start_ms + segment.end_ms)
            text = " ".join(segment.text.split())
            if text and end > start:
                segments.append(Segment(start_ms=start, end_ms=end, text=text))
        return segments

    def _cache_key(self, video: VideoRecord) -> str:
        """Identity of the transcript: which bytes, which model, which address, which slice length.

        The credential is deliberately absent: rotating a key does not change what
        the model would answer, and a secret has no business in a stored key.
        """
        material = f"{video.fingerprint}|{self._model}|{self._address}|{self._chunk_ms}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
