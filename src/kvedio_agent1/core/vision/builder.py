"""Describing what the recording shows, one scene at a time.

Scene boundaries come from the picture, descriptions from a vision model. Frames
live in the workspace so a description can be checked later, and both the frame
and the description are keyed to the video bytes and the model: a new model or a
new file replaces them instead of mixing two readings of the same shot.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..domain import Scene, Segment, VideoRecord
from ..llm import Vision
from ..media.scenes import MAX_SCENE_MS, DetectedScene, detect_scenes, grab_frame
from ..prompts import render
from ..storage import Storage
from ..tasks import stop_if
from ..workspace import Workspace

MAX_SCENES = 80  # beyond this the descriptions cost more than they explain
DETECT_SHARE = 0.2  # the sampling pass is the first fifth of the progress bar
Report = Callable[[float, str | None], None]


@dataclass(frozen=True)
class VisionStats:
    """How many scenes the recording was cut into, and how many are described."""

    scenes: int
    described: int


class VisionBuilder:
    """Cuts a recording into scenes and writes one description per scene."""

    def __init__(
        self,
        store: Storage,
        vision: Vision | None,
        workspace: Workspace,
        *,
        sample_ms: int = 1_000,
        min_ms: int = 2_000,
        max_ms: int = MAX_SCENE_MS,
        threshold: float = 0.12,
        max_scenes: int = MAX_SCENES,
    ) -> None:
        self._store = store
        self._vision = vision
        self._workspace = workspace
        self._sample_ms = sample_ms
        self._min_ms = min_ms
        self._max_ms = max_ms
        self._threshold = threshold
        self._max_scenes = max(1, max_scenes)

    def key(self, video: VideoRecord) -> str:
        model = self._vision.model if self._vision is not None else ""
        material = (
            f"{video.fingerprint}|{model}|{self._sample_ms}|{self._min_ms}"
            f"|{self._max_ms}|{self._threshold}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def current(self, video: VideoRecord) -> bool:
        """Whether the last completed run was made from this file and this model."""
        if self._vision is None:
            return True
        run = self._store.scene_run(video.id)
        return run is not None and run[0] == self.key(video)

    def build(
        self, video: VideoRecord, report: Report, *, cancelled: Callable[[], bool] | None = None
    ) -> VisionStats:
        """Detect scenes, then describe the ones that have no description yet."""
        stop_if(cancelled)
        if self._vision is None:
            return VisionStats(scenes=0, described=0)
        detected = _merge(
            detect_scenes(
                video.path,
                sample_ms=self._sample_ms,
                min_ms=self._min_ms,
                max_ms=self._max_ms,
                threshold=self._threshold,
                progress=lambda fraction: report(fraction * DETECT_SHARE, "识别场景"),
                check=lambda: stop_if(cancelled),
            ),
            self._max_scenes,
        )
        key = self.key(video)
        self._store.prune_scenes(video.id, key)
        done = self._store.scene_positions(video.id, key)
        described = len(done)
        # The picture branch does not wait for transcription: these segments may
        # still be growing while the scenes are described. They are context for
        # the description, not a quotation, so an incomplete set is usable.
        segments = self._store.transcript_segments(video.id)
        for position, scene in enumerate(detected):
            if position in done:
                continue
            stop_if(cancelled)
            frame = self._workspace.video_dir(video.id) / "scenes" / f"{position}.jpg"
            if not frame.exists() and not grab_frame(
                video.path, (scene.start_ms + scene.end_ms) // 2, frame
            ):
                report(_described(position, len(detected)), f"场景 {position + 1}：取帧失败")
                continue
            description = self._describe(scene, frame, segments)
            self._store.save_scene(
                video.id,
                key,
                position,
                Scene(
                    start_ms=scene.start_ms,
                    end_ms=scene.end_ms,
                    description=description,
                    frame_path=str(frame),
                ),
            )
            described += 1
            report(_described(position, len(detected)), f"场景 {position + 1}/{len(detected)}")
        self._store.save_scene_run(video.id, key, len(detected))
        return VisionStats(scenes=len(detected), described=described)

    def _describe(
        self, scene: DetectedScene, frame: Path, segments: list[Segment]
    ) -> str:
        spoken = [
            segment
            for segment in segments
            if segment.end_ms > scene.start_ms and segment.start_ms < scene.end_ms
        ][:8]
        prompt = render(
            "scene",
            start_ms=scene.start_ms,
            end_ms=scene.end_ms,
            segments=spoken,
        )
        return self._vision.describe(frame, prompt)


def _described(position: int, total: int) -> float:
    """Progress after describing one scene, leaving the sampling pass its share."""
    return DETECT_SHARE + (1.0 - DETECT_SHARE) * (position + 1) / total


def _merge(scenes: list[DetectedScene], limit: int) -> list[DetectedScene]:
    """Fold neighbours together until there are at most ``limit`` scenes.

    Called only when a recording is cut into more shots than the budget pays for;
    the boundaries that survive are the ones a reader would notice anyway.
    """
    if len(scenes) <= limit:
        return scenes
    group = (len(scenes) + limit - 1) // limit
    merged: list[DetectedScene] = []
    for start in range(0, len(scenes), group):
        batch = scenes[start : start + group]
        merged.append(DetectedScene(batch[0].start_ms, batch[-1].end_ms))
    return merged
