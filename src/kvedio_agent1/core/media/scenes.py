"""Cutting a recording into scenes, and taking one frame out of each.

A scene boundary is where the picture stops looking like the frame before it.
That is measured on a 32×32 grey signature: cheap to compute, indifferent to
compression noise, and enough to find a cut or a slide change. Sampling seeks to
one moment per second instead of decoding the whole file, so an hour of video
costs hundreds of seeks rather than a hundred thousand decodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

SIGNATURE_SIZE = 32
SAMPLE_MS = 1_000
MIN_SCENE_MS = 2_000
MAX_SCENE_MS = 90_000  # a fixed camera still needs frames along its length
THRESHOLD = 0.12  # mean absolute difference on a 0..1 grey signature
FRAME_WIDTH = 512


@dataclass(frozen=True)
class DetectedScene:
    """One stretch of the recording that looks like a single shot."""

    start_ms: int
    end_ms: int


def detect_scenes(
    path: Path,
    *,
    sample_ms: int = SAMPLE_MS,
    min_ms: int = MIN_SCENE_MS,
    max_ms: int = MAX_SCENE_MS,
    threshold: float = THRESHOLD,
    progress: Callable[[float], None] | None = None,
    check: Callable[[], None] | None = None,
) -> list[DetectedScene]:
    """Split a video into scenes between ``min_ms`` and ``max_ms`` long.

    A cut closer than ``min_ms`` to the previous one is ignored: flicker and
    quick camera moves would otherwise produce a scene per half second, and every
    scene costs one model call later.

    A scene longer than ``max_ms`` is divided into equal parts. A fixed camera
    never cuts, yet what it points at changes — a lecture's slides, a dashboard's
    numbers — and one frame for ten minutes describes none of it.

    ``progress`` receives a fraction of the sampling pass about a hundred times.
    ``check`` is called once per sampled moment; it may raise to stop the pass,
    which is how cancellation and a stage budget reach a long seek loop.
    """
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"没有视频轨：{path.name}")
        duration_ms = _duration_ms(container)
        samples = max(1, duration_ms // sample_ms)
        stride = max(1, samples // 100)
        scenes: list[DetectedScene] = []
        previous: np.ndarray | None = None
        start = 0
        moment = 0
        index = 0
        while moment < duration_ms:
            if check is not None:
                check()
            signature = _signature_at(container, stream, moment)
            if signature is not None:
                if previous is not None:
                    difference = float(np.mean(np.abs(signature - previous)))
                    if difference >= threshold and moment - start >= min_ms:
                        scenes.append(DetectedScene(start, moment))
                        start = moment
                previous = signature
            moment += sample_ms
            index += 1
            if progress is not None and index % stride == 0:
                progress(min(1.0, moment / max(1, duration_ms)))
        if progress is not None:
            progress(1.0)
    if not scenes:
        return _split(0, max(duration_ms, 1), max_ms)
    if duration_ms - start < min_ms:
        last = scenes.pop()
        scenes.append(DetectedScene(last.start_ms, duration_ms))
    else:
        scenes.append(DetectedScene(start, duration_ms))
    return [piece for scene in scenes for piece in _split(scene.start_ms, scene.end_ms, max_ms)]


def _split(start_ms: int, end_ms: int, max_ms: int) -> list[DetectedScene]:
    """Divide a stretch into equal parts no longer than ``max_ms``."""
    if max_ms <= 0 or end_ms - start_ms <= max_ms:
        return [DetectedScene(start_ms, end_ms)]
    parts = (end_ms - start_ms + max_ms - 1) // max_ms
    step = (end_ms - start_ms) / parts
    return [
        DetectedScene(round(start_ms + index * step), round(start_ms + (index + 1) * step))
        for index in range(parts)
    ]


def grab_frame(
    path: Path, moment_ms: int, target: Path, *, width: int = FRAME_WIDTH
) -> bool:
    """Write one JPEG taken at ``moment_ms``; false when the moment cannot be decoded."""
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            return False
        frame = _frame_at(container, stream, moment_ms)
        if frame is None:
            return False
        image = frame.to_image()
        if image.width > width:
            image = image.resize((width, max(1, round(image.height * width / image.width))))
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=80)
    return True


def _duration_ms(container: av.container.InputContainer) -> int:
    if container.duration:
        return int(container.duration / 1000)
    for stream in container.streams:
        if stream.type == "video" and stream.duration and stream.time_base:
            return int(stream.duration * stream.time_base * 1000)
    return 0


def _signature_at(
    container: av.container.InputContainer, stream: av.video.stream.VideoStream, moment_ms: int
) -> np.ndarray | None:
    frame = _frame_at(container, stream, moment_ms)
    if frame is None:
        return None
    return _signature(frame)


def _frame_at(
    container: av.container.InputContainer, stream: av.video.stream.VideoStream, moment_ms: int
) -> av.VideoFrame | None:
    """Decode the first frame at or after a moment, seeking to the keyframe before it."""
    if stream.time_base is None:
        return None
    offset = int(moment_ms / 1000 / float(stream.time_base))
    try:
        container.seek(max(0, offset), stream=stream, backward=True)
    except (av.error.FFmpegError, ValueError):
        return None
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if int(frame.pts * stream.time_base * 1000) >= moment_ms:
            return frame
    return None


def _signature(frame: av.VideoFrame) -> np.ndarray:
    array = frame.to_ndarray(format="gray")
    height, width = array.shape[:2]
    step_y = max(1, height // SIGNATURE_SIZE)
    step_x = max(1, width // SIGNATURE_SIZE)
    small = array[::step_y, ::step_x][:SIGNATURE_SIZE, :SIGNATURE_SIZE]
    return small.astype(np.float32) / 255.0
