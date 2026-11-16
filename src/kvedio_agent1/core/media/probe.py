"""What a media file declares about itself, read through the decoder in the wheel.

PyAV carries its own FFmpeg build, so no system binary is required and the
probe behaves the same on every machine.
"""

from __future__ import annotations

from pathlib import Path

import av

from ..domain import MediaInfo


class ProbeError(Exception):
    """The file cannot be treated as media, or does not declare a usable duration."""


def decoder_version() -> str:
    """Decoder build in use, so two machines can be compared from one command."""
    return f"PyAV {av.__version__} / FFmpeg {av.ffmpeg_version_info}"


def probe(path: Path) -> MediaInfo:
    """Read the stream layout and the duration.

    Raises ``ProbeError`` when the container cannot be opened or reports no
    duration: an unknown length cannot be placed on a timeline, and every later
    stage segments by time.
    """
    try:
        container = av.open(str(path))
    except av.FFmpegError as exc:
        raise ProbeError(f"无法读取媒体文件：{path.name}（{exc}）") from None
    with container:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        duration_us = _duration_us(container, video)
        if duration_us is None:
            raise ProbeError(f"媒体文件未声明时长：{path.name}")
        video_codec = video.codec_context if video is not None else None
        audio_codec = audio.codec_context if audio is not None else None
        return MediaInfo(
            duration_ms=int(duration_us // 1000),
            container=container.format.name,
            video_codec=video_codec.name if video_codec is not None else None,
            width=video_codec.width if video_codec is not None else None,
            height=video_codec.height if video_codec is not None else None,
            fps=_fps(video),
            audio_codec=audio_codec.name if audio_codec is not None else None,
            sample_rate=audio_codec.sample_rate if audio_codec is not None else None,
            channels=audio_codec.channels if audio_codec is not None else None,
        )


def _duration_us(
    container: av.container.InputContainer, video: av.VideoStream | None
) -> int | None:
    """Length in microseconds from the container header, falling back to the video stream."""
    if container.duration:
        return container.duration
    if video is not None and video.duration and video.time_base is not None:
        return int(video.duration * video.time_base * 1_000_000)
    return None


def _fps(video: av.VideoStream | None) -> float | None:
    """Average frame rate; variable-rate files report the declared average."""
    if video is None or video.average_rate is None:
        return None
    rate = float(video.average_rate)
    return rate if rate > 0 else None
