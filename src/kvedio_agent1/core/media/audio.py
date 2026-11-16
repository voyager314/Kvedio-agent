"""Decoding the audio track into the shape a transcriber accepts.

PyAV carries its own FFmpeg build, so extraction works on a machine with no
system ffmpeg. Output is 16 kHz mono signed 16-bit PCM in a WAV container: the
lowest rate that still carries speech, and the only shape the slicer has to know.
"""

from __future__ import annotations

import os
import wave
from collections.abc import Callable
from pathlib import Path

import av

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes per sample, signed 16-bit
CHANNELS = 1


class NoAudioTrack(Exception):
    """The file carries no audio stream, so there is nothing to transcribe."""


def extract_audio(
    source: Path,
    target: Path,
    *,
    expected_ms: int,
    progress: Callable[[float], None] | None = None,
) -> int:
    """Decode the first audio stream and write it as a mono 16 kHz WAV.

    ``expected_ms`` is the container's declared length and is used only to turn
    decoded samples into a fraction. The return value is the length of the audio
    actually written, in milliseconds, measured from the samples: a caller that
    compares it with the declaration learns whether the decode was whole.
    ``progress`` receives at most a hundred fractions. The file appears at
    ``target`` only when complete: an interrupted write stays at a temporary name.
    """
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with av.open(str(source)) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise NoAudioTrack(f"没有音轨：{source.name}")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            expected_samples = max(1, expected_ms * SAMPLE_RATE // 1000)
            reported = -1
            with wave.open(str(partial), "wb") as out:
                out.setnchannels(CHANNELS)
                out.setsampwidth(SAMPLE_WIDTH)
                out.setframerate(SAMPLE_RATE)
                for frame in container.decode(stream):
                    for resampled in resampler.resample(frame):
                        out.writeframes(resampled.to_ndarray().tobytes())
                        total += resampled.samples
                    if progress is not None:
                        percent = min(99, total * 100 // expected_samples)
                        if percent > reported:
                            reported = percent
                            progress(percent / 100)
                for resampled in resampler.resample(None):
                    out.writeframes(resampled.to_ndarray().tobytes())
                    total += resampled.samples
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, target)
    return round(total * 1000 / SAMPLE_RATE)
