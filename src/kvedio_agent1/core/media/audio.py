"""Decoding the audio track into the shape a transcriber accepts.

PyAV carries its own FFmpeg build, so extraction works on a machine with no
system ffmpeg. Output is 16 kHz mono signed 16-bit PCM in a WAV container: the
lowest rate that still carries speech, and the only shape the slicer has to know.

A track recorded far below full scale is lifted toward it. A recogniser fed a
signal tens of decibels down follows the room instead of the words and repeats
one sentence for minutes, which costs the whole transcript; the gain is decided
from the samples, so a correctly recorded track is left untouched.
"""

from __future__ import annotations

import os
import wave
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes per sample, signed 16-bit
CHANNELS = 1
TARGET_PEAK = 29_000  # about -1 dBFS, the loudest sample that still leaves the encoder room
QUIET_PEAK = 3_200  # about -20 dBFS: below this the recording is unusably quiet
MAX_GAIN = 32.0  # +30 dB, past which a near-silent track only becomes amplified noise
BLOCK_FRAMES = 1 << 20
AUDIO_PROFILE = f"s16-mono-{SAMPLE_RATE}-peak{TARGET_PEAK}"


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
    decoded samples into a fraction. A track whose peak sits below ``QUIET_PEAK``
    is amplified toward ``TARGET_PEAK``, capped at ``MAX_GAIN``; a track already
    at a usable level is written unchanged. The return value is the length of the
    audio actually written, in milliseconds, measured from the samples: a caller
    that compares it with the declaration learns whether the decode was whole.
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
        _raise_level(partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, target)
    return round(total * 1000 / SAMPLE_RATE)


def needs_level_fix(path: Path) -> bool:
    """Whether an extracted file still sits below the level a recogniser can follow.

    A caller that finds a prepared file from an earlier run asks this before
    reusing it: a quiet file predates the current extraction and has to be
    remade, while a usable one costs nothing to keep.
    """
    peak = _peak(path)
    return 0 < peak < QUIET_PEAK


def _raise_level(path: Path) -> None:
    """Amplify a too-quiet WAV in place; leave a usable one alone.

    The peak is measured over the file and the whole file is rewritten only when
    that peak is below ``QUIET_PEAK``. Gain is capped at ``MAX_GAIN`` and the
    signal is never attenuated, so a recording that is already loud keeps its
    samples byte for byte.
    """
    peak = _peak(path)
    if peak <= 0 or peak >= QUIET_PEAK:
        return
    gain = min(MAX_GAIN, TARGET_PEAK / peak)
    boosted = path.with_name(path.name + ".level")
    with wave.open(str(path), "rb") as source, wave.open(str(boosted), "wb") as out:
        out.setparams(source.getparams())
        while True:
            block = source.readframes(BLOCK_FRAMES)
            if not block:
                break
            samples = np.frombuffer(block, dtype=np.int16).astype(np.float32) * gain
            out.writeframes(np.clip(samples, -32768, 32767).astype(np.int16).tobytes())
    os.replace(boosted, path)


def _peak(path: Path) -> int:
    """The largest absolute sample in the file, as a positive integer."""
    peak = 0
    with wave.open(str(path), "rb") as handle:
        while True:
            block = handle.readframes(BLOCK_FRAMES)
            if not block:
                return peak
            if len(block) % SAMPLE_WIDTH:
                block = block[: len(block) - len(block) % SAMPLE_WIDTH]
            peak = max(peak, int(np.abs(np.frombuffer(block, dtype=np.int16)).max(initial=0)))
