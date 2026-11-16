"""Where the files derived from one video live.

Artifacts are addressed by video id rather than by source path: a renamed source
still finds its work, and a rewritten source is a new record with its own
directory, so nothing from the previous content is read back by mistake.

Does not decide what an artifact contains or when it is stale; each stage owns
that judgement and the key it stores under.
"""

from __future__ import annotations

from pathlib import Path


class Workspace:
    """The directory tree under which one run's derived files are kept."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def video_dir(self, video_id: int) -> Path:
        """The directory owned by one video, created on first use."""
        path = self.root / f"video_{video_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def audio_path(self, video_id: int) -> Path:
        """The prepared 16 kHz mono WAV for one video; its existence implies a complete write."""
        return self.video_dir(video_id) / "audio.wav"

    def scratch_dir(self, video_id: int) -> Path:
        """Space for files whose lifetime is one chunk; stages delete what they create."""
        path = self.video_dir(video_id) / "tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path
