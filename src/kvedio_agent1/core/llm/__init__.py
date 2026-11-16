"""Model access: one client per capability, each addressed by a configured URL."""

from .transcription import Transcriber, TranscriptionError

__all__ = ["Transcriber", "TranscriptionError"]
