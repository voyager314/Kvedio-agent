"""Model access: one client per capability, each addressed by a configured URL."""

from .chat import Chat, Message, Reply
from .errors import ModelError
from .transcription import Transcriber

__all__ = ["Chat", "Message", "ModelError", "Reply", "Transcriber"]
