"""Decode-level media work: identity of a file, stream layout of its container."""

from .fingerprint import identify
from .probe import ProbeError, probe

__all__ = ["ProbeError", "identify", "probe"]
