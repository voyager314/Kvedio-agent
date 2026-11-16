"""Identity of a file on disk, cheap enough to compute on every reference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..domain import FileIdentity

SAMPLE_BYTES = 4 * 1024 * 1024


def identify(path: Path) -> FileIdentity:
    """Fingerprint a file from its size, its modification time and two samples.

    Reads at most 8 MiB whatever the file's length, so a 20 GB recording is
    identified in milliseconds. The modification time participates in the hash: a
    rewrite that keeps the byte count still changes identity.
    """
    stat = path.stat()
    with path.open("rb") as handle:
        head = handle.read(SAMPLE_BYTES)
        tail = b""
        if stat.st_size > SAMPLE_BYTES:
            handle.seek(max(SAMPLE_BYTES, stat.st_size - SAMPLE_BYTES))
            tail = handle.read(SAMPLE_BYTES)
    digest = hashlib.sha256()
    digest.update(stat.st_size.to_bytes(8, "big", signed=True))
    digest.update(stat.st_mtime_ns.to_bytes(8, "big", signed=True))
    digest.update(head)
    digest.update(tail)
    return FileIdentity(
        path=path,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        fingerprint=digest.hexdigest(),
    )
