"""Reading one submitted line into what the workbench should do with it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

USAGE = "用法：@<路径> <问题>；/help 查看全部命令"


@dataclass(frozen=True)
class Reference:
    """A file to analyse, with the user's own wording of what to find out.

    ``path`` is absolute, resolved against the working directory at submission
    time; ``question`` is stored verbatim and never rewritten.
    """

    path: Path
    question: str


@dataclass(frozen=True)
class Command:
    """A slash command: its name without the slash, and the arguments that followed."""

    name: str
    arguments: tuple[str, ...]


Submission = Reference | Command


def parse_submission(text: str, base: Path) -> Submission | None:
    """Interpret one submitted line.

    A line starting with ``/`` is a command. A line starting with ``@`` names a
    file, optionally in double quotes, followed by the question; a path without a
    question is refused rather than guessed at. Anything else returns ``None`` so
    the caller can show the usage instead of inventing an intention.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("/"):
        parts = stripped.split()
        return Command(name=parts[0][1:].lower(), arguments=tuple(parts[1:]))
    if not stripped.startswith("@"):
        return None
    rest = stripped[1:]
    if not rest.strip():
        return None
    if rest.startswith('"'):
        closing = rest.find('"', 1)
        if closing == -1:
            return None
        raw_path = rest[1:closing]
        question = rest[closing + 1 :].strip()
    else:
        parts = rest.split(maxsplit=1)
        raw_path = parts[0]
        question = parts[1].strip() if len(parts) > 1 else ""
    if not raw_path or not question:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base / path
    return Reference(path=path, question=question)
