"""Command metadata and input parsing for the interactive terminal prompt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Describe one prompt command without coupling it to its handler."""

    name: str  # Canonical command name without the leading slash.
    aliases: tuple[str, ...]  # Alternate names accepted by the prompt.
    description: str  # Short text shown in command suggestions and help.
    usage: str  # Optional argument form shown after the command name.


COMMANDS = (
    CommandSpec("help", (), "Show available prompt commands", ""),
    CommandSpec("exit", ("quit",), "Exit the workbench", ""),
    CommandSpec("model", ("models",), "Show configured model connections", ""),
    CommandSpec("jobs", (), "Refresh and list analysis jobs", ""),
    CommandSpec("uploads", (), "List resumable uploads", ""),
    CommandSpec("upload", (), "Upload a local MP4", "<@video.mp4>"),
    CommandSpec("analyze", (), "Start analysis for a video", "<objective>"),
    CommandSpec("ask", (), "Ask about the selected job", "<question>"),
    CommandSpec("resume", (), "Resume a saved analysis job", "<job-id>"),
    CommandSpec("show", (), "Show evidence for a job", "[job-id]"),
    CommandSpec("locate", (), "Show evidence in a source-time range", "<start-ms> <end-ms>"),
    CommandSpec("cancel", (), "Mark a job as cancelled", "[job-id]"),
    CommandSpec("delete", (), "Delete generated job data", "[job-id]"),
    CommandSpec("config", (), "Show non-secret runtime settings", ""),
    CommandSpec("reference", (), "Show provider compatibility notes", ""),
)

_COMMANDS_BY_NAME = {
    name.casefold(): command
    for command in COMMANDS
    for name in (command.name, *command.aliases)
}


def parse_command(value: str) -> tuple[str, str] | None:
    """Split one leading slash command from its argument text.

    value: Submitted input; it is not mutated.
    Returns a case-folded command name and trimmed arguments, or None for natural input.
    """

    value = value.strip()
    if not value.startswith("/"):
        return None
    command, separator, arguments = value[1:].partition(" ")
    if not separator:
        command, separator, arguments = command.partition("\t")
    return command.casefold(), arguments.strip()


def command_spec(name: str) -> CommandSpec | None:
    """Resolve a command name or alias without regard to letter case."""

    return _COMMANDS_BY_NAME.get(name.casefold())


def command_label(command: CommandSpec) -> str:
    """Render the usage and aliases shown for one command.

    command: Registered command metadata; it is not mutated.
    Returns a slash-prefixed label with optional usage and aliases.
    """

    label = f"/{command.name}"
    if command.usage:
        label += f" {command.usage}"
    if command.aliases:
        label += " (aliases: " + ", ".join(f"/{alias}" for alias in command.aliases) + ")"
    return label


def command_suggestions(value: str) -> list[CommandSpec]:
    """Return commands matching an incomplete slash token.

    value: Current input; only a leading slash token without whitespace is searched.
    Returns matching commands in registration order; an empty list means no match.
    """

    if not value.startswith("/") or any(char.isspace() for char in value):
        return []
    prefix = value[1:].casefold()
    return [
        command
        for command in COMMANDS
        if command.name.casefold().startswith(prefix)
        or any(alias.casefold().startswith(prefix) for alias in command.aliases)
    ]
