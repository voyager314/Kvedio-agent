"""User-owned model settings and first-run initialization."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console
from rich.panel import Panel

SETTINGS_FILENAME = "settings.json"
TEMPLATE_RESOURCE = "settings.example.json"
REFERENCE_RESOURCE = "reference.md"


def config_dir() -> Path:
    """Return the fixed user configuration directory."""

    return Path.home() / ".kvedio_ai"


def settings_path() -> Path:
    """Return the user settings path."""

    return config_dir() / SETTINGS_FILENAME


def _strip_jsonc(text: str) -> str:
    """Remove comments outside JSON strings without changing string values."""

    result: list[str] = []
    index = 0
    in_string = False
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if char == "\\" and index + 1 < len(text):
                result.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text):
            if text[index + 1] == "/":
                index += 2
                while index < len(text) and text[index] != "\n":
                    index += 1
                continue
            if text[index + 1] == "*":
                index += 2
                while index + 1 < len(text) and text[index : index + 2] != "*/":
                    index += 1
                index = min(index + 2, len(text))
                continue
        result.append(char)
        index += 1
    return "".join(result)


class ModelEntry(BaseModel):
    """One provider connection and the limits governing every call made over it."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)  # Provider model identifier sent on each call.
    base_url: str | None = None  # Provider endpoint URL, or None for the SDK default.
    api_key: str | None = None  # Provider credential held in the user-owned settings file.
    timeout_seconds: float = Field(default=120, gt=0)  # Positive per-call timeout, seconds.
    max_attempts: int = Field(default=3, gt=0)  # Positive maximum attempts for one call.
    # Requests per minute allowed against this connection; None is unlimited.
    requests_per_minute: int | None = Field(default=None, gt=0)
    # USD per million input tokens; None is unknown.
    input_cost_per_million: float | None = Field(default=None, ge=0)
    # USD per million output tokens; None is unknown.
    output_cost_per_million: float | None = Field(default=None, ge=0)
    # USD per audio minute; None is unknown.
    audio_cost_per_minute: float | None = Field(default=None, ge=0)

    def resolve_key(self) -> str | None:
        """Return the provider credential configured for this connection."""

        return self.api_key


class Settings(BaseModel):
    """Validated contents of the user-owned settings file.

    Each capability names its own connection, so a run reads a model from the
    capability it needs rather than from a separate table of bindings. One provider
    may back both capabilities by repeating its details under each.
    """

    model_config = ConfigDict(extra="forbid")

    # Connection that reads the recording and answers questions about it; required.
    video: ModelEntry
    # Connection that transcribes audio; None leaves speech untranscribed.
    asr: ModelEntry | None = None
    # Runtime parameters validated by Config; unknown keys are rejected there.
    runtime: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Load settings without exposing invalid values in raised errors."""

        target = path or settings_path()
        try:
            data = json.loads(_strip_jsonc(target.read_text(encoding="utf-8")))
            if not isinstance(data, dict):
                raise ValueError("the top-level value must be an object")
            return cls.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            raise ValueError(f"invalid settings file: {target}") from None


def render_template() -> str:
    """Return the provider-free settings template shipped with the package."""

    return resources.files("kvedio_agent.defaults").joinpath(TEMPLATE_RESOURCE).read_text("utf-8")


def render_reference() -> str:
    """Return the packaged provider reference document."""

    return resources.files("kvedio_agent.defaults").joinpath(REFERENCE_RESOURCE).read_text("utf-8")


def ensure_settings(console: Console) -> Settings | None:
    """Create the full first-run configuration template and return no settings."""

    path = settings_path()
    if path.exists():
        return Settings.load(path)
    config_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(render_template(), encoding="utf-8")
    console.print(
        Panel(
            f"Created complete settings template at\n[dim]{path}[/dim]\n\n"
            "Replace every REPLACE_WITH_* value with your provider details and API keys, "
            "then run again. Optional limits, prices, and runtime parameters are included.",
            title="First run",
            border_style="green",
        )
    )
    return None
