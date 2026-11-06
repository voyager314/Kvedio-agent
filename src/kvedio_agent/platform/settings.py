"""User-owned model settings and first-run initialization."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console
from rich.panel import Panel

Role = Literal["chat", "asr", "vision"]
ROLES: tuple[Role, ...] = ("chat", "asr", "vision")

SETTINGS_FILENAME = "settings.json"
STATE_FILENAME = "state.json"
TEMPLATE_RESOURCE = "settings.example.json"
REFERENCE_RESOURCE = "reference.md"


def config_dir() -> Path:
    """Return the fixed user configuration directory."""

    return Path.home() / ".kvedio_ai"


def settings_path() -> Path:
    """Return the user settings path."""

    return config_dir() / SETTINGS_FILENAME


def state_path() -> Path:
    """Return the persisted model binding state path."""

    return config_dir() / STATE_FILENAME


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
    """A provider connection, credential, and the roles it is allowed to serve."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)  # Provider model identifier sent on each call.
    api_base: str | None = None  # Provider endpoint URL, or None for SDK defaults.
    api_key: str | None = None  # Provider credential stored in the user-owned settings file.
    roles: list[Role] = Field(min_length=1)  # Allowed roles: chat, asr, and vision.
    note: str | None = None  # User-owned non-secret label, or None when absent.
    timeout_seconds: float = Field(default=120, gt=0)  # Positive synchronous timeout, seconds.
    max_attempts: int = Field(default=3, gt=0)  # Positive maximum attempts for one call.
    # Requests per minute; None is unlimited.
    requests_per_minute: int | None = Field(default=None, gt=0)
    # USD per million input tokens; None is unknown.
    input_cost_per_million: float | None = Field(default=None, ge=0)
    # USD per million output tokens; None is unknown.
    output_cost_per_million: float | None = Field(default=None, ge=0)
    # USD per audio minute; None is unknown.
    audio_cost_per_minute: float | None = Field(default=None, ge=0)

    def resolve_key(self) -> str | None:
        """Return the provider credential configured for this model."""

        return self.api_key


class RuntimeState(BaseModel):
    """Persisted role-to-model selections made from the terminal."""

    model_config = ConfigDict(extra="forbid")

    active: dict[str, str] = Field(default_factory=dict)  # Role names to configured model keys.


class Settings(BaseModel):
    """Validated contents of the user-owned settings file."""

    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelEntry] = Field(default_factory=dict)  # User-named provider connections.
    active: dict[str, str] = Field(default_factory=dict)  # Role names to configured model keys.
    runtime: dict[str, Any] = Field(default_factory=dict)  # Runtime parameters validated by Config.

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

    def candidates(self, role: Role) -> list[tuple[str, ModelEntry]]:
        """Return models explicitly permitted to serve a role."""

        return [(name, entry) for name, entry in self.models.items() if role in entry.roles]

    def resolve(self, role: Role, name: str | None = None) -> tuple[str, ModelEntry]:
        """Resolve an explicit or persisted role binding without choosing a default."""

        selected = name or self.active.get(role)
        if selected is None:
            raise LookupError(f"no model is bound to the {role} role")
        entry = self.models.get(selected)
        if entry is None:
            raise LookupError(f"configured model {selected!r} does not exist")
        if role not in entry.roles:
            raise LookupError(f"configured model {selected!r} does not support the {role} role")
        return selected, entry

    def with_state(self, state: RuntimeState) -> Settings:
        """Return settings with persisted bindings overlaid on file bindings."""

        return self.model_copy(update={"active": {**self.active, **state.active}})


def load_state() -> RuntimeState:
    """Load persisted bindings; missing or invalid state behaves as empty state."""

    path = state_path()
    if not path.exists():
        return RuntimeState()
    try:
        return RuntimeState.model_validate(
            json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
        )
    except (OSError, json.JSONDecodeError, ValidationError):
        return RuntimeState()


def save_state(state: RuntimeState) -> None:
    """Atomically persist role bindings without writing credentials."""

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = state_path().with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(state_path())


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
            "then run again. Optional limits, prices, "
            "model tuning, role bindings, and runtime parameters are included.",
            title="First run",
            border_style="green",
        )
    )
    return None

