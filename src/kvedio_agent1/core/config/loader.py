"""Discovery, creation and validation of the user-level configuration.

Locates the configuration directory, creates the first configuration from the
packaged template, turns parse and validation failures into messages that carry
field paths but never field values, resolves ``env:`` key references, and reads
or writes the program's own selections next to the user's file.

Does not cache the parsed configuration and does not write the user's file: it
belongs to the user, and rewriting it destroys comments and layout.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import AppConfig, ModelBinding, Role

HOME_ENV = "KVEDIO_AGENT_HOME"
CONFIG_FILENAME = "config.yml"
STATE_FILENAME = "state.yml"
DATABASE_FILENAME = "kvedio.db"
TEMPLATE = "default/config.example.yml"


class ConfigError(Exception):
    """The configuration cannot be used: absent, unparsable, or carrying an illegal field.

    The message names the file and the offending field path. It never repeats a
    field's value, because one of those values may be a key.
    """


@dataclass(frozen=True)
class Loaded:
    """A validated configuration together with where it came from."""

    config: AppConfig
    path: Path
    created: bool


@dataclass(frozen=True)
class Selection:
    """The binding a role runs on, and the request that could not be honoured.

    ``requested`` is the binding name asked for by the state file or the config
    default when no such binding exists; ``None`` means no request was overridden.
    """

    role: Role
    binding: ModelBinding | None
    requested: str | None

    @property
    def fell_back(self) -> bool:
        return self.requested is not None


def home_dir() -> Path:
    """Directory holding the configuration, the state file and the database.

    ``KVEDIO_AGENT_HOME`` overrides it so a test or a second instance can run
    without touching the user's own directory.
    """
    override = os.environ.get(HOME_ENV)
    return Path(override).expanduser() if override else Path.home() / ".kvedio_agent"


def config_path() -> Path:
    return home_dir() / CONFIG_FILENAME


def state_path() -> Path:
    return home_dir() / STATE_FILENAME


def database_path() -> Path:
    return home_dir() / DATABASE_FILENAME


def template_text() -> str:
    """The packaged configuration template, shipped inside the package rather than the tree."""
    return resources.files("kvedio_agent1").joinpath(TEMPLATE).read_text(encoding="utf-8")


def ensure_config(path: Path | None = None) -> tuple[Path, bool]:
    """Create the configuration from the template when it is absent.

    Returns the path and whether this call created the file. Permissions are
    narrowed on POSIX because the template invites users to store a literal key.
    """
    target = path or config_path()
    if target.exists():
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template_text(), encoding="utf-8")
    if os.name == "posix":
        target.chmod(0o600)
    return target, True


def load(path: Path | None = None) -> AppConfig:
    """Parse and validate the configuration.

    Raises ``ConfigError`` naming the file and the failing field paths when the
    file is absent, is not a mapping, or carries an unknown or illegal field.
    """
    target = path or config_path()
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"配置不存在：{target}") from None
    except OSError as exc:
        raise ConfigError(f"配置不可读：{target}（{exc.strerror}）") from None
    except yaml.YAMLError as exc:
        reason = getattr(exc, "problem", None) or "无法解析"
        raise ConfigError(f"配置不是合法 YAML：{target}（{reason}）") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置顶层必须是映射：{target}")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"配置字段非法：{target}（{_describe(exc)}）") from None


def load_or_create(path: Path | None = None) -> Loaded:
    """Load the configuration, creating it from the template first when it is missing."""
    target, created = ensure_config(path)
    return Loaded(config=load(target), path=target, created=created)


def resolved_api_key(binding: ModelBinding) -> str | None:
    """The key to present upstream: the environment value for a reference, the literal otherwise."""
    env_var = binding.env_var
    if env_var is not None:
        return os.environ.get(env_var) or None
    return binding.api_key or None


def select(config: AppConfig, role: Role, state: Mapping[Role, str]) -> Selection:
    """Pick the binding a role runs on.

    Precedence is the program state file, then the file's own ``active`` entry,
    then the first declared binding. A requested name that is not declared falls
    back to the first binding and reports the name in ``Selection.requested``.
    """
    bindings = config.roles.for_role(role)
    requested = state.get(role) or config.active.get(role)
    if requested is not None:
        for binding in bindings:
            if binding.name == requested:
                return Selection(role=role, binding=binding, requested=None)
        return Selection(role=role, binding=bindings[0] if bindings else None, requested=requested)
    return Selection(role=role, binding=bindings[0] if bindings else None, requested=None)


def read_active(path: Path | None = None) -> dict[Role, str]:
    """Selections the program persisted; a missing or damaged file yields no selections."""
    target = path or state_path()
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    active = raw.get("active")
    if not isinstance(active, dict):
        return {}
    selections: dict[Role, str] = {}
    for name, value in active.items():
        if not isinstance(value, str) or not value:
            continue
        try:
            selections[Role(name)] = value
        except ValueError:
            continue
    return selections


def write_active(role: Role, name: str, path: Path | None = None) -> None:
    """Persist one role's selection, leaving every other role's entry untouched."""
    target = path or state_path()
    selections = read_active(target)
    selections[role] = name
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = {member.value: selections[member] for member in Role if member in selections}
    target.write_text(
        yaml.safe_dump({"active": ordered}, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )


def _describe(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
