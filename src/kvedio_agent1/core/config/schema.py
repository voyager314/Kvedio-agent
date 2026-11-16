"""Shape of the user configuration file.

Fixes the vocabulary of ``config.yml`` as types and rejects unknown fields or
out-of-range values at load time.

Does not touch the filesystem, resolve ``env:`` references, or persist program
state; those depend on the machine and belong to the loader.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    """A capability slot that binds to one model."""

    SUPERVISOR = "supervisor"
    ASR = "asr"
    VIDEO = "video"


class ModelBinding(BaseModel):
    """One selectable model for one role.

    ``name`` is the handle a user switches between, unique within its role.
    ``api_key`` never leaves the object: it is excluded from serialization and
    from the repr, so a bound key cannot reach a log line, an error message or a
    rendered panel. Reporting uses ``env_var`` and ``has_key`` instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)
    api_key: str = Field(default="", exclude=True, repr=False)

    @property
    def env_var(self) -> str | None:
        """Environment variable the key is read from; ``None`` when the key is stored literally."""
        if not self.api_key.startswith("env:"):
            return None
        name = self.api_key.removeprefix("env:").strip()
        return name or None

    @property
    def has_key(self) -> bool:
        """Whether a key is declared.

        An ``env:`` reference counts as declared even when the variable is unset.
        """
        return bool(self.api_key)


class RoleBindings(BaseModel):
    """Every binding declared per role.

    A role with no binding is unconfigured rather than invalid: a freshly created
    template legitimately leaves slots empty until the user fills them in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    supervisor: tuple[ModelBinding, ...] = ()
    asr: tuple[ModelBinding, ...] = ()
    video: tuple[ModelBinding, ...] = ()

    def for_role(self, role: Role) -> tuple[ModelBinding, ...]:
        return getattr(self, role.value)

    @model_validator(mode="after")
    def _names_unique(self) -> RoleBindings:
        for role in Role:
            names = [binding.name for binding in self.for_role(role)]
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"role {role.value} declares the binding name twice: {', '.join(repeated)}"
                )
        return self


class RuntimeLimits(BaseModel):
    """Resource ceilings the pipeline honours, independent of any model choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_video_duration_seconds: int = Field(default=7200, gt=0)
    retention_days: int = Field(default=30, ge=0)
    max_global_workers: int = Field(default=6, ge=1, le=64)
    request_timeout_seconds: int = Field(default=120, gt=0)
    workspace_dir: Path | None = None


class AppConfig(BaseModel):
    """The whole file: declared bindings, a default selection, and resource limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: RoleBindings = RoleBindings()
    active: dict[Role, str] = {}
    runtime: RuntimeLimits = RuntimeLimits()
