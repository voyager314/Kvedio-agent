"""Shape of the user configuration file.

Fixes the vocabulary of ``config.yml`` as types and rejects unknown fields or
out-of-range values at load time.

Does not touch the filesystem or resolve ``env:`` references; those depend on the
machine and belong to the loader.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Role(StrEnum):
    """A capability slot that binds to one model."""

    SUPERVISOR = "supervisor"
    ASR = "asr"
    VIDEO = "video"
    EMBEDDING = "embedding"


class ModelBinding(BaseModel):
    """The one model behind one role.

    ``url`` is the complete request address and is used exactly as written. Sending
    paths differ between providers, so nothing is appended to it and no trailing
    slash is removed: a half address succeeds only up to the first call.

    ``api_key`` never leaves the object: it is excluded from serialization and
    from the repr, so a bound key cannot reach a log line, an error message or a
    rendered panel. Reporting uses ``env_var`` and ``has_key`` instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    url: str = Field(min_length=1)
    api_key: str = Field(default="", exclude=True, repr=False)

    @field_validator("url")
    @classmethod
    def _complete_address(cls, value: str) -> str:
        """Refuse a bare host, which would read as configured until the first request fails."""
        if not value.startswith(("http://", "https://")):
            raise ValueError("必须是以 http:// 或 https:// 开头的完整 URL")
        return value

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


class RuntimeLimits(BaseModel):
    """Resource ceilings the pipeline honours, independent of any model choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_video_duration_seconds: int = Field(default=7200, gt=0)
    retention_days: int = Field(default=30, ge=0)
    max_global_workers: int = Field(default=6, ge=1, le=64)
    request_timeout_seconds: int = Field(default=300, gt=0)
    stage_budget_floor_seconds: int = Field(default=300, ge=30, le=3600)
    stage_budget_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    stage_budget_ceiling_seconds: int = Field(default=0, ge=0, le=86400)
    asr_chunk_seconds: int = Field(default=300, ge=10, le=1800)
    asr_concurrency: int = Field(default=2, ge=1, le=16)
    coverage_block_seconds: int = Field(default=180, ge=30, le=900)
    scene_max_seconds: int = Field(default=90, ge=10, le=600)
    workspace_dir: Path | None = None


class AppConfig(BaseModel):
    """The whole file: one model per role, plus the resource limits.

    A role left out is unconfigured rather than invalid: a freshly created
    template legitimately carries a slot whose key is not filled in yet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    supervisor: ModelBinding | None = None
    asr: ModelBinding | None = None
    video: ModelBinding | None = None
    embedding: ModelBinding | None = None
    runtime: RuntimeLimits = RuntimeLimits()

    def for_role(self, role: Role) -> ModelBinding | None:
        """The model configured for ``role``; ``None`` when the role was left out."""
        return getattr(self, role.value)
