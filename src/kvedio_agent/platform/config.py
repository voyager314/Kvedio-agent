"""Non-secret runtime parameters for the video analysis application."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kvedio_agent.platform.settings import Settings


class Config(BaseModel):
    """Validated runtime behavior independent of provider connections."""

    model_config = ConfigDict(extra="forbid")

    max_video_duration_seconds: int = Field(default=7200, gt=0)  # Duration ceiling in seconds.
    retention_days: int = Field(default=30, gt=0)  # Positive artifact retention period in days.
    max_segment_duration_seconds: int = Field(default=120, gt=0)  # Window length in seconds.
    segment_overlap_seconds: int = Field(default=2, ge=0)  # Overlap in seconds; zero disables it.
    max_media_workers: int = Field(default=2, gt=0)  # Positive preprocessing worker limit.
    max_asr_workers: int = Field(default=4, gt=0)  # Positive ASR stage worker limit.
    max_video_workers: int = Field(default=2, gt=0)  # Positive video stage worker limit.
    max_global_workers: int = Field(default=6, gt=0)  # Positive cap across active work stages.
    max_job_cost_usd: float | None = Field(default=None, ge=0)  # USD ceiling; None means unlimited.

    @classmethod
    def from_settings(cls, settings: Settings) -> Config:
        """Build runtime parameters from the optional settings runtime object."""

        try:
            return cls.model_validate(settings.runtime)
        except ValidationError:
            raise ValueError("invalid runtime configuration") from None

