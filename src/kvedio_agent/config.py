"""Non-secret runtime parameters for the video analysis application."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kvedio_agent.settings import Settings


class Config(BaseModel):
    """Validated runtime behavior independent of provider connections."""

    model_config = ConfigDict(extra="forbid")

    max_video_duration_seconds: int = Field(default=7200, gt=0)
    retention_days: int = Field(default=30, gt=0)
    max_segment_duration_seconds: int = Field(default=120, gt=0)
    segment_overlap_seconds: int = Field(default=2, ge=0)
    max_media_workers: int = Field(default=2, gt=0)
    max_asr_workers: int = Field(default=4, gt=0)
    max_vision_workers: int = Field(default=2, gt=0)

    @classmethod
    def from_settings(cls, settings: Settings) -> Config:
        """Build runtime parameters from the optional settings runtime object."""

        try:
            return cls.model_validate(settings.runtime)
        except ValidationError:
            raise ValueError("invalid runtime configuration") from None

