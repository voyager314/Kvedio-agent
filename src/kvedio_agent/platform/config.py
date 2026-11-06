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
    # Frames per second kept when a window is re-encoded as a clip; what a video model
    # is billed for is frames, so this sets the price of one perception call.
    clip_frames_per_second: float = Field(default=1.0, gt=0)
    clip_max_width: int = Field(default=640, gt=0)  # Clip width ceiling in pixels; height follows.
    max_clip_events: int = Field(default=12, gt=0)  # Observations kept from one clip, at most.
    max_media_workers: int = Field(default=2, gt=0)  # Positive preprocessing worker limit.
    max_asr_workers: int = Field(default=4, gt=0)  # Positive ASR stage worker limit.
    max_video_workers: int = Field(default=2, gt=0)  # Positive video stage worker limit.
    max_global_workers: int = Field(default=6, gt=0)  # Positive cap across active work stages.
    max_job_cost_usd: float | None = Field(default=None, ge=0)  # USD ceiling; None means unlimited.

    @classmethod
    def from_settings(cls, settings: Settings) -> Config:
        """Build runtime parameters from settings, sizing the clip to the video connection.

        The window one perception call reads follows the rate the configured model
        bills media at, so the same settings read ten minutes per call on a cheap
        model and two on an expensive one. Deriving it here rather than in the stage
        keeps the window that produced a manifest identical to the window a later
        run expects, which is what makes preprocessing resumable.
        """

        try:
            config = cls.model_validate(settings.runtime)
        except ValidationError:
            raise ValueError("invalid runtime configuration") from None
        rate = settings.video.video_tokens_per_second
        budget = settings.video.video_budget_tokens
        if rate is None or budget is None:
            return config
        affordable = max(1, int(budget / rate))
        window = min(config.max_segment_duration_seconds, affordable)
        return config.model_copy(update={"max_segment_duration_seconds": window})

