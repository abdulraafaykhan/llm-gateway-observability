"""Data shape for one simulated model's behaviour in fake mode."""

from pydantic import BaseModel, Field


class FakeProfile(BaseModel):
    """Made-up timing, size, and failure settings for one model in fake mode."""

    base_latency_ms: float
    ms_per_output_token: float
    jitter_percent: float
    slow_outlier_probability: float
    slow_outlier_multiplier: float
    error_probability: float
    truncate_probability: float
    min_output_tokens: int
    max_output_tokens: int
    max_concurrent_requests: int = Field(ge=1)
