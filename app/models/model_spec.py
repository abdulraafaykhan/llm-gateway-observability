"""Data shape for one configured model entry."""

from typing import Literal, Self

from pydantic import BaseModel, model_validator


class ModelSpec(BaseModel):
    """One model entry from config/models.toml."""

    name: str
    provider: str
    model_id: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _reasoning_effort_only_for_groq(self) -> Self:
        """Reject reasoning_effort on gemini models; Gemini requests send no thinking setting."""
        if self.provider == "gemini" and self.reasoning_effort is not None:
            raise ValueError("reasoning_effort is not supported for provider 'gemini'")
        return self

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Return the cost in US dollars for the given token counts, rounded to 10 decimals."""
        return round(
            (
                input_tokens * self.input_price_per_million_tokens
                + output_tokens * self.output_price_per_million_tokens
            )
            / 1_000_000,
            10,
        )
