"""Data shape for one configured model entry."""

from pydantic import BaseModel


class ModelSpec(BaseModel):
    """One model entry from config/models.toml."""

    name: str
    provider: str
    model_id: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float

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
