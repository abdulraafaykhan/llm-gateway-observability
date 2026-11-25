"""Data shape for one configured model entry."""

from pydantic import BaseModel


class ModelSpec(BaseModel):
    """One model entry from config/models.toml."""

    name: str
    provider: str
    model_id: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float
