"""JSON shapes for the Gateway page: non-secret settings, live model health, route preview."""

from pydantic import BaseModel, Field

from app.models.routing import RoutingConfig, RuleHit


class ModelConfigView(BaseModel):
    """One configured model and its prices (per million tokens)."""

    name: str
    provider: str
    model_id: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float


class GatewayConfigResponse(BaseModel):
    """Body of GET /api/gateway/config: only non-secret settings, no keys and no paths."""

    mode: str
    default_model_name: str
    max_output_tokens: int
    temperature: float
    request_timeout_seconds: float
    database_file: str
    models: list[ModelConfigView]
    routing: RoutingConfig


class ModelHealthView(BaseModel):
    """Live numbers for one model, from the attempts of requests in the window."""

    name: str
    attempts: int
    answered: int
    truncated: int
    errors: int
    rate_limit_errors: int
    error_rate_percent: float
    p50_ms: float | None
    p95_ms: float | None
    avg_output_tokens: float | None
    cost_usd: float
    health: str


class ModelsHealthResponse(BaseModel):
    """Body of GET /api/gateway/models."""

    minutes: int
    models: list[ModelHealthView]


class RoutePreviewRequest(BaseModel):
    """Body of POST /api/gateway/route-preview."""

    question: str = Field(min_length=1, max_length=4000)


class RoutePreviewResponse(BaseModel):
    """What the router would choose for a question; no model is called."""

    model: str
    score: int
    threshold: int
    hits: list[RuleHit]
    reason: str
    not_fired: list[str]
