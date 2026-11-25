"""Request and response shapes for the chat endpoint."""

from pydantic import BaseModel, Field

from app.models.completion import CompletionResult, Message


class ChatRequest(BaseModel):
    """Body of POST /v1/chat."""

    messages: list[Message] = Field(min_length=1)
    model: str | None = None


class ChatResponse(BaseModel):
    """Body returned by POST /v1/chat."""

    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    truncated: bool

    @classmethod
    def from_completion(
        cls, result: CompletionResult, model_name: str, latency_ms: float
    ) -> "ChatResponse":
        """Build the response from a provider result, the model name, and the latency."""
        return cls(
            answer=result.text,
            model=model_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=round(latency_ms, 1),
            truncated=result.truncated,
        )
