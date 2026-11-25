"""Data shapes for chat messages and provider results."""

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """One chat message sent to a model."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionResult(BaseModel):
    """The answer and token counts returned by a provider."""

    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False
