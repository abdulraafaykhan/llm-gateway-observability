"""Router for the chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_chat_service
from app.services.chat_service import ChatService
from app.views.chat_view import ChatRequest, ChatResponse
from app.views.error_view import ErrorResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def chat(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Answer a chat request with the chosen model."""
    result, model_spec, latency_ms = await chat_service.chat(request.messages, request.model)
    return ChatResponse.from_completion(result, model_spec.name, latency_ms)
