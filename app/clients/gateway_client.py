"""HTTP implementation of the gateway client."""

from typing import Any

import httpx

from app.clients.base import GatewayClient
from app.models.eval import GatewayReply

ERROR_TEXT_LIMIT: int = 300


class HttpGatewayClient(GatewayClient):
    """Calls POST /v1/chat on a running gateway using a shared httpx.AsyncClient."""

    def __init__(
        self, base_url: str, http_client: httpx.AsyncClient, timeout_seconds: float = 120.0
    ) -> None:
        self._chat_url: str = f"{base_url.rstrip('/')}/v1/chat"
        self._http_client: httpx.AsyncClient = http_client
        self._timeout_seconds: float = timeout_seconds

    async def ask(self, question: str, model: str) -> GatewayReply:
        """Send one question; connection errors and timeouts come back as http_status 0."""
        body = {"messages": [{"role": "user", "content": question}], "model": model}
        try:
            response = await self._http_client.post(
                self._chat_url, json=body, timeout=self._timeout_seconds
            )
        except httpx.TimeoutException:
            return _failed_reply(0, f"timed out after {self._timeout_seconds} seconds")
        except httpx.TransportError as error:
            return _failed_reply(0, f"connection error: {type(error).__name__}")
        if response.status_code != 200:
            return _failed_reply(response.status_code, _error_text(response))
        return _success_reply(response)


def _success_reply(response: httpx.Response) -> GatewayReply:
    """Build a reply from a 200 response body."""
    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        reply = _failed_reply(200, "gateway returned a 200 response that is not valid JSON")
        return reply
    return GatewayReply(
        http_status=200,
        answer=data.get("answer"),
        model=data.get("model"),
        input_tokens=int(data.get("input_tokens", 0)),
        output_tokens=int(data.get("output_tokens", 0)),
        latency_ms=data.get("latency_ms"),
        truncated=bool(data.get("truncated", False)),
        error_message=None,
        cost_usd=float(data.get("cost_usd", 0.0)),
        route_reason=str(data.get("route_reason", "")),
        fallback_used=bool(data.get("fallback_used", False)),
    )


def _failed_reply(http_status: int, error_message: str) -> GatewayReply:
    """Build a reply for a failed call."""
    return GatewayReply(
        http_status=http_status,
        answer=None,
        model=None,
        input_tokens=0,
        output_tokens=0,
        latency_ms=None,
        truncated=False,
        error_message=error_message,
    )


def _error_text(response: httpx.Response) -> str:
    """Return error.message from the JSON body, or the raw text."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:ERROR_TEXT_LIMIT]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return response.text[:ERROR_TEXT_LIMIT]
