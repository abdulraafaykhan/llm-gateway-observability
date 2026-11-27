"""Lists the Gemini models this key can use and probes them with one tiny request each."""

import time
from typing import Any, NamedTuple

import httpx

from app.models.completion import Message
from app.providers.base import ProviderError
from app.providers.gemini_provider import (
    API_KEY_HEADER,
    ERROR_DETAIL_LIMIT,
    GEMINI_API_BASE_URL,
    GeminiProvider,
    error_detail,
)

MODELS_PAGE_SIZE: int = 1000
MODEL_ID_PREFIX: str = "models/"
GENERATE_METHOD: str = "generateContent"
FREE_TIER_REMINDER: str = (
    "The API does not say which models are free. Check your limits in Google AI Studio "
    "(rate limits page) before choosing."
)
PROBE_TEXT: str = "Reply with the word OK."
TABLE_HEADERS: tuple[str, ...] = ("id", "display name", "input limit", "output limit", "note")


class GeminiModelInfo(NamedTuple):
    """One model from the Gemini list endpoint, with the "models/" prefix removed."""

    model_id: str
    display_name: str
    input_token_limit: int | None
    output_token_limit: int | None
    methods: tuple[str, ...]


class ProbeResult(NamedTuple):
    """The outcome of one probe request."""

    model_id: str
    ok: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    error: str | None


class GeminiModelsError(Exception):
    """The list request failed; the message never contains the API key."""


def parse_models_page(data: Any) -> tuple[list[GeminiModelInfo], str | None]:
    """Return the models on one list page and the token for the next page, if any."""
    data = data if isinstance(data, dict) else {}
    models = [_parse_model(entry) for entry in data.get("models") or [] if isinstance(entry, dict)]
    return models, data.get("nextPageToken") or None


def _parse_model(entry: dict[str, Any]) -> GeminiModelInfo:
    """Turn one list entry into a GeminiModelInfo."""
    name = str(entry.get("name", ""))
    model_id = name.removeprefix(MODEL_ID_PREFIX)
    return GeminiModelInfo(
        model_id=model_id,
        display_name=str(entry.get("displayName") or ""),
        input_token_limit=entry.get("inputTokenLimit"),
        output_token_limit=entry.get("outputTokenLimit"),
        methods=tuple(entry.get("supportedGenerationMethods") or ()),
    )


def generate_content_models(models: list[GeminiModelInfo]) -> list[GeminiModelInfo]:
    """Keep only models that support generateContent, sorted by id."""
    usable = [model for model in models if GENERATE_METHOD in model.methods]
    return sorted(usable, key=lambda model: model.model_id)


def model_note(model_id: str) -> str:
    """Return "preview" or "experimental" when the id says so, else an empty note."""
    notes = []
    if "preview" in model_id:
        notes.append("preview")
    if "exp" in model_id:
        notes.append("experimental")
    return ", ".join(notes)


def render_models_table(models: list[GeminiModelInfo]) -> str:
    """Render the usable models as a text table, ending with the free-tier reminder."""
    rows = [
        (
            model.model_id,
            model.display_name,
            _limit_text(model.input_token_limit),
            _limit_text(model.output_token_limit),
            model_note(model.model_id),
        )
        for model in generate_content_models(models)
    ]
    widths = [max(len(row[i]) for row in [TABLE_HEADERS, *rows]) for i in range(len(TABLE_HEADERS))]
    lines = [
        _table_line(TABLE_HEADERS, widths),
        _table_line(tuple("-" * w for w in widths), widths),
    ]
    lines.extend(_table_line(row, widths) for row in rows)
    if not rows:
        lines.append("(no models support generateContent)")
    lines.extend(["", FREE_TIER_REMINDER])
    return "\n".join(lines)


def _limit_text(limit: int | None) -> str:
    """Show a token limit, or "?" when the API did not give one."""
    return str(limit) if limit is not None else "?"


def _table_line(cells: tuple[str, ...], widths: list[int]) -> str:
    """Pad each cell to its column width."""
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()


async def fetch_models(
    api_key: str,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[GeminiModelInfo]:
    """Fetch every page of the model list; the key goes in a header, never in the URL."""
    models: list[GeminiModelInfo] = []
    params: dict[str, Any] = {"pageSize": MODELS_PAGE_SIZE}
    async with httpx.AsyncClient(
        base_url=GEMINI_API_BASE_URL,
        headers={API_KEY_HEADER: api_key},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        while True:
            try:
                response = await client.get("/models", params=params)
            except httpx.HTTPError as error:
                message = f"Could not list Gemini models: {type(error).__name__}"
                raise GeminiModelsError(message) from error
            if response.status_code != 200:
                detail = error_detail(response).replace(api_key, "[hidden]")
                raise GeminiModelsError(
                    f"Gemini API error (HTTP {response.status_code}): {detail[:ERROR_DETAIL_LIMIT]}"
                )
            page, next_token = parse_models_page(response.json())
            models.extend(page)
            if next_token is None:
                return models
            params = {"pageSize": MODELS_PAGE_SIZE, "pageToken": next_token}


def probe_messages() -> list[Message]:
    """Return the single user message every probe sends."""
    return [Message(role="user", content=PROBE_TEXT)]


async def probe_model(provider: GeminiProvider, model_id: str) -> ProbeResult:
    """Send one tiny request through the provider and time it; failures are reported, not raised."""
    started = time.perf_counter()
    try:
        answer = await provider.generate(model_id, probe_messages())
    except ProviderError as error:
        return ProbeResult(
            model_id=model_id,
            ok=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=error.input_tokens,
            output_tokens=error.output_tokens,
            finish_reason=error.finish_reason,
            error=str(error),
        )
    return ProbeResult(
        model_id=model_id,
        ok=True,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        finish_reason=answer.finish_reason,
        error=None,
    )


def render_probe_line(result: ProbeResult) -> str:
    """One line per probe: id, OK/FAILED, latency, tokens (output includes thinking), reason."""
    line = (
        f"{result.model_id}  {'OK' if result.ok else 'FAILED'}  "
        f"latency={result.latency_ms:.0f}ms  input_tokens={result.input_tokens}  "
        f"output_tokens={result.output_tokens} (incl. thinking)  "
        f"finish_reason={result.finish_reason}"
    )
    return line if result.error is None else f"{line}  error={result.error}"
