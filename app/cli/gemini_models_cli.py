"""Command line: list the Gemini models your key can use, or probe some with one tiny request.

Run with: uv run python -m app.cli.gemini_models_cli [--probe MODEL_ID [MODEL_ID ...]]
"""

import argparse
import asyncio
import sys

from app.config import Settings, get_settings
from app.providers.gemini_provider import GeminiProvider
from app.services.gemini_models_service import (
    GeminiModelsError,
    fetch_models,
    probe_model,
    render_models_table,
    render_probe_line,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="List or probe the Gemini models for your key.")
    parser.add_argument(
        "--probe",
        nargs="+",
        metavar="MODEL_ID",
        help='send "Reply with the word OK." to each model id and report the result',
    )
    return parser.parse_args(argv)


async def list_models_text(settings: Settings, api_key: str) -> str:
    """Fetch the model list and return the rendered table."""
    models = await fetch_models(api_key, settings.request_timeout_seconds)
    return render_models_table(models)


async def probe_lines(settings: Settings, api_key: str, model_ids: list[str]) -> list[str]:
    """Probe each model id in turn and return one line per model."""
    provider = GeminiProvider(
        api_key=api_key,
        timeout_seconds=settings.request_timeout_seconds,
        max_output_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
    )
    return [render_probe_line(await probe_model(provider, model_id)) for model_id in model_ids]


def main(argv: list[str] | None = None) -> int:
    """Run the list or probe command and return the process exit code."""
    args = parse_args(argv)
    settings = get_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set; add it to .env first.", file=sys.stderr)
        return 1
    if args.probe:
        print("\n".join(asyncio.run(probe_lines(settings, settings.gemini_api_key, args.probe))))
        return 0
    try:
        print(asyncio.run(list_models_text(settings, settings.gemini_api_key)))
    except GeminiModelsError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
