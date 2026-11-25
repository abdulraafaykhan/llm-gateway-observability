"""Simulated provider for load testing: realistic delays, canned answers, queueing, failures."""

import asyncio
import random
from typing import NamedTuple

from app.models.completion import CompletionResult, Message
from app.models.fake_profile import FakeProfile
from app.models.model_spec import ModelSpec
from app.providers.base import LLMProvider, ProviderError

WORDS_TO_TOKENS: float = 1.3


class FakeDraw(NamedTuple):
    """The random numbers drawn for one simulated call."""

    error_roll: float
    truncate_roll: float
    output_tokens: int
    jitter_factor: float
    outlier_roll: float


class FakeProvider(LLMProvider):
    """Pretends to be the models, with a per-model limit on concurrent requests."""

    def __init__(self, profiles: dict[str, FakeProfile], seed: int) -> None:
        self._profiles: dict[str, FakeProfile] = profiles
        self._random: random.Random = random.Random(seed)
        self._semaphores: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(profile.max_concurrent_requests)
            for name, profile in profiles.items()
        }

    async def complete(self, model_spec: ModelSpec, messages: list[Message]) -> CompletionResult:
        """Wait a simulated time (queueing when busy), then fail, cut off, or answer."""
        profile = self._profiles.get(model_spec.name)
        if profile is None:
            raise ProviderError(f"no fake profile for model {model_spec.name}")
        draw = self._draw(profile)
        latency_ms = simulated_latency_ms(profile, draw)
        async with self._semaphores[model_spec.name]:
            if draw.error_roll < profile.error_probability:
                await asyncio.sleep(profile.base_latency_ms / 1000)
                raise ProviderError("simulated provider failure")
            await asyncio.sleep(latency_ms / 1000)
        truncated = draw.truncate_roll < profile.truncate_probability
        return CompletionResult(
            text=f"[simulated answer from {model_spec.name}]",
            input_tokens=count_input_tokens(messages),
            output_tokens=profile.max_output_tokens if truncated else draw.output_tokens,
            truncated=truncated,
        )

    def _draw(self, profile: FakeProfile) -> FakeDraw:
        """Draw every random number for one call, always in the same order."""
        jitter = profile.jitter_percent / 100
        return FakeDraw(
            error_roll=self._random.random(),
            truncate_roll=self._random.random(),
            output_tokens=self._random.randint(
                profile.min_output_tokens, profile.max_output_tokens
            ),
            jitter_factor=self._random.uniform(1 - jitter, 1 + jitter),
            outlier_roll=self._random.random(),
        )


def simulated_latency_ms(profile: FakeProfile, draw: FakeDraw) -> float:
    """Return the simulated time for one call, including jitter and rare slow outliers."""
    latency_ms = (
        profile.base_latency_ms + profile.ms_per_output_token * draw.output_tokens
    ) * draw.jitter_factor
    if draw.outlier_roll < profile.slow_outlier_probability:
        latency_ms *= profile.slow_outlier_multiplier
    return latency_ms


def count_input_tokens(messages: list[Message]) -> int:
    """Estimate input tokens as 1.3 per whitespace-separated word, at least 1."""
    word_count = sum(len(message.content.split()) for message in messages)
    return max(1, round(WORDS_TO_TOKENS * word_count))
