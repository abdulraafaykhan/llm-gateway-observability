"""Builds the send times for a load test scenario (random, Poisson-style arrivals)."""

import random

# Each phase is (rate in requests per second, duration in seconds).
SCENARIOS: dict[str, list[tuple[float, float]]] = {
    "smoke": [(0.5, 40)],
    "normal": [(2, 60)],
    "spike": [(2, 20), (10, 20), (2, 20)],
}


def build_schedule(scenario: str, seed: int) -> list[float]:
    """Return ascending send times (seconds from start) for every request in the scenario."""
    if scenario not in SCENARIOS:
        valid = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown scenario '{scenario}'. Valid scenarios: {valid}")
    rng = random.Random(seed)
    send_times: list[float] = []
    phase_start = 0.0
    for rate, duration in SCENARIOS[scenario]:
        phase_end = phase_start + duration
        current = phase_start + rng.expovariate(rate)
        while current < phase_end:
            send_times.append(current)
            current += rng.expovariate(rate)
        phase_start = phase_end
    return send_times
