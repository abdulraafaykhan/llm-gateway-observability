"""Router that adds up rule points and picks big when the score reaches the threshold."""

from app.models.completion import Message
from app.models.routing import RouteDecision, RuleHit
from app.routing.base import Router
from app.routing.rules import RoutingRule


class RuleRouter(Router):
    """Scores the last user message with simple rules."""

    def __init__(
        self, rules: list[RoutingRule], small_model: str, big_model: str, threshold: int
    ) -> None:
        self._rules: list[RoutingRule] = rules
        self._small_model: str = small_model
        self._big_model: str = big_model
        self._threshold: int = threshold

    def choose(self, messages: list[Message]) -> RouteDecision:
        """Pick big when the total score is at or above the threshold, otherwise small."""
        text = routed_text(messages)
        hits = [hit for rule in self._rules if (hit := rule.score(text)) is not None]
        score = sum(hit.points for hit in hits)
        model_name = self._big_model if score >= self._threshold else self._small_model
        return RouteDecision(
            model_name=model_name,
            score=score,
            threshold=self._threshold,
            hits=hits,
            reason=describe_route(score, self._threshold, hits),
        )


def routed_text(messages: list[Message]) -> str:
    """Return the last user message's content, or the last message's content if none."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


def describe_route(score: int, threshold: int, hits: list[RuleHit]) -> str:
    """Explain the decision, for example 'score 2 (threshold 2): keywords +1 [compare], ...'."""
    prefix = f"score {score} (threshold {threshold}): "
    if not hits:
        return prefix + "no rules matched"
    return prefix + ", ".join(render_hit(hit) for hit in hits)


def render_hit(hit: RuleHit) -> str:
    """Render one rule hit as 'name +points [detail]'."""
    text = f"{hit.rule} +{hit.points}"
    return f"{text} [{hit.detail}]" if hit.detail else text
