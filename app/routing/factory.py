"""Builds the rule router from the routing config."""

from app.models.routing import RoutingConfig
from app.routing.rule_router import RuleRouter
from app.routing.rules import (
    KeywordsRule,
    LongQuestionRule,
    ManyNumbersRule,
    MultipleQuestionsRule,
    RoutingRule,
)


def build_rule_router(config: RoutingConfig, threshold: int | None = None) -> RuleRouter:
    """Build the four rules from the config; threshold overrides config.threshold when given."""
    rules: list[RoutingRule] = [
        LongQuestionRule(config.medium_words, config.long_words),
        KeywordsRule(config.keywords_points, config.keywords),
        ManyNumbersRule(config.many_numbers_points, config.min_numbers),
        MultipleQuestionsRule(config.multiple_questions_points, config.min_question_marks),
    ]
    return RuleRouter(
        rules=rules,
        small_model=config.small_model,
        big_model=config.big_model,
        threshold=config.threshold if threshold is None else threshold,
    )
