from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    """Static contract for an operator-routed numerical skill."""

    name: str
    layer: str
    intent: str
    required_slots: frozenset[str]
    optional_slots: frozenset[str]
    executor_operator: str | None = None
    output_type: str = "numeric"
    description: str = ""

    @property
    def canonical_executor_operator(self) -> str:
        return self.executor_operator or self.name


COMMON_FILTER_SLOTS = frozenset({"filters", "unit", "year", "industry", "entity"})
FIELD_AGGREGATE_OPTIONAL = frozenset({"filters", "unit", "year", "industry"})


OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "LOOKUP": OperatorSpec(
        name="LOOKUP",
        layer="atomic",
        intent="lookup",
        required_slots=frozenset({"entity", "target_field"}),
        optional_slots=frozenset({"filters", "unit", "year", "industry"}),
        description="Lookup one entity-field value under optional time or metadata filters.",
    ),
    "YEAR_LIST": OperatorSpec(
        name="YEAR_LIST",
        layer="atomic",
        intent="lookup",
        required_slots=frozenset({"target_field"}),
        optional_slots=frozenset({"filters", "industry"}),
        description="Return the fiscal years represented by a grounded field.",
    ),
    "SUM": OperatorSpec(
        name="SUM",
        layer="atomic",
        intent="aggregation",
        required_slots=frozenset({"target_field"}),
        optional_slots=FIELD_AGGREGATE_OPTIONAL,
        description="Sum values for a grounded field over a filtered record set.",
    ),
    "COUNT": OperatorSpec(
        name="COUNT",
        layer="atomic",
        intent="condition-counting",
        required_slots=frozenset({"count_target", "condition_field"}),
        optional_slots=frozenset(
            {
                "filters",
                "unit",
                "year",
                "industry",
                "condition_op",
                "condition_threshold",
            }
        ),
        description="Count records or distinct entities satisfying an optional condition.",
    ),
    "AVG": OperatorSpec(
        name="AVG",
        layer="atomic",
        intent="aggregation",
        required_slots=frozenset({"target_field"}),
        optional_slots=FIELD_AGGREGATE_OPTIONAL,
        description="Average values for a grounded field over a filtered record set.",
    ),
    "PRODUCT": OperatorSpec(
        name="PRODUCT",
        layer="derived",
        intent="multiplicative-composition",
        required_slots=frozenset(),
        optional_slots=frozenset(
            {
                "factor_token_ids",
                "factor_fields",
                "target_field",
                "filters",
                "unit",
                "year",
                "industry",
                "entity",
            }
        ),
        description="Multiply two or more grounded numeric factors, preserving compound-unit metadata.",
    ),
    "MIN": OperatorSpec(
        name="MIN",
        layer="atomic",
        intent="comparison",
        required_slots=frozenset({"target_field"}),
        optional_slots=FIELD_AGGREGATE_OPTIONAL,
        description="Return the minimum value for a grounded field.",
    ),
    "MAX": OperatorSpec(
        name="MAX",
        layer="atomic",
        intent="comparison",
        required_slots=frozenset({"target_field"}),
        optional_slots=FIELD_AGGREGATE_OPTIONAL,
        description="Return the maximum value for a grounded field.",
    ),
    "ARGMIN": OperatorSpec(
        name="ARGMIN",
        layer="atomic",
        intent="ranking",
        required_slots=frozenset({"target_entity_type", "target_field"}),
        optional_slots=FIELD_AGGREGATE_OPTIONAL,
        description="Return the entity with the minimum value for a grounded field.",
    ),
    "ARGMAX": OperatorSpec(
        name="ARGMAX",
        layer="atomic",
        intent="ranking",
        required_slots=frozenset({"target_entity_type", "target_field"}),
        optional_slots=FIELD_AGGREGATE_OPTIONAL,
        description="Return the entity with the maximum value for a grounded field.",
    ),
    "RATIO": OperatorSpec(
        name="RATIO",
        layer="derived",
        intent="ratio",
        required_slots=frozenset({"numerator_field", "denominator_field"}),
        optional_slots=COMMON_FILTER_SLOTS,
        description="Divide an aggregated numerator field by an aggregated denominator field.",
    ),
    "SHARE": OperatorSpec(
        name="SHARE",
        layer="derived",
        intent="ratio",
        required_slots=frozenset({"numerator_field", "denominator_field"}),
        optional_slots=COMMON_FILTER_SLOTS,
        executor_operator="RATIO",
        description="Compute a part-over-whole share as a ratio.",
    ),
    "MARGIN": OperatorSpec(
        name="MARGIN",
        layer="derived",
        intent="ratio",
        required_slots=frozenset({"numerator_field", "denominator_field"}),
        optional_slots=COMMON_FILTER_SLOTS,
        executor_operator="RATIO",
        description="Compute a profit or operating margin as a ratio.",
    ),
    "GROWTH": OperatorSpec(
        name="GROWTH",
        layer="derived",
        intent="change-over-time",
        required_slots=frozenset({"target_field", "from_time", "to_time"}),
        optional_slots=COMMON_FILTER_SLOTS,
        description="Compute (to - from) / from for one field across two times.",
    ),
    "PERCENT_CHANGE": OperatorSpec(
        name="PERCENT_CHANGE",
        layer="derived",
        intent="change-over-time",
        required_slots=frozenset({"target_field", "from_time", "to_time"}),
        optional_slots=COMMON_FILTER_SLOTS,
        executor_operator="GROWTH",
        description="Alias of growth with explicit percentage-change intent.",
    ),
    "DIFFERENCE": OperatorSpec(
        name="DIFFERENCE",
        layer="derived",
        intent="comparison",
        required_slots=frozenset({"target_field"}),
        optional_slots=frozenset(
            {
                "filters",
                "unit",
                "year",
                "industry",
                "left_entity",
                "right_entity",
                "left_time",
                "right_time",
            }
        ),
        description="Subtract a right value from a left value.",
    ),
    "TOP_K": OperatorSpec(
        name="TOP_K",
        layer="composite",
        intent="ranking",
        required_slots=frozenset({"target_entity_type", "target_field", "k"}),
        optional_slots=frozenset({"filters", "unit", "year", "industry", "order"}),
        description="Return the top-k record set and k-th cutoff value.",
    ),
    "RANK": OperatorSpec(
        name="RANK",
        layer="composite",
        intent="ranking",
        required_slots=frozenset({"target_entity_type", "target_field", "k"}),
        optional_slots=frozenset({"filters", "unit", "year", "industry", "order"}),
        executor_operator="TOP_K",
        description="Ranking-list alias backed by TOP_K execution.",
    ),
    "COMPARE": OperatorSpec(
        name="COMPARE",
        layer="composite",
        intent="comparison",
        required_slots=frozenset({"target_field"}),
        optional_slots=frozenset(
            {
                "filters",
                "unit",
                "year",
                "industry",
                "left_entity",
                "right_entity",
                "comparison_type",
            }
        ),
        executor_operator="DIFFERENCE",
        description="Compare two entities or records over the same field.",
    ),
    "TREND": OperatorSpec(
        name="TREND",
        layer="composite",
        intent="change-over-time",
        required_slots=frozenset({"target_field"}),
        optional_slots=frozenset({"filters", "unit", "entity", "industry", "from_time", "to_time"}),
        description="Analyze direction over a value series.",
    ),
    "PREDICT": OperatorSpec(
        name="PREDICT",
        layer="predictive",
        intent="prediction",
        required_slots=frozenset({"target_field", "entity", "history_range", "horizon"}),
        optional_slots=frozenset({"filters", "unit", "method", "selector_config", "forecast_profile", "industry"}),
        description="Predict a future value from a historical series.",
    ),
    "FORECAST": OperatorSpec(
        name="FORECAST",
        layer="predictive",
        intent="prediction",
        required_slots=frozenset({"target_field", "entity", "history_range", "horizon"}),
        optional_slots=frozenset({"filters", "unit", "method", "selector_config", "forecast_profile", "industry"}),
        executor_operator="PREDICT",
        description="Forecast alias backed by PREDICT execution.",
    ),
}


PUBLIC_OPERATORS: tuple[str, ...] = tuple(OPERATOR_SPECS)
EXECUTOR_OPERATOR_ALIASES: dict[str, str] = {
    name: spec.canonical_executor_operator
    for name, spec in OPERATOR_SPECS.items()
    if spec.canonical_executor_operator != name
}


class OperatorRegistry:
    """Read-only registry for operator contracts shared across modules."""

    def __init__(self, specs: dict[str, OperatorSpec] | None = None) -> None:
        self._specs = dict(specs or OPERATOR_SPECS)

    def get(self, operator: str) -> OperatorSpec:
        try:
            return self._specs[operator]
        except KeyError as exc:
            raise KeyError(f"Unknown operator: {operator}") from exc

    def required_slots(self, operator: str) -> frozenset[str]:
        return self.get(operator).required_slots

    def optional_slots(self, operator: str) -> frozenset[str]:
        return self.get(operator).optional_slots

    def canonical_executor_operator(self, operator: str) -> str:
        return self.get(operator).canonical_executor_operator

    def intent(self, operator: str) -> str:
        return self.get(operator).intent

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)


OPERATOR_REGISTRY = OperatorRegistry()
