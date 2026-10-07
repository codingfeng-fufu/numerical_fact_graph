"""Post-execution verifier for OperatorPlan results.

Runs 5 deterministic checks against the ExecutionResult and populates
result.checks with a uniform dict[str, bool].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from graph_numeric.core.attribute_graph import token_matches_year
from graph_numeric.operators.operator_registry import OPERATOR_REGISTRY
from graph_numeric.core.unit_resolver import UnitInfo, UnitResolver

if TYPE_CHECKING:
    from graph_numeric.core.attribute_graph import AttributeValueGraph
    from graph_numeric.operators.executor import ExecutionResult
    from graph_numeric.operators.operator_plan import OperatorPlan

_TOLERANCE = 1e-6

_COMPUTATIONAL_CHECKS = frozenset({
    "plan_valid",
    "field_exists",
    "tokens_nonempty",
    "tokens_field_consistent",
    "filter_satisfied",
    "unit_compatible",
    "arithmetic_verified",
    "output_type_valid",
})

_CHECK_CATEGORIES: dict[str, str] = {
    "plan_valid": "planning",
    "field_exists": "grounding",
    "tokens_nonempty": "grounding",
    "tokens_field_consistent": "grounding",
    "filter_satisfied": "filter",
    "unit_compatible": "unit",
    "arithmetic_verified": "arithmetic",
    "output_type_valid": "planning",
    "evidence_trace_complete": "evidence",
}


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of a single verification run."""

    checks: dict[str, bool]
    warnings: list[str]
    passed: bool  # True iff all checks are True

    def to_dict(self) -> dict[str, Any]:
        failed_checks = [name for name, ok in self.checks.items() if not ok]
        grouped: dict[str, dict[str, bool]] = {}
        for name, ok in self.checks.items():
            category = _CHECK_CATEGORIES.get(name, "other")
            grouped.setdefault(category, {})[name] = ok
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "warnings": list(self.warnings),
            "failed_checks": failed_checks,
            "failure_categories": sorted({
                _CHECK_CATEGORIES.get(name, "other")
                for name in failed_checks
            }),
            "grouped_checks": grouped,
        }


class Verifier:
    """Verifies an ExecutionResult against the originating plan and graph.

    Checks (all operator-agnostic keys, always present):
      plan_valid              — all required slots are grounded
      field_exists            — every grounded field name is in graph.fields
      tokens_nonempty         — selected_tokens non-empty when answer > 0
      tokens_field_consistent — token field_names match the plan's field slots
      filter_satisfied        — selected tokens satisfy grounded time/entity filters
      unit_compatible         — selected token units are compatible with operator semantics
      arithmetic_verified     — recomputed answer matches result.answer
      output_type_valid       — answer type matches the operator output contract
    """

    def verify(
        self,
        graph: AttributeValueGraph,
        plan: OperatorPlan,
        result: ExecutionResult,
    ) -> VerificationReport:
        warnings: list[str] = []
        checks: dict[str, bool] = {}

        ok, w = self._check_plan_valid(plan)
        checks["plan_valid"] = ok
        warnings.extend(w)

        ok, w = self._check_field_exists(graph, plan)
        checks["field_exists"] = ok
        warnings.extend(w)

        ok, w = self._check_tokens_nonempty(result)
        checks["tokens_nonempty"] = ok
        warnings.extend(w)

        ok, w = self._check_tokens_field_consistent(plan, result)
        checks["tokens_field_consistent"] = ok
        warnings.extend(w)

        ok, w = self._check_filter_satisfied(plan, result)
        checks["filter_satisfied"] = ok
        warnings.extend(w)

        ok, w = self._check_unit_compatible(plan, result)
        checks["unit_compatible"] = ok
        warnings.extend(w)

        ok, w = self._check_arithmetic(plan, result)
        checks["arithmetic_verified"] = ok
        warnings.extend(w)

        ok, w = self._check_output_type(plan, result)
        checks["output_type_valid"] = ok
        warnings.extend(w)

        ok, w = self._check_evidence_trace(result)
        checks["evidence_trace_complete"] = ok
        warnings.extend(w)

        return VerificationReport(
            checks=checks,
            warnings=warnings,
            passed=all(v for k, v in checks.items() if k in _COMPUTATIONAL_CHECKS),
        )

    # ---- individual checks ----

    def _check_plan_valid(self, plan: OperatorPlan) -> tuple[bool, list[str]]:
        if plan.is_valid():
            return True, []
        missing = plan.missing_required_slots()
        return False, [f"plan_valid: missing required slots {missing}"]

    def _check_field_exists(
        self, graph: AttributeValueGraph, plan: OperatorPlan
    ) -> tuple[bool, list[str]]:
        op = plan.operator
        graph_fields = set(graph.fields)
        warnings: list[str] = []

        field_slot_keys: list[str] = []
        if op in (
            "SUM",
            "AVG",
            "MAX",
            "MIN",
            "ARGMAX",
            "ARGMIN",
            "DIFFERENCE",
            "COMPARE",
            "GROWTH",
            "PERCENT_CHANGE",
            "LOOKUP",
            "YEAR_LIST",
            "TOP_K",
            "RANK",
            "TREND",
            "PREDICT",
            "FORECAST",
        ):
            field_slot_keys = ["target_field"]
        elif op in ("DIFFERENCE", "COMPARE"):
            left_field = _get_slot_value(plan, "left_field")
            right_field = _get_slot_value(plan, "right_field")
            if left_field is not None and right_field is not None:
                field_slot_keys = ["left_field", "right_field"]
            else:
                field_slot_keys = ["target_field"]
        elif op == "COUNT":
            field_slot_keys = ["condition_field"]
        elif op in ("RATIO", "SHARE", "MARGIN"):
            field_slot_keys = ["numerator_field", "denominator_field"]

        expected_values: list[str] = []
        all_ok = True
        for key in field_slot_keys:
            val = _get_slot_value(plan, key)
            if val is None or val == "__step_result__":
                continue
            expected_values.append(str(val))
        target_fields = _get_slot_value(plan, "target_fields")
        if op == "SUM" and isinstance(target_fields, (list, tuple, set, frozenset)):
            expected_values.extend(str(field) for field in target_fields)
        for val in dict.fromkeys(expected_values):
            if val not in graph_fields:
                all_ok = False
                warnings.append(f"field_exists: '{val}' not in graph fields {sorted(graph_fields)}")
        return all_ok, warnings

    def _check_tokens_nonempty(self, result: ExecutionResult) -> tuple[bool, list[str]]:
        # Empty tokens are only valid when the answer is legitimately 0
        if len(result.selected_tokens) == 0 and _answer_is_nonzero(result.answer):
            return False, [
                f"tokens_nonempty: 0 tokens but answer={result.answer}"
            ]
        return True, []

    def _check_tokens_field_consistent(
        self, plan: OperatorPlan, result: ExecutionResult
    ) -> tuple[bool, list[str]]:
        op = plan.operator
        tokens = result.selected_tokens
        if not tokens:
            return True, []
        proposal_ids = _proposal_bound_token_ids(plan)
        if proposal_ids:
            unexpected = [token.token_id for token in tokens if token.token_id not in proposal_ids]
            if unexpected:
                return False, [
                    "tokens_field_consistent: selected token(s) outside S4prime proposal "
                    f"{unexpected}"
                ]
            return True, []

        if op in ("RATIO", "SHARE", "MARGIN"):
            num_field = _get_slot_value(plan, "numerator_field")
            den_field = _get_slot_value(plan, "denominator_field")
            expected_fields = {f for f in (num_field, den_field) if f}
            numerator_fields = _get_slot_value(plan, "numerator_fields")
            if isinstance(numerator_fields, (list, tuple, set, frozenset)):
                expected_fields.update(str(field) for field in numerator_fields)
            bad = [t for t in tokens if t.field_name not in expected_fields]
        elif op in ("DIFFERENCE", "COMPARE"):
            left_field = _get_slot_value(plan, "left_field")
            right_field = _get_slot_value(plan, "right_field")
            if left_field is not None and right_field is not None:
                expected_fields = {f for f in (left_field, right_field) if f}
                bad = [t for t in tokens if t.field_name not in expected_fields]
            else:
                target = _get_slot_value(plan, "target_field")
                bad = [t for t in tokens if t.field_name != target] if target else []
        elif op in ("GROWTH", "PERCENT_CHANGE"):
            target = _get_slot_value(plan, "target_field")
            bad = [t for t in tokens if t.field_name != target] if target else []
        else:
            key = "condition_field" if op == "COUNT" else "target_field"
            target = _get_slot_value(plan, key)
            target_fields = _get_slot_value(plan, "target_fields")
            if op == "SUM" and isinstance(target_fields, (list, tuple, set, frozenset)):
                expected_fields = {str(field) for field in target_fields}
                bad = [t for t in tokens if t.field_name not in expected_fields]
            else:
                bad = [t for t in tokens if t.field_name != target] if target else []

        if bad:
            unexpected = {t.field_name for t in bad}
            return False, [
                f"tokens_field_consistent: unexpected field(s) {unexpected} in selected_tokens"
            ]
        return True, []

    def _check_filter_satisfied(
        self, plan: OperatorPlan, result: ExecutionResult
    ) -> tuple[bool, list[str]]:
        tokens = result.selected_tokens
        if not tokens:
            return True, []
        if _proposal_bound_token_ids(plan):
            return True, []
        warnings: list[str] = []
        years = _expected_years(plan)
        if years:
            bad = [
                t.token_id
                for t in tokens
                if not any(token_matches_year(t, year) for year in years)
            ]
            if bad:
                warnings.append(
                    f"filter_satisfied: {len(bad)} token(s) outside expected year(s) {sorted(years)}"
                )
        industry = _get_slot_value(plan, "industry")
        if industry is not None:
            bad = [t.token_id for t in tokens if t.industry != industry]
            if bad:
                warnings.append(
                    f"filter_satisfied: {len(bad)} token(s) outside industry '{industry}'"
                )
        entity = _get_slot_value(plan, "entity")
        if entity is not None:
            bad = [t.token_id for t in tokens if not _matches_entity(t, entity)]
            if bad:
                warnings.append(
                    f"filter_satisfied: {len(bad)} token(s) outside entity '{entity}'"
                )
        if plan.operator == "COUNT":
            condition_field = _get_slot_value(plan, "condition_field")
            condition_op = _get_slot_value(plan, "condition_op") or ">="
            condition_threshold = _get_slot_value(plan, "condition_threshold")
            if condition_field is not None and condition_threshold is not None:
                threshold = float(condition_threshold)
                condition_unit = _get_slot_value(plan, "condition_unit") or _get_slot_value(plan, "unit")
                values_by_token = _normalized_values_by_token(tokens, condition_unit)
                bad = [
                    t.token_id
                    for t in tokens
                    if t.field_name == condition_field
                    and not _satisfies_threshold(
                        values_by_token.get(t.token_id, float(t.value)),
                        str(condition_op),
                        threshold,
                    )
                ]
                if bad:
                    warnings.append(
                        "filter_satisfied: "
                        f"{len(bad)} COUNT token(s) violate "
                        f"{condition_field} {condition_op} {threshold}"
                    )
        for slot_key in (
            "dimension_filters",
            "numerator_dimension_filters",
            "denominator_dimension_filters",
        ):
            dimensions = _get_slot_value(plan, slot_key)
            if dimensions is None:
                continue
            scoped_tokens = _tokens_for_dimension_slot(plan, result, slot_key)
            bad = [
                t.token_id
                for t in scoped_tokens
                if not _token_matches_dimensions(t, dimensions)
            ]
            if bad:
                warnings.append(
                    f"filter_satisfied: {len(bad)} token(s) violate {slot_key}"
                )
        return not warnings, warnings

    def _check_unit_compatible(
        self, plan: OperatorPlan, result: ExecutionResult
    ) -> tuple[bool, list[str]]:
        units = _units_for_check(plan, result)
        if not units:
            return True, []
        canonical_op = OPERATOR_REGISTRY.canonical_executor_operator(plan.operator)
        warnings = UnitResolver().check_compatibility(canonical_op, units)
        return len(warnings) == 0, [f"unit_compatible: {warning}" for warning in warnings]

    def _check_arithmetic(
        self, plan: OperatorPlan, result: ExecutionResult
    ) -> tuple[bool, list[str]]:
        op = OPERATOR_REGISTRY.canonical_executor_operator(plan.operator)
        tokens = result.selected_tokens
        expected = result.answer
        output_slot = str(_get_slot_value(plan, "output_slot") or "").lower()

        if op in {"ARGMAX", "ARGMIN"} and output_slot in {"entity", "company", "company_name"}:
            if not tokens:
                return False, [f"arithmetic_verified: no tokens for {op} entity output"]
            expected_entity = (tokens[0].dimensions or {}).get("segment_label")
            if expected_entity is None:
                expected_entity = (tokens[0].dimensions or {}).get("row_label")
            if expected_entity is None:
                expected_entity = tokens[0].company_name or tokens[0].raw_label
            if str(result.answer) == str(expected_entity):
                return True, []
            return False, [
                f"arithmetic_verified: selected entity {expected_entity} != result.answer {result.answer}"
            ]

        try:
            recomputed = self._recompute(op, plan, result)
        except Exception as exc:
            return False, [f"arithmetic_verified: recomputation failed: {exc}"]

        if recomputed is None:
            return True, []  # operator not verified (unknown)

        ok = abs(recomputed - expected) < _TOLERANCE
        if not ok:
            return False, [
                f"arithmetic_verified: recomputed {recomputed} != result.answer {expected}"
            ]
        return True, []

    def _check_output_type(
        self, plan: OperatorPlan, result: ExecutionResult
    ) -> tuple[bool, list[str]]:
        try:
            spec = OPERATOR_REGISTRY.get(plan.operator)
        except KeyError:
            return False, [f"output_type_valid: unknown operator {plan.operator}"]
        if spec.output_type == "numeric" and not isinstance(result.answer, (int, float)):
            output_slot = str(_get_slot_value(plan, "output_slot") or "").lower()
            if plan.operator in {"ARGMAX", "ARGMIN"} and output_slot in {"entity", "company", "company_name"}:
                return True, []
            return False, [
                f"output_type_valid: answer for {plan.operator} must be numeric, got {type(result.answer).__name__}"
            ]
        if isinstance(result.answer, float) and (result.answer != result.answer):
            return False, ["output_type_valid: answer is NaN"]
        return True, []

    def _check_evidence_trace(self, result: ExecutionResult) -> tuple[bool, list[str]]:
        tokens = result.selected_tokens
        if not tokens:
            return True, []
        missing = [t.token_id for t in tokens if t.source is None]
        if missing:
            return False, [
                f"evidence_trace_complete: {len(missing)} token(s) missing source"
            ]
        return True, []

    def _recompute(
        self,
        op: str,
        plan: OperatorPlan,
        result: ExecutionResult,
    ) -> float | None:
        tokens = result.selected_tokens
        values = _values_by_index(result)

        if op == "SUM":
            return float(sum(_value_at(values, i, t) for i, t in enumerate(tokens)))

        if op == "YEAR_LIST":
            years = (result.metadata or {}).get("year_list")
            if isinstance(years, list) and years:
                return float(years[0])
            if not tokens or tokens[0].year is None:
                raise ValueError("YEAR_LIST needs at least one selected token with a year")
            return float(tokens[0].year)

        if op == "COUNT":
            count_target = _get_slot_value(plan, "count_target") or "company"
            if count_target in ("company", "entity"):
                return float(len({t.entity_id for t in tokens}))
            return float(len(tokens))

        if op == "AVG":
            if not tokens:
                raise ValueError("no tokens for AVG recomputation")
            return sum(_value_at(values, i, t) for i, t in enumerate(tokens)) / len(tokens)

        if op == "PRODUCT":
            if len(tokens) < 2:
                raise ValueError("PRODUCT needs >= 2 factor tokens")
            answer = 1.0
            for i, token in enumerate(tokens):
                answer *= _value_at(values, i, token)
            return float(answer)

        if op in ("MAX", "MIN", "ARGMAX", "ARGMIN"):
            if not tokens:
                raise ValueError(f"no tokens for {op} recomputation")
            output_slot = str(_get_slot_value(plan, "output_slot") or "").lower()
            if op in {"ARGMAX", "ARGMIN"} and output_slot in {"year", "time"} and tokens[0].year is not None:
                return float(tokens[0].year)
            return float(_value_at(values, 0, tokens[0]))

        if op == "RATIO":
            num_field = _get_slot_value(plan, "numerator_field")
            den_field = _get_slot_value(plan, "denominator_field")
            normalization = (result.metadata or {}).get("output_normalization")
            if isinstance(normalization, dict) and normalization.get("method") == "bound_display_value_ratio":
                num_ids = set((result.metadata or {}).get("numerator_token_ids") or [])
                den_ids = set((result.metadata or {}).get("denominator_token_ids") or [])
                numerator = sum(float(token.value) for token in tokens if token.token_id in num_ids)
                denominator = sum(float(token.value) for token in tokens if token.token_id in den_ids)
                if abs(denominator) < 1e-12:
                    raise ZeroDivisionError("denominator display value is zero")
                return numerator / denominator
            scalar_num = (result.metadata or {}).get("numerator_value")
            scalar_den = (result.metadata or {}).get("denominator_value")
            if scalar_num is not None and scalar_den is not None:
                if abs(float(scalar_den)) < 1e-12:
                    raise ZeroDivisionError("denominator value is zero")
                return float(scalar_num) / float(scalar_den)
            num_ids = set((result.metadata or {}).get("numerator_token_ids") or [])
            den_ids = set((result.metadata or {}).get("denominator_token_ids") or [])
            if num_ids or den_ids:
                num_pairs = [(i, t) for i, t in enumerate(tokens) if t.token_id in num_ids]
                den_pairs = [(i, t) for i, t in enumerate(tokens) if t.token_id in den_ids]
            else:
                num_pairs = [(i, t) for i, t in enumerate(tokens) if t.field_name == num_field]
                den_pairs = [(i, t) for i, t in enumerate(tokens) if t.field_name == den_field]
            num_tokens = [t for _, t in num_pairs]
            den_tokens = [t for _, t in den_pairs]
            if not num_tokens or not den_tokens:
                raise ValueError(
                    f"cannot separate numerator/denominator tokens "
                    f"(num_field={num_field}, den_field={den_field})"
                )
            den_val = sum(_value_at(values, i, t) for i, t in den_pairs)
            if abs(den_val) < 1e-12:
                raise ZeroDivisionError("denominator sums to zero")
            return sum(_value_at(values, i, t) for i, t in num_pairs) / den_val

        if op == "GROWTH":
            from_time = _get_slot_value(plan, "from_time")
            to_time = _get_slot_value(plan, "to_time")
            from_year = int(from_time) if from_time is not None else None
            to_year = int(to_time) if to_time is not None else None
            from_pairs = [(i, t) for i, t in enumerate(tokens) if t.year == from_year]
            to_pairs = [(i, t) for i, t in enumerate(tokens) if t.year == to_year]
            from_tokens = [t for _, t in from_pairs]
            to_tokens = [t for _, t in to_pairs]
            if not from_tokens or not to_tokens:
                raise ValueError(
                    f"cannot separate from/to tokens "
                    f"(from_year={from_year}, to_year={to_year})"
                )
            v_from = sum(_value_at(values, i, t) for i, t in from_pairs)
            if abs(v_from) < 1e-12:
                raise ZeroDivisionError("from-value is zero for GROWTH")
            v_to = sum(_value_at(values, i, t) for i, t in to_pairs)
            growth = (
                v_to - v_from
            ) / v_from
            return float(growth)

        if op == "DIFFERENCE":
            difference_mode = str((result.metadata or {}).get("difference_mode") or "signed")

            def finalize_difference(left: float, right: float) -> float:
                delta = left - right
                if plan.operator == "COMPARE" and difference_mode == "absolute_gap":
                    return float(abs(delta))
                return float(delta)

            left_field = _get_slot_value(plan, "left_field")
            right_field = _get_slot_value(plan, "right_field")
            if left_field is not None and right_field is not None:
                left_vals = [
                    _value_at(values, i, t)
                    for i, t in enumerate(tokens)
                    if t.field_name == left_field
                ]
                right_vals = [
                    _value_at(values, i, t)
                    for i, t in enumerate(tokens)
                    if t.field_name == right_field
                ]
                if left_vals or right_vals:
                    return finalize_difference(sum(left_vals), sum(right_vals))
            left_entity = _get_slot_value(plan, "left_entity")
            right_entity = _get_slot_value(plan, "right_entity")
            left_time = _get_slot_value(plan, "left_time")
            right_time = _get_slot_value(plan, "right_time")
            if left_time is not None and right_time is not None:
                left_year = int(left_time)
                right_year = int(right_time)
                left_vals = [
                    _value_at(values, i, t)
                    for i, t in enumerate(tokens)
                    if t.year == left_year
                ]
                right_vals = [
                    _value_at(values, i, t)
                    for i, t in enumerate(tokens)
                    if t.year == right_year
                ]
                return finalize_difference(sum(left_vals), sum(right_vals))
            if left_entity and right_entity:
                left_vals = [
                    _value_at(values, i, t)
                    for i, t in enumerate(tokens)
                    if t.entity_id == left_entity or t.company_name == left_entity
                ]
                right_vals = [
                    _value_at(values, i, t)
                    for i, t in enumerate(tokens)
                    if t.entity_id == right_entity or t.company_name == right_entity
                ]
                return finalize_difference(sum(left_vals), sum(right_vals))
            if len(tokens) < 2:
                raise ValueError("DIFFERENCE needs >= 2 tokens without entity binding")
            return finalize_difference(
                _value_at(values, 0, tokens[0]),
                _value_at(values, 1, tokens[1]),
            )

        if op == "LOOKUP":
            return float(sum(_value_at(values, i, t) for i, t in enumerate(tokens)))

        if op == "TOP_K":
            k = int(_get_slot_value(plan, "k") or 3)
            order = str(_get_slot_value(plan, "order") or "descending")
            reverse = order != "ascending"
            sorted_vals = sorted(
                [_value_at(values, i, t) for i, t in enumerate(tokens)],
                reverse=reverse,
            )
            if not sorted_vals:
                raise ValueError("no tokens for TOP_K recomputation")
            return float(sorted_vals[min(k, len(sorted_vals)) - 1])

        if op == "TREND":
            from_time = _get_slot_value(plan, "from_time")
            to_time = _get_slot_value(plan, "to_time")
            years = sorted({t.year for t in tokens if t.year is not None})
            if from_time is not None and to_time is not None:
                years = [y for y in years if int(from_time) <= y <= int(to_time)]
            if len(years) < 2:
                raise ValueError("TREND needs ≥2 years in tokens")
            series = [
                sum(_value_at(values, i, t) for i, t in enumerate(tokens) if t.year == yr)
                for yr in years
            ]
            if _get_slot_value(plan, "trend_metric") == "annual_delta":
                return float((series[-1] - series[0]) / max(len(series) - 1, 1))
            diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
            if all(d > 0 for d in diffs):
                return 1.0
            if all(d < 0 for d in diffs):
                return -1.0
            return 0.0

        return None  # unknown operator — skip


# ---- helper ----

def _get_slot_value(plan: OperatorPlan, key: str) -> object | None:
    from graph_numeric.operators.operator_plan import Slot
    val = (plan.slots or {}).get(key)
    if isinstance(val, Slot):
        return val.grounded_value
    return val


def _answer_is_nonzero(answer: object) -> bool:
    if isinstance(answer, (int, float)):
        return abs(float(answer)) > _TOLERANCE
    return answer not in (None, "")


def _expected_years(plan: OperatorPlan) -> set[int]:
    op = plan.operator
    years: set[int] = set()
    if op in ("GROWTH", "PERCENT_CHANGE"):
        for key in ("from_time", "to_time"):
            value = _get_slot_value(plan, key)
            if value is not None:
                years.add(int(value))
        return years
    if op in ("TREND", "PREDICT", "FORECAST"):
        from_time = _get_slot_value(plan, "from_time")
        to_time = _get_slot_value(plan, "to_time")
        history_range = _get_slot_value(plan, "history_range")
        if from_time is not None and to_time is not None:
            return set(range(int(from_time), int(to_time) + 1))
        if history_range is not None and "-" in str(history_range):
            start, end = str(history_range).split("-", 1)
            return set(range(int(start), int(end) + 1))
    value = _get_slot_value(plan, "year")
    return {int(value)} if value is not None else set()


def _units_for_check(plan: OperatorPlan, result: ExecutionResult) -> list[UnitInfo]:
    units = [
        UnitResolver().detect(token.unit)
        for token in result.selected_tokens
        if token.unit
    ]
    if units:
        return units
    plan_unit = _get_slot_value(plan, "unit")
    if plan_unit is not None:
        return [UnitResolver().detect(str(plan_unit))]
    return []


def _values_by_index(result: ExecutionResult) -> list[float | None]:
    rows = (result.metadata or {}).get("unit_normalization")
    if not isinstance(rows, list):
        return []
    values: list[float | None] = []
    for row in rows:
        if not isinstance(row, dict):
            values.append(None)
            continue
        normalized = row.get("normalized_value")
        values.append(float(normalized) if normalized is not None else None)
    return values


def _all_percentage_units(result: ExecutionResult) -> bool:
    rows = (result.metadata or {}).get("unit_normalization")
    if not isinstance(rows, list) or not rows:
        return False
    return all(isinstance(row, dict) and row.get("unit_category") == "percentage" for row in rows)


def _normalized_values_by_token(
    tokens: tuple[object, ...],
    requested_unit: object | None,
) -> dict[str, float]:
    if requested_unit is None:
        return {}
    resolver = UnitResolver()
    target = resolver.detect(str(requested_unit))
    values: dict[str, float] = {}
    for token in tokens:
        unit = getattr(token, "unit", None)
        if not unit:
            continue
        detected = resolver.detect(str(unit))
        try:
            values[str(getattr(token, "token_id"))] = resolver.normalize(
                float(getattr(token, "value")),
                detected,
                target,
            )
        except Exception:
            continue
    return values


def _tokens_for_dimension_slot(
    plan: OperatorPlan,
    result: ExecutionResult,
    slot_key: str,
) -> tuple[object, ...]:
    if slot_key == "numerator_dimension_filters":
        ids = set((result.metadata or {}).get("numerator_token_ids") or [])
        if ids:
            return tuple(token for token in result.selected_tokens if token.token_id in ids)
        field = _get_slot_value(plan, "numerator_field")
        return tuple(token for token in result.selected_tokens if token.field_name == field)
    if slot_key == "denominator_dimension_filters":
        ids = set((result.metadata or {}).get("denominator_token_ids") or [])
        if ids:
            return tuple(token for token in result.selected_tokens if token.token_id in ids)
        field = _get_slot_value(plan, "denominator_field")
        return tuple(token for token in result.selected_tokens if token.field_name == field)
    return result.selected_tokens


def _token_matches_dimensions(token: object, dimensions: object) -> bool:
    if not isinstance(dimensions, Mapping):
        return True
    token_dimensions = getattr(token, "dimensions", None) or {}
    for key, expected in dimensions.items():
        if str(key) not in token_dimensions:
            continue
        if not _dimension_value_matches(token_dimensions.get(str(key)), expected):
            return False
    return True


def _dimension_value_matches(actual: object | None, expected: object | None) -> bool:
    if actual is None or expected is None:
        return False
    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    actual_terms = _entity_terms(actual_text)
    expected_terms = _entity_terms(expected_text)
    if not actual_terms or not expected_terms:
        return False
    return expected_terms.issubset(actual_terms) or actual_terms.issubset(expected_terms)


def _value_at(values: list[float | None], index: int, token: object) -> float:
    if index < len(values) and values[index] is not None:
        return float(values[index])
    return float(getattr(token, "value"))


def _satisfies_threshold(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    return True


def _proposal_bound_token_ids(plan: OperatorPlan) -> set[str]:
    bindings = _get_slot_value(plan, "proposal_token_bindings")
    if not isinstance(bindings, Mapping):
        return set()
    return {
        str(token_id)
        for token_ids in bindings.values()
        if isinstance(token_ids, (list, tuple, set, frozenset))
        for token_id in token_ids
    }


def _matches_entity(token: object, entity: object) -> bool:
    entity_str = str(entity)
    company = str(getattr(token, "company_name", ""))
    return (
        getattr(token, "entity_id", None) == entity_str
        or company == entity_str
        or _entity_text_matches(company, entity_str)
        or str(getattr(token, "entity_id", "")).startswith(f"{entity_str}:")
    )


def _entity_text_matches(company: str, entity: str) -> bool:
    company_terms = _entity_terms(company)
    entity_terms = _entity_terms(entity)
    if not company_terms or not entity_terms:
        return False
    if len(entity_terms) == 1:
        return next(iter(entity_terms)) in company_terms
    return entity_terms.issubset(company_terms)


def _entity_terms(value: str) -> set[str]:
    import re

    suffixes = {
        "and",
        "the",
        "for",
        "from",
        "with",
        "year",
        "years",
        "total",
        "net",
        "sales",
        "revenue",
        "income",
        "operating",
        "gross",
        "research",
        "development",
        "cash",
        "equivalents",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "company",
        "co",
        "ltd",
        "limited",
        "plc",
        "com",
        "class",
        "common",
        "stock",
    }
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if len(term) >= 3 and term not in suffixes
    }
