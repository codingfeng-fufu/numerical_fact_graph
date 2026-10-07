from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource, normalize_identifier
from graph_numeric.learning.forecasting import forecast_profile, forecast_series
from graph_numeric.operators.operator_registry import OPERATOR_REGISTRY
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot
from graph_numeric.core.unit_resolver import UnitIncompatibleError, UnitInfo, UnitResolver
from graph_numeric.runtime.output_normalization import normalize_execution_output


@dataclass(frozen=True)
class ExecutionResult:
    answer: float | str
    selected_tokens: tuple[AttributeValueToken, ...]
    calculation: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def execute(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    """Dispatch execution to the correct operator handler, then verify."""
    if not plan.is_valid():
        missing = plan.missing_required_slots()
        raise ValueError(
            f"Plan for operator '{plan.operator}' is missing required slots: {missing}"
        )

    executor_operator = OPERATOR_REGISTRY.canonical_executor_operator(plan.operator)

    if executor_operator == "SUM":
        result = execute_sum(graph, plan)
    elif executor_operator == "COUNT":
        result = execute_count(graph, plan)
    elif executor_operator == "AVG":
        result = execute_avg(graph, plan)
    elif executor_operator == "PRODUCT":
        result = execute_product(graph, plan)
    elif executor_operator == "MAX":
        result = execute_max(graph, plan)
    elif executor_operator == "MIN":
        result = execute_min(graph, plan)
    elif executor_operator == "ARGMAX":
        result = execute_argmax(graph, plan)
    elif executor_operator == "ARGMIN":
        result = execute_argmin(graph, plan)
    elif executor_operator == "RATIO":
        result = execute_ratio(graph, plan)
    elif executor_operator == "GROWTH":
        result = execute_growth(graph, plan)
    elif executor_operator == "DIFFERENCE":
        result = execute_difference(graph, plan)
    elif executor_operator == "LOOKUP":
        result = execute_lookup(graph, plan)
    elif executor_operator == "YEAR_LIST":
        result = execute_year_list(graph, plan)
    elif executor_operator == "TOP_K":
        result = execute_top_k(graph, plan)
    elif executor_operator == "TREND":
        result = execute_trend(graph, plan)
    elif executor_operator == "PREDICT":
        result = execute_predict(graph, plan)
    else:
        raise ValueError(f"Unsupported operator: {plan.operator}")

    result = dataclasses.replace(
        result,
        metadata={**dict(result.metadata or {}), "executor_operator": executor_operator},
    )
    result = normalize_execution_output(_attach_evidence_metadata(result), plan)

    from graph_numeric.operators.verifier import Verifier
    report = Verifier().verify(graph, plan, result)
    return dataclasses.replace(result, checks=report.checks)


def execute_composite(
    graph: AttributeValueGraph,
    plan: CompositeOperatorPlan,
) -> ExecutionResult:
    """Execute a multi-step plan with explicit dependency bindings."""
    if not plan.is_valid():
        raise ValueError("Composite plan is invalid or references unavailable steps")

    results_by_step: dict[str, ExecutionResult] = {}
    step_rows: list[dict[str, Any]] = []
    binding_rows: dict[str, dict[str, Any]] = {}
    evidence_tokens: list[AttributeValueToken] = []

    for index, step in enumerate(plan.steps):
        step_id = _step_id(step, index)
        resolved_step, bindings = _bind_step_dependencies(step, results_by_step)
        result = execute(graph, resolved_step)
        results_by_step[step_id] = result
        evidence_tokens.extend(result.selected_tokens)
        binding_rows[step_id] = bindings
        step_rows.append(
            {
                "step_id": step_id,
                "operator": resolved_step.operator,
                "answer": result.answer,
                "calculation": result.calculation,
                "checks": dict(result.checks),
                "bindings": bindings,
                "selected_token_ids": [token.token_id for token in result.selected_tokens],
                "records_used": list(result.metadata.get("records_used", [])),
            }
        )

    if not step_rows:
        raise ValueError("Composite plan requires at least one step")

    final_step_id = step_rows[-1]["step_id"]
    final_result = results_by_step[str(final_step_id)]
    selected_tokens = _dedupe_tokens(tuple(evidence_tokens))
    calculation = "; ".join(
        f"{row['step_id']}({row['operator']})={row['answer']}"
        for row in step_rows
    )
    checks = {
        "composite_plan_valid": True,
        "dependencies_bound": all(
            len(row["bindings"]) == len((plan.steps[i].depends_on or {}))
            for i, row in enumerate(step_rows)
        ),
        "steps_verified": all(
            all(
                row_check
                for check_name, row_check in row["checks"].items()
                if check_name != "evidence_trace_complete"
            )
            for row in step_rows
        ),
    }
    metadata = {
        "executor_operator": "COMPOSITE",
        "steps": step_rows,
        "dependency_bindings": binding_rows,
        "final_step_id": final_step_id,
        "output_unit": final_result.metadata.get("output_unit"),
    }
    return _attach_evidence_metadata(
        ExecutionResult(
            answer=float(final_result.answer),
            selected_tokens=selected_tokens,
            calculation=calculation,
            checks=checks,
            metadata=metadata,
        )
    )


def execute_sum(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    target_fields = _resolve_optional_slot(plan, "target_fields")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    if isinstance(target_fields, (list, tuple, set, frozenset)) and target_fields:
        selected_rows: list[AttributeValueToken] = []
        seen_token_ids: set[str] = set()
        for field_name in target_fields:
            for token in graph.select(field_name=str(field_name), year=year, industry=industry):
                if token.token_id in seen_token_ids:
                    continue
                seen_token_ids.add(token.token_id)
                selected_rows.append(token)
        selected = tuple(selected_rows)
    else:
        selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)

    selected_token_ids = _resolve_optional_slot(plan, "selected_token_ids")
    if selected_token_ids:
        sel_ids = set(selected_token_ids)
        selected = tuple(t for t in selected if t.token_id in sel_ids)

    requested_unit = _resolve_optional_slot(plan, "unit")
    answer, normalized = _sum_normalized(selected, requested_unit)
    calc_values = [row["normalized_value"] for row in normalized]
    calc = " + ".join(str(v) for v in calc_values) + f" = {answer}" if selected else "0"
    metadata = _normalization_metadata(normalized, selected)
    table_enumeration_check = _table_sum_enumeration_check(
        selected,
        graph,
        requested_unit=requested_unit,
        selected_sum=answer,
    )
    if table_enumeration_check is not None:
        metadata["table_enumeration_check"] = table_enumeration_check

    return ExecutionResult(
        answer=float(answer),
        selected_tokens=selected,
        calculation=calc,
        checks={"field_grounded": target_field is not None, "records_found": len(selected) > 0},
        metadata=metadata,
    )


def execute_count(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    count_target = _resolve_slot(plan, "count_target")
    condition_field = _resolve_slot(plan, "condition_field")
    condition_op = _resolve_optional_slot(plan, "condition_op") or ">="
    condition_threshold = _resolve_optional_slot(plan, "condition_threshold")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=condition_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)

    if condition_threshold is not None:
        threshold_val = float(condition_threshold)
        condition_unit = (
            _resolve_optional_slot(plan, "condition_unit")
            or _resolve_optional_slot(plan, "unit")
        )
        normalized_rows = _normalized_token_rows(selected, condition_unit)
        values_by_token = {
            str(row["token_id"]): float(row["normalized_value"])
            for row in normalized_rows
        }
        if condition_op == ">=":
            selected = tuple(t for t in selected if values_by_token.get(t.token_id, t.value) >= threshold_val)
        elif condition_op == ">":
            selected = tuple(t for t in selected if values_by_token.get(t.token_id, t.value) > threshold_val)
        elif condition_op == "<=":
            selected = tuple(t for t in selected if values_by_token.get(t.token_id, t.value) <= threshold_val)
        elif condition_op == "<":
            selected = tuple(t for t in selected if values_by_token.get(t.token_id, t.value) < threshold_val)
        elif condition_op == "==":
            selected = tuple(t for t in selected if values_by_token.get(t.token_id, t.value) == threshold_val)

    if count_target in ("company", "entity"):
        unique_entities = set(t.entity_id for t in selected)
        answer = float(len(unique_entities))
    else:
        answer = float(len(selected))

    return ExecutionResult(
        answer=answer,
        selected_tokens=selected,
        calculation=f"count(distinct {count_target}) = {answer}",
        checks={"condition_applied": True, "records_found": len(selected) > 0},
    )


def execute_avg(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    total, normalized = _sum_normalized(selected, _resolve_optional_slot(plan, "unit"))
    answer = total / len(selected)
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=selected,
        calculation=f"avg({len(selected)} values) = {answer}",
        metadata=_normalization_metadata(normalized, selected),
    )


def execute_product(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    graph_factors = _product_factor_tokens(graph, plan)
    constant_factors = _product_question_constant_tokens(plan)
    selected = graph_factors + constant_factors
    if len(selected) < 2:
        raise ValueError("PRODUCT requires at least 2 factor tokens")

    normalized = _normalized_token_rows(selected, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    answer = 1.0
    calc_values: list[float] = []
    for index, token in enumerate(selected):
        value = _value_at_index(values, index, token)
        calc_values.append(value)
        answer *= value
    metadata = _normalization_metadata(normalized, selected)
    metadata["executor_operator"] = "PRODUCT"
    metadata["output_unit"] = _product_output_unit(normalized)
    metadata["product_unit_scale"] = _product_unit_scale(normalized)
    metadata["factor_token_ids"] = [token.token_id for token in selected]
    if constant_factors:
        metadata["question_constant_factors"] = _resolve_product_question_constants(plan)
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=selected,
        calculation=" * ".join(str(value) for value in calc_values) + f" = {answer}",
        metadata=metadata,
    )


def _product_factor_tokens(
    graph: AttributeValueGraph,
    plan: OperatorPlan,
) -> tuple[AttributeValueToken, ...]:
    factor_token_ids = _resolve_optional_slot(plan, "factor_token_ids")
    if factor_token_ids:
        return _tokens_by_ids(graph, factor_token_ids)

    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")
    selected_rows: list[AttributeValueToken] = []
    seen_token_ids: set[str] = set()
    factor_fields = _resolve_optional_slot(plan, "factor_fields")
    if isinstance(factor_fields, (list, tuple, set, frozenset)) and factor_fields:
        for field_name in factor_fields:
            for token in graph.select(field_name=str(field_name), year=year, industry=industry):
                if token.token_id in seen_token_ids:
                    continue
                seen_token_ids.add(token.token_id)
                selected_rows.append(token)
        return _apply_common_filters(tuple(selected_rows), plan)

    target_field = _resolve_optional_slot(plan, "target_field")
    if target_field is not None:
        selected = graph.select(field_name=str(target_field), year=year, industry=industry)
        return _apply_common_filters(selected, plan)

    if len(graph.tokens) <= 4:
        return _apply_common_filters(tuple(graph.tokens), plan)
    return ()


def _product_question_constant_tokens(plan: OperatorPlan) -> tuple[AttributeValueToken, ...]:
    constants = [
        row
        for row in _resolve_product_question_constants(plan)
        if row.get("role") == "multiplicative_factor"
    ]
    tokens: list[AttributeValueToken] = []
    for row in constants:
        surface = str(row.get("surface") or "question constant")
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        token_id = f"question_constant:{normalize_identifier(surface)}"
        tokens.append(
            AttributeValueToken(
                token_id=token_id,
                entity_id="question_text",
                company_name="question_text",
                field_name="question_constant",
                field_label=surface,
                value=value,
                source=TokenSource(
                    document_id="question_text",
                    text_excerpt=surface,
                ),
                dimensions={
                    "source": "question_text",
                    "constant_slot": "true",
                    "role": str(row.get("role") or "multiplicative_factor"),
                },
                raw_label=surface,
            )
        )
    return tuple(tokens)


def _resolve_product_question_constants(plan: OperatorPlan) -> list[dict[str, object]]:
    constants = _resolve_optional_slot(plan, "constant_slot")
    if constants is None:
        return []
    if isinstance(constants, Mapping):
        constants = [constants]
    if not isinstance(constants, (list, tuple)):
        return []
    rows: list[dict[str, object]] = []
    for row in constants:
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _product_output_unit(rows: list[dict[str, Any]]) -> str | None:
    categories = [str(row.get("unit_category") or "unknown") for row in rows]
    units = [str(row.get("normalized_unit") or row.get("raw_unit") or "") for row in rows]
    base_units = [unit for unit in units if unit]
    if "per_share" in categories and "shares" in categories:
        for unit in base_units:
            lowered = unit.lower()
            if "usd" in lowered or "$" in lowered or "dollar" in lowered:
                return "USD"
        return "money"
    if "money" in categories and all(category in {"money", "count", "unknown"} for category in categories):
        return next((unit for unit, category in zip(base_units, categories) if category == "money"), "money")
    if "shares" in categories and all(category in {"shares", "count", "unknown"} for category in categories):
        return "shares"
    if all(category in {"count", "unknown"} for category in categories):
        return "count"
    return "*".join(base_units) if base_units else None


def _product_unit_scale(rows: list[dict[str, Any]]) -> float:
    scale = 1.0
    for row in rows:
        try:
            scale *= float(row.get("unit_scale") or 1.0)
        except (TypeError, ValueError):
            continue
    return float(scale)


def execute_max(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    normalized = _normalized_token_rows(selected, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    max_index, max_token = max(
        enumerate(selected),
        key=lambda item: _value_at_index(values, item[0], item[1]),
    )
    answer = _value_at_index(values, max_index, max_token)
    selected_norm = [normalized[max_index]]
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=(max_token,),
        calculation=f"max = {answer}",
        metadata=_normalization_metadata(selected_norm, (max_token,)),
    )


def execute_min(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    normalized = _normalized_token_rows(selected, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    min_index, min_token = min(
        enumerate(selected),
        key=lambda item: _value_at_index(values, item[0], item[1]),
    )
    answer = _value_at_index(values, min_index, min_token)
    selected_norm = [normalized[min_index]]
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=(min_token,),
        calculation=f"min = {answer}",
        metadata=_normalization_metadata(selected_norm, (min_token,)),
    )


def execute_argmax(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    output_slot = str(_resolve_optional_slot(plan, "output_slot") or "").lower()
    if output_slot in {"entity", "company", "company_name"}:
        grouped = _group_arg_entity_candidates(selected)
        group_scores: list[tuple[float, int, str, tuple[AttributeValueToken, ...]]] = []
        for index, (label, group_tokens) in enumerate(grouped):
            values = [float(token.value) for token in group_tokens]
            if not values:
                continue
            score = abs(max(values) - min(values)) if len(values) >= 2 else max(values)
            row_index = min(
                token.source.row
                for token in group_tokens
                if token.source is not None and token.source.row is not None
            ) if any(token.source is not None and token.source.row is not None for token in group_tokens) else 10**6
            group_scores.append((score, row_index, label, group_tokens))
        if not group_scores:
            raise ValueError(f"No entity groups found for field '{target_field}'")
        group_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, _, label, group_tokens = group_scores[0]
        answer = label
        selected_tokens = _dedupe_tokens(group_tokens)
        selected_norm = _normalized_token_rows(selected_tokens, _resolve_optional_slot(plan, "unit"))
        evidence_text = [
            str(token.source.text_excerpt or token.raw_label or token.field_label or "")
            for token in selected_tokens
        ]
        return ExecutionResult(
            answer=answer,
            selected_tokens=selected_tokens,
            calculation=f"argmax entity = {answer}",
            metadata={
                "arg_value": float(max(float(token.value) for token in selected_tokens)),
                "arg_entity": answer,
                "arg_year": selected_tokens[0].year if selected_tokens and selected_tokens[0].year is not None else None,
                "evidence_text": evidence_text,
                **_normalization_metadata(selected_norm, selected_tokens),
            },
        )

    normalized = _normalized_token_rows(selected, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    max_index, max_token = max(
        enumerate(selected),
        key=lambda item: _value_at_index(values, item[0], item[1]),
    )
    answer = _value_at_index(values, max_index, max_token)
    output_answer = _arg_output_answer(plan, max_token, answer)
    selected_norm = [normalized[max_index]]
    return ExecutionResult(
        answer=output_answer,
        selected_tokens=(max_token,),
        calculation=f"argmax: entity={max_token.entity_id}, value={answer}",
        metadata={
            "arg_value": float(answer),
            "arg_entity": max_token.entity_id,
            "arg_year": max_token.year,
            **_normalization_metadata(selected_norm, (max_token,)),
        },
    )


def _group_arg_entity_candidates(tokens: tuple[AttributeValueToken, ...]) -> list[tuple[str, tuple[AttributeValueToken, ...]]]:
    grouped: dict[str, list[AttributeValueToken]] = {}
    for token in tokens:
        label = _arg_entity_label(token)
        grouped.setdefault(label, []).append(token)
    return [
        (label, tuple(group))
        for label, group in grouped.items()
    ]


def _arg_entity_label(token: AttributeValueToken) -> str:
    segment_label = (token.dimensions or {}).get("segment_label") if token.dimensions else None
    if segment_label and not _looks_like_yearish_label(segment_label):
        return str(segment_label)
    row_label = (token.dimensions or {}).get("row_label") if token.dimensions else None
    if row_label and not _looks_like_yearish_label(row_label):
        return str(row_label)
    if token.company_name and not _looks_like_yearish_label(token.company_name):
        return token.company_name
    if token.raw_label and not _looks_like_yearish_label(token.raw_label):
        return token.raw_label
    return token.entity_id


def execute_argmin(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    normalized = _normalized_token_rows(selected, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    min_index, min_token = min(
        enumerate(selected),
        key=lambda item: _value_at_index(values, item[0], item[1]),
    )
    answer = _value_at_index(values, min_index, min_token)
    output_answer = _arg_output_answer(plan, min_token, answer)
    selected_norm = [normalized[min_index]]
    return ExecutionResult(
        answer=output_answer,
        selected_tokens=(min_token,),
        calculation=f"argmin: entity={min_token.entity_id}, value={answer}",
        metadata={
            "arg_value": float(answer),
            "arg_entity": min_token.entity_id,
            "arg_year": min_token.year,
            **_normalization_metadata(selected_norm, (min_token,)),
        },
    )


def execute_ratio(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    num_field = _resolve_slot(plan, "numerator_field")
    den_field = _resolve_slot(plan, "denominator_field")
    year = _resolve_optional_slot(plan, "year")
    entity = _resolve_optional_slot(plan, "entity")
    industry = _resolve_optional_slot(plan, "industry")
    numerator_token_ids = _resolve_optional_slot(plan, "numerator_token_ids")
    denominator_token_ids = _resolve_optional_slot(plan, "denominator_token_ids")
    numerator_value = _resolve_optional_slot(plan, "numerator_value")
    denominator_value = _resolve_optional_slot(plan, "denominator_value")

    if numerator_token_ids or denominator_token_ids or numerator_value is not None or denominator_value is not None:
        num_tokens = _tokens_by_ids(graph, numerator_token_ids)
        den_tokens = _tokens_by_ids(graph, denominator_token_ids)
        if numerator_value is None:
            num_val, num_norm = _sum_normalized(num_tokens, _resolve_optional_slot(plan, "unit"))
        else:
            num_val, num_norm = float(numerator_value), []
        if denominator_value is None:
            den_val, den_norm = _sum_normalized(
                den_tokens,
                _normalization_target_unit(num_norm) or _resolve_optional_slot(plan, "unit"),
            )
        else:
            den_val, den_norm = float(denominator_value), []
        if den_val == 0:
            raise ZeroDivisionError(f"Denominator '{den_field}' sums to zero")
        answer = num_val / den_val
        return ExecutionResult(
            answer=float(answer),
            selected_tokens=num_tokens + den_tokens,
            calculation=f"{num_val} / {den_val} = {answer}",
            metadata={
                **_normalization_metadata(num_norm + den_norm, num_tokens + den_tokens),
                "output_unit": "ratio_dimensionless",
                "executor_operator": "RATIO",
                "numerator_token_ids": [token.token_id for token in num_tokens],
                "denominator_token_ids": [token.token_id for token in den_tokens],
                "numerator_value": float(num_val),
                "denominator_value": float(den_val),
                "replay_operands": [float(num_val), float(den_val)],
            },
        )

    numerator_fields = _resolve_optional_slot(plan, "numerator_fields")
    num_tokens = _select_ratio_numerator_tokens(
        graph,
        num_field,
        numerator_fields,
        year=year,
        industry=industry,
    )
    den_tokens = graph.select(field_name=den_field, year=year, industry=industry)

    shared_dimensions = _resolve_optional_slot(plan, "dimension_filters")
    num_tokens = _filter_dimensions(num_tokens, shared_dimensions)
    den_tokens = _filter_dimensions(den_tokens, shared_dimensions)
    num_tokens = _filter_dimensions(
        num_tokens,
        _resolve_optional_slot(plan, "numerator_dimension_filters"),
    )
    den_tokens = _filter_dimensions(
        den_tokens,
        _resolve_optional_slot(plan, "denominator_dimension_filters"),
    )
    den_tokens = _apply_ratio_denominator_row_selector(num_tokens, den_tokens, plan)
    if entity:
        num_tokens = tuple(t for t in num_tokens if _matches_entity(t, entity))
        den_tokens = tuple(t for t in den_tokens if _matches_entity(t, entity))
    entities = _resolve_optional_slot(plan, "entities")
    num_tokens = _filter_entities(num_tokens, entities)
    den_tokens = _filter_entities(den_tokens, entities)

    num_val, num_norm = _sum_normalized(num_tokens, _resolve_optional_slot(plan, "unit"))
    den_val, den_norm = _sum_normalized(
        den_tokens,
        _normalization_target_unit(num_norm) or _resolve_optional_slot(plan, "unit"),
    )

    if den_val == 0:
        raise ZeroDivisionError(f"Denominator '{den_field}' sums to zero")

    answer = num_val / den_val
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=num_tokens + den_tokens,
        calculation=f"{num_val} / {den_val} = {answer}",
        metadata={
            **_normalization_metadata(num_norm + den_norm, num_tokens + den_tokens),
            "output_unit": "ratio_dimensionless",
            "executor_operator": "RATIO",
            "numerator_token_ids": [token.token_id for token in num_tokens],
            "denominator_token_ids": [token.token_id for token in den_tokens],
            "numerator_value": float(num_val),
            "denominator_value": float(den_val),
            "replay_operands": [float(num_val), float(den_val)],
        },
    )


def _select_ratio_numerator_tokens(
    graph: AttributeValueGraph,
    numerator_field: str,
    numerator_fields: object | None,
    *,
    year: object | None,
    industry: object | None,
) -> tuple[AttributeValueToken, ...]:
    if isinstance(numerator_fields, (list, tuple, set, frozenset)) and numerator_fields:
        tokens: list[AttributeValueToken] = []
        for field_name in numerator_fields:
            tokens.extend(
                graph.select(
                    field_name=str(field_name),
                    year=year,
                    industry=str(industry) if industry is not None else None,
                )
            )
        return tuple(tokens)
    return graph.select(field_name=numerator_field, year=year, industry=industry)


def _tokens_by_ids(
    graph: AttributeValueGraph,
    token_ids: object | None,
) -> tuple[AttributeValueToken, ...]:
    if token_ids is None:
        return ()
    if isinstance(token_ids, str):
        wanted_order = [token_ids]
    elif isinstance(token_ids, (list, tuple, set, frozenset)):
        wanted_order = [str(token_id) for token_id in token_ids]
    else:
        return ()
    selected: list[AttributeValueToken] = []
    used_indexes: set[int] = set()
    for wanted in wanted_order:
        for index, token in enumerate(graph.tokens):
            if index in used_indexes:
                continue
            if token.token_id != wanted:
                continue
            selected.append(token)
            used_indexes.add(index)
            break
    return tuple(selected)


def _apply_ratio_denominator_row_selector(
    num_tokens: tuple[AttributeValueToken, ...],
    den_tokens: tuple[AttributeValueToken, ...],
    plan: OperatorPlan,
) -> tuple[AttributeValueToken, ...]:
    selector = _resolve_optional_slot(plan, "denominator_row_selector")
    if selector != "next_total_after_numerator":
        return den_tokens
    if len(num_tokens) != 1 or len(den_tokens) <= 1:
        return den_tokens
    num_row = _token_source_row(num_tokens[0])
    if num_row is None:
        return den_tokens
    candidates = [
        token
        for token in den_tokens
        if _token_source_row(token) is not None and _token_source_row(token) > num_row
    ]
    if not candidates:
        return den_tokens
    selected = min(candidates, key=lambda token: _token_source_row(token) or 10**9)
    return (selected,)


def _token_source_row(token: AttributeValueToken) -> int | None:
    source = token.source
    return source.row if source is not None else None


def execute_growth(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    entity = _resolve_optional_slot(plan, "entity")
    industry = _resolve_optional_slot(plan, "industry")
    from_time = _resolve_slot(plan, "from_time")
    to_time = _resolve_slot(plan, "to_time")

    if from_time is None or to_time is None:
        raise ValueError("GROWTH requires from_time and to_time slots")

    try:
        from_year = int(from_time)
        to_year = int(to_time)
    except (TypeError, ValueError):
        raise ValueError(f"GROWTH time values must be integer years, got {from_time=}, {to_time=}")

    bound_from = _proposal_bound_tokens(graph, plan, "from_time")
    bound_to = _proposal_bound_tokens(graph, plan, "to_time")
    if bound_from and bound_to:
        from_tokens = bound_from
        to_tokens = bound_to
    else:
        from_tokens = graph.select(field_name=target_field, year=from_year, industry=industry)
        to_tokens = graph.select(field_name=target_field, year=to_year, industry=industry)
        from_tokens = _filter_dimensions(from_tokens, _resolve_optional_slot(plan, "dimension_filters"))
        to_tokens = _filter_dimensions(to_tokens, _resolve_optional_slot(plan, "dimension_filters"))
        if entity:
            from_tokens = tuple(t for t in from_tokens if _matches_entity(t, entity))
            to_tokens = tuple(t for t in to_tokens if _matches_entity(t, entity))
        entities = _resolve_optional_slot(plan, "entities")
        from_tokens = _filter_entities(from_tokens, entities)
        to_tokens = _filter_entities(to_tokens, entities)

    v_from, from_norm = _sum_normalized(from_tokens, _resolve_optional_slot(plan, "unit"))
    v_to, to_norm = _sum_normalized(
        to_tokens,
        _normalization_target_unit(from_norm) or _resolve_optional_slot(plan, "unit"),
    )

    if v_from == 0:
        raise ZeroDivisionError(f"From-value for '{target_field}' at {from_year} is zero")

    is_percent_change = plan.operator == "PERCENT_CHANGE"
    use_percentage_points = is_percent_change and _all_percentage_units(from_norm + to_norm)
    ratio_answer = (v_to - v_from) / v_from
    if use_percentage_points:
        answer = v_to - v_from
        output_unit = "percentage_point"
        calculation = f"{v_to} - {v_from} = {answer}"
    else:
        answer = ratio_answer * 100.0 if is_percent_change else ratio_answer
        output_unit = "percent" if is_percent_change else "ratio_dimensionless"
        calculation = f"({v_to} - {v_from}) / {v_from} = {ratio_answer}"
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=from_tokens + to_tokens,
        calculation=calculation,
        metadata={
            **_normalization_metadata(from_norm + to_norm, from_tokens + to_tokens),
            "output_unit": output_unit,
            "executor_operator": "GROWTH",
        },
    )


def execute_difference(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")
    left_field = _resolve_optional_slot(plan, "left_field")
    right_field = _resolve_optional_slot(plan, "right_field")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    entity = _resolve_optional_slot(plan, "entity")
    if entity:
        selected = tuple(t for t in selected if _matches_entity(t, entity))

    # Left and right can be entity pairs or time pairs
    left_entity = _resolve_optional_slot(plan, "left_entity")
    right_entity = _resolve_optional_slot(plan, "right_entity")
    left_time = _resolve_optional_slot(plan, "left_time")
    right_time = _resolve_optional_slot(plan, "right_time")
    left_dimension = _resolve_optional_slot(plan, "left_dimension_filter")
    right_dimension = _resolve_optional_slot(plan, "right_dimension_filter")

    bound_left = _proposal_bound_tokens(
        graph,
        plan,
        "left_field",
        "left_entity",
        "left_time",
    )
    bound_right = _proposal_bound_tokens(
        graph,
        plan,
        "right_field",
        "right_entity",
        "right_time",
    )
    if bound_left and bound_right:
        left_tokens = bound_left
        right_tokens = bound_right
        v_left, left_norm = _sum_normalized(left_tokens, _resolve_optional_slot(plan, "unit"))
        v_right, right_norm = _sum_normalized(
            right_tokens,
            _normalization_target_unit(left_norm) or _resolve_optional_slot(plan, "unit"),
        )
        selected_tokens = left_tokens + right_tokens
        normalized = left_norm + right_norm
    elif left_field is not None and right_field is not None:
        left_tokens = graph.select(field_name=str(left_field), year=year, industry=industry)
        right_tokens = graph.select(field_name=str(right_field), year=year, industry=industry)
        left_tokens = _apply_common_filters(left_tokens, plan)
        right_tokens = _apply_common_filters(right_tokens, plan)
        if entity:
            left_tokens = tuple(t for t in left_tokens if _matches_entity(t, entity))
            right_tokens = tuple(t for t in right_tokens if _matches_entity(t, entity))
        v_left, left_norm = _sum_normalized(left_tokens, _resolve_optional_slot(plan, "unit"))
        v_right, right_norm = _sum_normalized(
            right_tokens,
            _normalization_target_unit(left_norm) or _resolve_optional_slot(plan, "unit"),
        )
        selected_tokens = left_tokens + right_tokens
        normalized = left_norm + right_norm
    elif left_time is not None and right_time is not None:
        left_year = int(left_time)
        right_year = int(right_time)
        left_tokens = tuple(t for t in selected if t.year == left_year)
        right_tokens = tuple(t for t in selected if t.year == right_year)
        v_left, left_norm = _sum_normalized(left_tokens, _resolve_optional_slot(plan, "unit"))
        v_right, right_norm = _sum_normalized(
            right_tokens,
            _normalization_target_unit(left_norm) or _resolve_optional_slot(plan, "unit"),
        )
        selected_tokens = left_tokens + right_tokens
        normalized = left_norm + right_norm
    elif left_dimension and right_dimension:
        left_tokens = _filter_dimensions(selected, left_dimension)
        right_tokens = _filter_dimensions(selected, right_dimension)
        v_left, left_norm = _sum_normalized(left_tokens, _resolve_optional_slot(plan, "unit"))
        v_right, right_norm = _sum_normalized(
            right_tokens,
            _normalization_target_unit(left_norm) or _resolve_optional_slot(plan, "unit"),
        )
        selected_tokens = left_tokens + right_tokens
        normalized = left_norm + right_norm
    elif left_entity and right_entity:
        left_tokens = tuple(t for t in selected if _matches_entity(t, left_entity))
        right_tokens = tuple(t for t in selected if _matches_entity(t, right_entity))
        v_left, left_norm = _sum_normalized(left_tokens, _resolve_optional_slot(plan, "unit"))
        v_right, right_norm = _sum_normalized(
            right_tokens,
            _normalization_target_unit(left_norm) or _resolve_optional_slot(plan, "unit"),
        )
        selected_tokens = left_tokens + right_tokens
        normalized = left_norm + right_norm
    else:
        _, normalized = _sum_normalized(selected, _resolve_optional_slot(plan, "unit"))
        values = [row["normalized_value"] for row in normalized]
        if len(values) < 2:
            raise ValueError("DIFFERENCE requires at least 2 values or left/right entity bindings")
        v_left = values[0]
        v_right = values[1]
        selected_tokens = selected

    difference_mode = "signed"
    if plan.operator == "COMPARE" and (
        left_dimension is not None
        or right_dimension is not None
        or left_field is not None
        or right_field is not None
    ):
        answer = abs(v_left - v_right)
        difference_mode = "absolute_gap"
    else:
        answer = v_left - v_right
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=selected_tokens,
        calculation=f"{v_left} - {v_right} = {answer}",
        metadata={
            **_normalization_metadata(normalized, selected_tokens),
            "difference_mode": difference_mode,
        },
    )


def execute_lookup(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    entity = _resolve_slot(plan, "entity")
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")

    selected = _proposal_bound_tokens(graph, plan, "target_field")
    explicitly_bound = bool(selected)
    if not explicitly_bound:
        selected = graph.select(field_name=target_field, year=year, industry=industry)
        selected = _filter_dimensions(selected, _resolve_optional_slot(plan, "dimension_filters"))
        if entity:
            selected = tuple(t for t in selected if _matches_entity(t, entity))
        selected = _filter_entities(selected, _resolve_optional_slot(plan, "entities"))

    if not selected:
        raise ValueError(f"No record found for field='{target_field}', entity='{entity}'")

    answer, normalized = _sum_normalized(selected, _resolve_optional_slot(plan, "unit"))
    return ExecutionResult(
        answer=float(answer),
        selected_tokens=selected,
        calculation=str(answer),
        metadata=_normalization_metadata(normalized, selected),
    )


def execute_year_list(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    industry = _resolve_optional_slot(plan, "industry")

    selected = graph.select(field_name=target_field, industry=industry)
    selected = _filter_dimensions(selected, _resolve_optional_slot(plan, "dimension_filters"))
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    years = sorted({token.year for token in selected if token.year is not None}, reverse=True)
    if not years:
        raise ValueError(f"No year values found for field '{target_field}'")

    year_tokens: list[AttributeValueToken] = []
    for year in years:
        token = next(token for token in selected if token.year == year)
        year_tokens.append(token)

    answer = float(years[0])
    return ExecutionResult(
        answer=answer,
        selected_tokens=tuple(year_tokens),
        calculation="years = " + ", ".join(str(year) for year in years),
        metadata={
            **_normalization_metadata([], year_tokens),
            "output_unit": "year_list",
            "executor_operator": "YEAR_LIST",
            "year_list": years,
        },
    )


def execute_top_k(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    k = int(_resolve_slot(plan, "k"))
    year = _resolve_optional_slot(plan, "year")
    industry = _resolve_optional_slot(plan, "industry")
    order = str(_resolve_optional_slot(plan, "order") or "descending")

    selected = graph.select(field_name=target_field, year=year, industry=industry)
    selected = _apply_common_filters(selected, plan)
    if not selected:
        raise ValueError(f"No records found for field '{target_field}'")

    reverse = order != "ascending"
    normalized = _normalized_token_rows(selected, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    seen_entities: set[str] = set()
    top_pairs: list[tuple[int, AttributeValueToken]] = []
    for idx, t in sorted(
        enumerate(selected),
        key=lambda item: _value_at_index(values, item[0], item[1]),
        reverse=reverse,
    ):
        if t.entity_id not in seen_entities:
            seen_entities.add(t.entity_id)
            top_pairs.append((idx, t))
            if len(top_pairs) >= k:
                break

    top_tokens = [token for _, token in top_pairs]
    selected_norm = [normalized[idx] for idx, _ in top_pairs]
    # answer = value of k-th entity (the cutoff value)
    answer = (
        float(_value_at_index(values, top_pairs[-1][0], top_pairs[-1][1]))
        if top_pairs
        else 0.0
    )
    calc = ", ".join(
        f"{t.company_name}:{_value_at_index(values, idx, t)}"
        for idx, t in top_pairs
    )
    return ExecutionResult(
        answer=answer,
        selected_tokens=tuple(top_tokens),
        calculation=calc,
        metadata=_normalization_metadata(selected_norm, top_tokens),
    )


def execute_trend(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field = _resolve_slot(plan, "target_field")
    entity = _resolve_optional_slot(plan, "entity")
    industry = _resolve_optional_slot(plan, "industry")
    from_time = _resolve_optional_slot(plan, "from_time")
    to_time = _resolve_optional_slot(plan, "to_time")

    # Determine time range
    if from_time is not None and to_time is not None:
        years = list(range(int(from_time), int(to_time) + 1))
    else:
        years = sorted(graph.years)

    if not years:
        raise ValueError("TREND requires at least one year in the graph")

    # Gather tokens over the time range
    all_tokens: list[AttributeValueToken] = []
    for yr in years:
        tokens = graph.select(field_name=target_field, year=yr, industry=industry)
        tokens = _filter_dimensions(tokens, _resolve_optional_slot(plan, "dimension_filters"))
        if entity:
            tokens = tuple(t for t in tokens if _matches_entity(t, entity))
        tokens = _filter_entities(tokens, _resolve_optional_slot(plan, "entities"))
        all_tokens.extend(tokens)

    normalized = _normalized_token_rows(all_tokens, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)

    # Build year → aggregate value series
    series: list[float] = []
    for yr in years:
        yr_vals = [
            _value_at_index(values, idx, token)
            for idx, token in enumerate(all_tokens)
            if token.year == yr
        ]
        if yr_vals:
            series.append(sum(yr_vals))

    if len(series) < 2:
        raise ValueError(f"TREND needs ≥2 data points; got {len(series)}")

    direction = _analyze_trend(series)
    if _resolve_optional_slot(plan, "trend_metric") == "annual_delta":
        answer = (series[-1] - series[0]) / max(len(series) - 1, 1)
    else:
        answer = 1.0 if direction == "increasing" else (-1.0 if direction == "decreasing" else 0.0)
    calc = " → ".join(f"{yr}:{v:.1f}" for yr, v in zip(years, series)) + f" → {direction}"
    return ExecutionResult(
        answer=answer,
        selected_tokens=tuple(all_tokens),
        calculation=calc,
        metadata=_normalization_metadata(normalized, all_tokens),
    )


def _analyze_trend(values: list[float]) -> str:
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(d > 0 for d in diffs):
        return "increasing"
    if all(d < 0 for d in diffs):
        return "decreasing"
    return "fluctuating"


def execute_predict(graph: AttributeValueGraph, plan: OperatorPlan) -> ExecutionResult:
    target_field   = _resolve_slot(plan, "target_field")
    entity         = _resolve_slot(plan, "entity")
    history_range  = _resolve_slot(plan, "history_range")   # "YYYY-YYYY"
    horizon        = int(_resolve_slot(plan, "horizon"))
    industry       = _resolve_optional_slot(plan, "industry")
    method         = str(_resolve_optional_slot(plan, "method") or "selector")
    profile_name   = _resolve_optional_slot(plan, "forecast_profile")
    selector_config = _resolve_optional_slot(plan, "selector_config")
    if selector_config is None and profile_name is not None:
        selector_config = forecast_profile(str(profile_name))
    if selector_config is None and method.lower().replace("-", "_") in {"selector", "configured_auto"}:
        selector_config = forecast_profile("default")
    if selector_config is not None and not isinstance(selector_config, Mapping):
        raise ValueError("Optional slot 'selector_config' must be a mapping when provided")

    start_str, end_str = history_range.split("-")
    history_years = list(range(int(start_str), int(end_str) + 1))

    all_tokens: list[AttributeValueToken] = []
    tokens_by_year: dict[int, tuple[AttributeValueToken, ...]] = {}
    for yr in history_years:
        tokens = graph.select(field_name=target_field, year=yr, industry=industry)
        entity_tokens = tuple(
            t for t in tokens
            if _matches_entity(t, entity)
        )
        if entity_tokens:
            tokens_by_year[yr] = entity_tokens
            all_tokens.extend(entity_tokens)

    normalized = _normalized_token_rows(all_tokens, _resolve_optional_slot(plan, "unit"))
    values = _normalized_values_by_index(normalized)
    token_indexes_by_year: dict[int, list[int]] = {}
    for index, token in enumerate(all_tokens):
        if token.year is None:
            continue
        token_indexes_by_year.setdefault(token.year, []).append(index)
    year_values: dict[int, float] = {
        yr: sum(_value_at_index(values, index, all_tokens[index]) for index in indexes)
        for yr, indexes in token_indexes_by_year.items()
        if yr in tokens_by_year
    }

    if len(year_values) < 2:
        raise ValueError(
            f"PREDICT needs ≥2 historical data points; got {len(year_values)}"
        )

    forecast = forecast_series(
        year_values,
        horizon,
        method=method,
        selector_config=selector_config,
    )
    prediction = forecast.prediction

    history_str = ", ".join(f"{yr}:{year_values[yr]:.1f}" for yr in sorted(year_values))
    calc = (
        f"history=[{history_str}]; "
        f"method={forecast.method}, selected={forecast.selected_method}; "
        f"predict({horizon})={prediction:.4f}"
    )
    return ExecutionResult(
        answer=float(prediction),
        selected_tokens=tuple(all_tokens),
        calculation=calc,
        metadata={
            "output_unit": _common_unit(all_tokens),
            "executor_operator": "PREDICT",
            "method": forecast.method,
            "selected_method": forecast.selected_method,
            "selector_config": dict(selector_config) if selector_config is not None else None,
            "forecast_profile": (
                str(profile_name)
                if profile_name is not None
                else (
                    str(selector_config.get("name"))
                    if isinstance(selector_config, Mapping) and selector_config.get("name") is not None
                    else None
                )
            ),
            "forecast_diagnostics": forecast.diagnostics,
            "history_range": history_range,
            "horizon": horizon,
            **_normalization_metadata(normalized, all_tokens),
        },
    )


def _resolve_slot(plan: OperatorPlan, key: str) -> str:
    value = (plan.slots or {}).get(key)
    if isinstance(value, Slot):
        if value.grounded_value is None:
            raise ValueError(f"Required slot '{key}' has null grounded_value")
        return str(value.grounded_value)
    if value is not None:
        return str(value)
    raise ValueError(f"Required slot '{key}' is missing from plan")


def _resolve_optional_slot(plan: OperatorPlan, key: str) -> object | None:
    value = (plan.slots or {}).get(key)
    if isinstance(value, Slot):
        return value.grounded_value
    return value


def _arg_output_answer(plan: OperatorPlan, token: AttributeValueToken, value: float) -> float | str:
    output_slot = str(_resolve_optional_slot(plan, "output_slot") or "").lower()
    if output_slot in {"year", "time"} and token.year is not None:
        return float(token.year)
    if output_slot in {"entity", "company", "company_name"}:
        segment_label = (token.dimensions or {}).get("segment_label") if token.dimensions else None
        if segment_label and not _looks_like_yearish_label(segment_label):
            return str(segment_label)
        row_label = (token.dimensions or {}).get("row_label") if token.dimensions else None
        if row_label and not _looks_like_yearish_label(row_label):
            return str(row_label)
        if token.company_name and not _looks_like_yearish_label(token.company_name):
            return token.company_name
        if token.raw_label and not _looks_like_yearish_label(token.raw_label):
            return token.raw_label
        return token.company_name
    return float(value)


def _matches_entity(token: AttributeValueToken, entity: object) -> bool:
    entity_str = str(entity)
    company = str(token.company_name)
    return (
        token.entity_id == entity_str
        or company == entity_str
        or _entity_text_matches(company, entity_str)
        or token.entity_id.startswith(f"{entity_str}:")
    )


def _looks_like_yearish_label(value: object | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(re.fullmatch(r"(?:19|20)\d{2}", text))


def _entity_text_matches(company: str, entity: str) -> bool:
    company_terms = _entity_terms(company)
    entity_terms = _entity_terms(entity)
    if not company_terms or not entity_terms:
        return False
    if len(entity_terms) == 1:
        return next(iter(entity_terms)) in company_terms
    return entity_terms.issubset(company_terms)


def _entity_terms(value: str) -> set[str]:
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
        for term in __import__("re").findall(r"[a-zA-Z0-9]+", value.lower())
        if (len(term) >= 3 or term.isdigit()) and term not in suffixes
    }


def _common_unit(tokens: list[AttributeValueToken] | tuple[AttributeValueToken, ...]) -> str | None:
    units = {token.unit for token in tokens if token.unit}
    return next(iter(units)) if len(units) == 1 else None


def _proposal_bound_tokens(
    graph: AttributeValueGraph,
    plan: OperatorPlan,
    *roles: str,
) -> tuple[AttributeValueToken, ...]:
    bindings = _resolve_optional_slot(plan, "proposal_token_bindings")
    if not isinstance(bindings, Mapping):
        return ()
    token_ids: list[str] = []
    for role in roles:
        values = bindings.get(role)
        if isinstance(values, (list, tuple, set, frozenset)):
            token_ids.extend(str(value) for value in values)
    return _tokens_by_ids(graph, list(dict.fromkeys(token_ids)))


def _step_id(step: OperatorPlan, index: int) -> str:
    value = (step.trace or {}).get("step_id") if step.trace else None
    return str(value) if value else f"s{index + 1}"


def _bind_step_dependencies(
    step: OperatorPlan,
    results_by_step: Mapping[str, ExecutionResult],
) -> tuple[OperatorPlan, dict[str, Any]]:
    if not step.depends_on:
        return step, {}

    slots = dict(step.slots or {})
    bindings: dict[str, Any] = {}
    for slot_key, reference in step.depends_on.items():
        value = _resolve_dependency_reference(reference, results_by_step)
        slots[slot_key] = _binding_slot_value(str(slot_key), reference, value)
        bindings[str(slot_key)] = {
            "reference": reference,
            "value": value,
        }

    return dataclasses.replace(step, slots=slots), bindings


def _resolve_dependency_reference(
    reference: object,
    results_by_step: Mapping[str, ExecutionResult],
) -> object:
    if not isinstance(reference, str) or not reference.startswith("$"):
        return reference

    ref = reference[1:]
    step_id, _, attr = ref.partition(".")
    if not attr:
        attr = "answer"
    if step_id not in results_by_step:
        raise ValueError(f"Dependency reference '{reference}' points to an unavailable step")

    result = results_by_step[step_id]
    tokens = result.selected_tokens
    if attr in {"answer", "value"}:
        return result.answer
    if attr in {"entity", "company", "company_name"}:
        if not tokens:
            raise ValueError(f"Dependency reference '{reference}' has no selected token")
        return tokens[0].company_name
    if attr in {"entity_id"}:
        if not tokens:
            raise ValueError(f"Dependency reference '{reference}' has no selected token")
        return tokens[0].entity_id
    if attr in {"entities", "company_names"}:
        return _unique_values(token.company_name for token in tokens)
    if attr in {"entity_ids"}:
        return _unique_values(token.entity_id for token in tokens)
    if attr in {"token_ids", "selected_token_ids", "entity_token_ids"}:
        return [token.token_id for token in tokens]
    if attr in {"field", "field_name"}:
        if not tokens:
            raise ValueError(f"Dependency reference '{reference}' has no selected token")
        return tokens[0].field_name
    if attr in result.metadata:
        return result.metadata[attr]
    raise ValueError(f"Unsupported dependency reference attribute '{attr}' in '{reference}'")


def _binding_slot_value(slot_key: str, reference: object, value: object) -> object:
    if slot_key in {"selected_token_ids", "entities", "selector_config"}:
        return value
    return Slot(surface=str(reference), grounded_value=value, confidence=1.0)


def _unique_values(values: object) -> list[object]:
    unique: list[object] = []
    seen: set[object] = set()
    for value in values:  # type: ignore[union-attr]
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _dedupe_tokens(tokens: tuple[AttributeValueToken, ...]) -> tuple[AttributeValueToken, ...]:
    deduped: list[AttributeValueToken] = []
    seen: set[str] = set()
    for token in tokens:
        if token.token_id in seen:
            continue
        seen.add(token.token_id)
        deduped.append(token)
    return tuple(deduped)


def _filter_entities(
    tokens: tuple[AttributeValueToken, ...],
    entities: object | None,
) -> tuple[AttributeValueToken, ...]:
    if entities is None:
        return tokens
    if isinstance(entities, str):
        entity_values = [entities]
    elif isinstance(entities, (list, tuple, set, frozenset)):
        entity_values = list(entities)
    else:
        entity_values = [entities]
    if not entity_values:
        return ()
    return tuple(
        token
        for token in tokens
        if any(_matches_entity(token, entity) for entity in entity_values)
    )


def _apply_common_filters(
    tokens: tuple[AttributeValueToken, ...],
    plan: OperatorPlan,
) -> tuple[AttributeValueToken, ...]:
    selected = _filter_implicit_total_rows(tokens, plan)
    selected = _filter_dimensions(selected, _resolve_optional_slot(plan, "dimension_filters"))
    entity = _resolve_optional_slot(plan, "entity")
    if entity is not None:
        selected = tuple(t for t in selected if _matches_entity(t, entity))
    return _filter_entities(selected, _resolve_optional_slot(plan, "entities"))


def _filter_implicit_total_rows(
    tokens: tuple[AttributeValueToken, ...],
    plan: OperatorPlan,
) -> tuple[AttributeValueToken, ...]:
    if _plan_explicitly_filters_total_rows(plan):
        return tokens
    return tuple(
        token
        for token in tokens
        if not _token_is_aggregate(token)
    )


def _plan_explicitly_filters_total_rows(plan: OperatorPlan) -> bool:
    for slot_name in (
        "dimension_filters",
        "numerator_dimension_filters",
        "denominator_dimension_filters",
        "left_dimension_filter",
        "right_dimension_filter",
    ):
        dimensions = _resolve_optional_slot(plan, slot_name)
        if isinstance(dimensions, Mapping) and (
            _truthy(dimensions.get("is_total_row")) or _truthy(dimensions.get("is_aggregate"))
        ):
            return True
    return False


def _table_sum_enumeration_check(
    selected: tuple[AttributeValueToken, ...],
    graph: AttributeValueGraph,
    *,
    requested_unit: object | None,
    selected_sum: float,
) -> dict[str, Any] | None:
    if not selected:
        return None
    if any(not _token_is_table_token(token) or _token_is_aggregate(token) for token in selected):
        return None
    scopes = {_table_column_scope(token) for token in selected}
    if len(scopes) != 1:
        return None
    scope = next(iter(scopes))
    if scope is None:
        return None
    aggregate_candidates = tuple(
        token
        for token in graph.tokens
        if _table_column_scope(token) == scope
        and _token_is_aggregate(token)
        and token.field_name == selected[0].field_name
    )
    if not aggregate_candidates:
        return None
    normalized_aggregates = _normalized_token_rows(aggregate_candidates, requested_unit)
    aggregate_values = [float(row["normalized_value"]) for row in normalized_aggregates]
    unique_values = []
    for value in aggregate_values:
        if not any(math.isclose(value, existing, rel_tol=1e-9, abs_tol=1e-9) for existing in unique_values):
            unique_values.append(value)
    if len(unique_values) != 1:
        raise ValueError(
            "enumeration_inconsistent: multiple aggregate values in the same table column"
        )
    aggregate_value = unique_values[0]
    tolerance = max(1e-4, abs(aggregate_value) * 0.01)
    if abs(float(selected_sum) - aggregate_value) > tolerance:
        raise ValueError(
            f"enumeration_inconsistent: selected_sum={selected_sum} aggregate={aggregate_value}"
        )
    return {
        "status": "passed",
        "table_id": scope[0],
        "col_id": scope[1],
        "aggregate_token_id": aggregate_candidates[0].token_id,
        "aggregate_value": aggregate_value,
        "selected_sum": float(selected_sum),
        "selected_token_ids": [token.token_id for token in selected],
    }


def _table_column_scope(token: AttributeValueToken) -> tuple[object, object] | None:
    dimensions = token.dimensions or {}
    table_id = token.table_id or dimensions.get("table_id")
    if table_id is None and token.source is not None:
        table_id = token.source.table
    col_id = token.col_id
    if col_id is None:
        col_id = dimensions.get("col_id")
    if col_id is None and token.source is not None:
        col_id = token.source.column
    if table_id is None or col_id is None:
        return None
    return table_id, col_id


def _token_is_table_token(token: AttributeValueToken) -> bool:
    if token.provenance_channel == "table":
        return True
    dimensions = token.dimensions or {}
    if dimensions.get("provenance_channel") == "table":
        return True
    return token.table_id is not None or (token.source is not None and token.source.table is not None)


def _token_is_aggregate(token: AttributeValueToken) -> bool:
    if token.is_aggregate:
        return True
    dimensions = token.dimensions or {}
    return _truthy(dimensions.get("is_aggregate")) or _truthy(dimensions.get("is_total_row"))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _filter_dimensions(
    tokens: tuple[AttributeValueToken, ...],
    dimensions: object | None,
) -> tuple[AttributeValueToken, ...]:
    if not dimensions:
        return tokens
    if not isinstance(dimensions, Mapping):
        return tokens
    selected = tokens
    for key, value in dimensions.items():
        if not any(str(key) in (token.dimensions or {}) for token in selected):
            continue
        selected = tuple(
            token
            for token in selected
            if _dimension_value_matches((token.dimensions or {}).get(str(key)), value)
        )
    return selected


def _dimension_value_matches(actual: object | None, expected: object | None) -> bool:
    if actual is None or expected is None:
        return False
    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    actual_norm = _dimension_core_text(actual_text)
    expected_norm = _dimension_core_text(expected_text)
    if not actual_norm or not expected_norm:
        return False
    return expected_norm in actual_norm or actual_norm in expected_norm


def _dimension_core_text(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [
        token
        for token in text.split()
        if token and token not in {"m", "bn", "b", "k", "million", "millions", "billion", "billions", "thousand", "thousands"}
    ]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return " ".join(tokens)


def _attach_evidence_metadata(result: ExecutionResult) -> ExecutionResult:
    records = [_token_evidence(token) for token in result.selected_tokens]
    metadata = dict(result.metadata or {})
    metadata.setdefault("selected_token_ids", [token.token_id for token in result.selected_tokens])
    metadata.setdefault("records_used", records)
    metadata.setdefault(
        "evidence_trace_complete",
        all(token.source is not None for token in result.selected_tokens),
    )
    return dataclasses.replace(result, metadata=metadata)


def _token_evidence(token: AttributeValueToken) -> dict[str, Any]:
    source = token.source
    return {
        "token_id": token.token_id,
        "entity_id": token.entity_id,
        "company_name": token.company_name,
        "field_name": token.field_name,
        "field_label": token.field_label,
        "canonical_concept_id": token.canonical_concept_id,
        "dimensions": dict(token.dimensions or {}),
        "raw_label": token.raw_label,
        "external_concept_ids": list(token.external_concept_ids),
        "year": token.year,
        "industry": token.industry,
        "value": token.value,
        "unit": token.unit,
        "source": (
            {
                "document_id": source.document_id,
                "page": source.page,
                "table": source.table,
                "row": source.row,
                "column": source.column,
                "char_start": source.char_start,
                "char_end": source.char_end,
                "text_excerpt": source.text_excerpt,
            }
            if source is not None
            else None
        ),
    }


def _sum_normalized(
    tokens: tuple[AttributeValueToken, ...] | list[AttributeValueToken],
    requested_unit: object | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    normalized = _normalized_token_rows(tokens, requested_unit)
    return float(sum(row["normalized_value"] for row in normalized)), normalized


def _normalized_token_rows(
    tokens: tuple[AttributeValueToken, ...] | list[AttributeValueToken],
    requested_unit: object | None = None,
) -> list[dict[str, Any]]:
    token_list = list(tokens)
    if not token_list:
        return []

    resolver = UnitResolver()
    unit_infos = [
        resolver.detect(str(token.unit))
        if token.unit
        else UnitInfo("", "unknown", 1.0, "", source="missing")
        for token in token_list
    ]
    target = _choose_target_unit(unit_infos, requested_unit)
    rows: list[dict[str, Any]] = []
    for token, from_unit in zip(token_list, unit_infos):
        normalized_value = float(token.value)
        normalized_unit = token.unit
        try:
            if target is not None and _can_normalize(from_unit, target):
                normalized_value = resolver.normalize(float(token.value), from_unit, target)
                normalized_unit = target.unit_string
        except UnitIncompatibleError:
            normalized_value = float(token.value)
            normalized_unit = token.unit
        rows.append(
            {
                "token_id": token.token_id,
                "raw_value": float(token.value),
                "raw_unit": token.unit,
                "normalized_value": float(normalized_value),
                "normalized_unit": normalized_unit,
                "unit_category": from_unit.unit_category,
                "unit_scale": from_unit.unit_scale,
            }
        )
    return rows


def _normalized_values_by_index(rows: list[dict[str, Any]]) -> list[float | None]:
    return [
        float(row["normalized_value"]) if row.get("normalized_value") is not None else None
        for row in rows
    ]


def _value_at_index(
    values: list[float | None],
    index: int,
    token: AttributeValueToken,
) -> float:
    if index < len(values) and values[index] is not None:
        return float(values[index])
    return float(token.value)


def _choose_target_unit(
    unit_infos: list[UnitInfo],
    requested_unit: object | None,
) -> UnitInfo | None:
    resolver = UnitResolver()
    known_units = [unit for unit in unit_infos if unit.unit_category != "unknown"]
    if requested_unit is not None:
        requested = resolver.detect(str(requested_unit))
        if not known_units or any(_can_normalize(unit, requested) for unit in known_units):
            return requested
    return known_units[0] if known_units else None


def _can_normalize(from_unit: UnitInfo, to_unit: UnitInfo) -> bool:
    if (
        from_unit.unit_category == "unknown"
        or to_unit.unit_category == "unknown"
        or from_unit.unit_category != to_unit.unit_category
    ):
        return False
    if (
        from_unit.unit_category == "money"
        and from_unit.currency_code
        and to_unit.currency_code
        and from_unit.currency_code != to_unit.currency_code
    ):
        return False
    return True


def _normalization_target_unit(rows: list[dict[str, Any]]) -> str | None:
    units = [
        str(row["normalized_unit"])
        for row in rows
        if row.get("normalized_unit") is not None and row.get("unit_category") != "unknown"
    ]
    return units[0] if units else None


def _all_percentage_units(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(row.get("unit_category") == "percentage" for row in rows)


def _normalization_metadata(
    rows: list[dict[str, Any]],
    tokens: tuple[AttributeValueToken, ...] | list[AttributeValueToken],
) -> dict[str, Any]:
    common_normalized_unit = {
        row.get("normalized_unit")
        for row in rows
        if row.get("normalized_unit") is not None and row.get("unit_category") != "unknown"
    }
    return {
        "output_unit": (
            next(iter(common_normalized_unit))
            if len(common_normalized_unit) == 1
            else _common_unit(tuple(tokens))
        ),
        "unit_normalization": rows,
        "unit_normalized": any(
            abs(float(row["raw_value"]) - float(row["normalized_value"])) > 1e-12
            for row in rows
        ),
    }
