from __future__ import annotations

import dataclasses
import math
from typing import Any


def normalize_execution_output(
    result: Any,
    plan: Any,
    *,
    dataset_convention: str = "finqa",
) -> Any:
    """Apply dataset-level output conventions at the executor boundary."""
    if dataset_convention != "finqa":
        return result
    operator = str(getattr(plan, "operator", "") or "").upper()
    if operator == "RATIO":
        answer, replay_operands = _finqa_ratio_display_value(result, plan)
        if answer is None:
            return result
        metadata = dict(getattr(result, "metadata", {}) or {})
        metadata["replay_operands"] = list(replay_operands or [])
        metadata["output_normalization"] = {
            "dataset_convention": dataset_convention,
            "operator": operator,
            "raw_answer": getattr(result, "answer", None),
            "normalized_answer": float(answer),
            "method": "bound_display_value_ratio",
        }
        return dataclasses.replace(result, answer=float(answer), metadata=metadata)
    if operator != "PERCENT_CHANGE":
        return result
    answer, calculation = _percent_change_ratio(result, plan)
    if answer is None:
        answer = _legacy_percent_answer_to_ratio(result)
        calculation = getattr(result, "calculation", "")
    if answer is None:
        return result
    metadata = dict(getattr(result, "metadata", {}) or {})
    raw_answer = getattr(result, "answer", None)
    raw_output_unit = metadata.get("output_unit")
    metadata["output_unit"] = "ratio_dimensionless"
    metadata["output_convention"] = "finqa_decimal_ratio"
    metadata["output_normalization"] = {
        "dataset_convention": dataset_convention,
        "operator": operator,
        "raw_answer": raw_answer,
        "raw_output_unit": raw_output_unit,
        "normalized_answer": float(answer),
        "normalized_output_unit": "ratio_dimensionless",
    }
    return dataclasses.replace(
        result,
        answer=float(answer),
        calculation=calculation or getattr(result, "calculation", ""),
        metadata=metadata,
    )


def _legacy_percent_answer_to_ratio(result: Any) -> float | None:
    value = getattr(result, "answer", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)):
        return None
    output_unit = str((getattr(result, "metadata", {}) or {}).get("output_unit") or "")
    if output_unit == "percent":
        return float(value) / 100.0
    return None


def _finqa_ratio_display_value(
    result: Any,
    plan: Any,
) -> tuple[float | None, tuple[float, float] | None]:
    note = str((getattr(plan, "trace", {}) or {}).get("proposal_note") or "").lower()
    if "percent" not in note and "percentage" not in note:
        return None, None
    tokens = tuple(getattr(result, "selected_tokens", ()) or ())
    metadata = getattr(result, "metadata", {}) or {}
    numerator_ids = set(metadata.get("numerator_token_ids") or [])
    denominator_ids = set(metadata.get("denominator_token_ids") or [])
    numerator = sum(float(token.value) for token in tokens if token.token_id in numerator_ids)
    denominator = sum(float(token.value) for token in tokens if token.token_id in denominator_ids)
    if not numerator_ids or not denominator_ids or abs(denominator) < 1e-12:
        return None, None
    return numerator / denominator, (numerator, denominator)


def _percent_change_ratio(result: Any, plan: Any) -> tuple[float | None, str | None]:
    from_time = _optional_int_slot(plan, "from_time")
    to_time = _optional_int_slot(plan, "to_time")
    if from_time is None or to_time is None:
        return None, None
    tokens = tuple(getattr(result, "selected_tokens", ()) or ())
    rows = (getattr(result, "metadata", {}) or {}).get("unit_normalization")
    if not tokens or not isinstance(rows, list):
        return None, None
    values: list[float | None] = []
    for index, token in enumerate(tokens):
        row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
        value = row.get("normalized_value")
        try:
            values.append(float(value) if value is not None else float(token.value))
        except (TypeError, ValueError):
            values.append(None)
    from_values = [
        value
        for value, token in zip(values, tokens, strict=False)
        if value is not None and _token_year(token) == from_time
    ]
    to_values = [
        value
        for value, token in zip(values, tokens, strict=False)
        if value is not None and _token_year(token) == to_time
    ]
    if not from_values or not to_values:
        return None, None
    from_sum = float(sum(from_values))
    to_sum = float(sum(to_values))
    if abs(from_sum) < 1e-12:
        return None, None
    answer = (to_sum - from_sum) / from_sum
    return answer, f"({to_sum} - {from_sum}) / {from_sum} = {answer}"


def _optional_int_slot(plan: Any, key: str) -> int | None:
    slots = getattr(plan, "slots", {}) or {}
    value = slots.get(key)
    value = getattr(value, "grounded_value", value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_year(token: Any) -> int | None:
    value = getattr(token, "year", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
