from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def compare_text_target_recall(
    targets: Sequence[Mapping[str, Any]],
    *,
    baseline_tokens_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    t8_tokens_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare complete operand recall for a fixed set of text-evidence targets."""
    rows: list[dict[str, Any]] = []
    for target in targets:
        sample_id = str(target["sample_id"])
        operands = [_as_float(value) for value in target.get("document_operands", [])]
        baseline_values = _token_values(baseline_tokens_by_sample.get(sample_id, []))
        t8_values = _token_values(t8_tokens_by_sample.get(sample_id, []))
        baseline_recalled = _all_operands_recalled(operands, baseline_values)
        t8_recalled = _all_operands_recalled(operands, t8_values)
        rows.append(
            {
                **dict(target),
                "baseline_recalled": baseline_recalled,
                "t8_recalled": t8_recalled,
                "baseline_missing_operands": _missing_operands(operands, baseline_values),
                "t8_missing_operands": _missing_operands(operands, t8_values),
                "status": _recall_status(baseline_recalled, t8_recalled),
            }
        )

    return {
        "summary": {
            "target_count": len(rows),
            "baseline_recalled_count": sum(row["baseline_recalled"] for row in rows),
            "t8_recalled_count": sum(row["t8_recalled"] for row in rows),
            "newly_recalled_count": sum(row["status"] == "newly_recalled" for row in rows),
            "regressed_count": sum(row["status"] == "regressed" for row in rows),
            "still_missing_count": sum(row["status"] == "still_missing" for row in rows),
        },
        "rows": rows,
    }


def _token_values(tokens: Iterable[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for token in tokens:
        try:
            values.append(_as_float(token.get("value")))
        except (TypeError, ValueError):
            continue
    return values


def _as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _all_operands_recalled(operands: Sequence[float], values: Sequence[float]) -> bool:
    return bool(operands) and not _missing_operands(operands, values)


def _missing_operands(operands: Sequence[float], values: Sequence[float]) -> list[float]:
    return [operand for operand in operands if not any(_numeric_equal(operand, value) for value in values)]


def _numeric_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)


def _recall_status(baseline_recalled: bool, t8_recalled: bool) -> str:
    if baseline_recalled and t8_recalled:
        return "retained_recalled"
    if not baseline_recalled and t8_recalled:
        return "newly_recalled"
    if baseline_recalled and not t8_recalled:
        return "regressed"
    return "still_missing"
