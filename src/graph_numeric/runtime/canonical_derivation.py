from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


Number = int | float


@dataclass(frozen=True)
class CanonicalOperand:
    operand_id: str
    source_kind: Literal["token", "question_constant"]
    token_id: str | None
    raw_value: float | str
    normalized_value: float | str
    unit: str | None
    scale: float
    sign: int
    question_span: str | None = None


@dataclass(frozen=True)
class EvaluationNode:
    kind: Literal["operand", "constant", "operator", "normalize"]
    value: Any
    children: tuple["EvaluationNode", ...] = ()


@dataclass(frozen=True)
class CanonicalDerivation:
    schema_version: str
    operator: str
    operands: tuple[CanonicalOperand, ...]
    evaluation: EvaluationNode
    output: float | str
    provenance: Mapping[str, Any]


def replay(plan: CanonicalDerivation, *, fact_values: Mapping[str, Any]) -> float | str:
    if plan.schema_version != "canonical_derivation_v1":
        raise ValueError(f"unsupported canonical derivation schema: {plan.schema_version}")
    return _evaluate(plan.evaluation, plan=plan, fact_values=fact_values)


def to_dict(plan: CanonicalDerivation) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "operator": plan.operator,
        "operands": [dataclasses.asdict(operand) for operand in plan.operands],
        "evaluation": _node_to_dict(plan.evaluation),
        "output": plan.output,
        "provenance": dict(plan.provenance),
    }


def from_dict(payload: Mapping[str, Any]) -> CanonicalDerivation:
    return CanonicalDerivation(
        schema_version=str(payload["schema_version"]),
        operator=str(payload["operator"]),
        operands=tuple(CanonicalOperand(**operand) for operand in payload["operands"]),
        evaluation=_node_from_dict(payload["evaluation"]),
        output=payload["output"],
        provenance=dict(payload["provenance"]),
    )


def _node_to_dict(node: EvaluationNode) -> dict[str, Any]:
    return {
        "kind": node.kind,
        "value": node.value,
        "children": [_node_to_dict(child) for child in node.children],
    }


def _node_from_dict(payload: Mapping[str, Any]) -> EvaluationNode:
    return EvaluationNode(
        str(payload["kind"]),
        payload["value"],
        tuple(_node_from_dict(child) for child in payload.get("children", ())),
    )


def _evaluate(
    node: EvaluationNode,
    *,
    plan: CanonicalDerivation,
    fact_values: Mapping[str, Any],
) -> float | str:
    if node.kind == "operand":
        return _operand_value(plan.operands[int(node.value)], fact_values=fact_values)
    if node.kind == "constant":
        return node.value
    if node.kind == "operator":
        return _evaluate_operator(
            str(node.value),
            [_evaluate(child, plan=plan, fact_values=fact_values) for child in node.children],
        )
    if node.kind == "normalize":
        return _evaluate_normalize(
            str(node.value),
            [_evaluate(child, plan=plan, fact_values=fact_values) for child in node.children],
        )
    raise ValueError(f"unsupported evaluation node kind: {node.kind}")


def _operand_value(operand: CanonicalOperand, *, fact_values: Mapping[str, Any]) -> float | str:
    if operand.source_kind == "question_constant":
        return operand.normalized_value
    if operand.source_kind != "token":
        raise ValueError(f"unsupported canonical operand source: {operand.source_kind}")
    if operand.token_id is None or operand.token_id not in fact_values:
        raise KeyError(f"missing canonical operand token: {operand.token_id}")
    value = fact_values[operand.token_id]
    if isinstance(value, (int, float)) and isinstance(operand.scale, (int, float)):
        return float(value) * float(operand.scale) * int(operand.sign)
    return value


def _evaluate_operator(operator: str, values: list[float | str]) -> float | str:
    op = operator.upper()
    if op == "LOOKUP":
        _require_arity(op, values, 1)
        return values[0]
    if op == "SUM":
        return sum(_as_number(value) for value in values)
    if op == "COUNT":
        return float(len(values))
    if op == "AVG":
        if not values:
            raise ValueError("AVG requires at least one operand")
        return sum(_as_number(value) for value in values) / len(values)
    if op == "PRODUCT":
        product = 1.0
        for value in values:
            product *= _as_number(value)
        return product
    if op == "MAX":
        if not values:
            raise ValueError("MAX requires at least one operand")
        return max(_as_number(value) for value in values)
    if op == "MIN":
        if not values:
            raise ValueError("MIN requires at least one operand")
        return min(_as_number(value) for value in values)
    if op == "DIFFERENCE":
        _require_arity(op, values, 2)
        result = _as_number(values[0])
        for value in values[1:]:
            result -= _as_number(value)
        return result
    if op == "RATIO":
        _require_arity(op, values, 2)
        denominator = _as_number(values[1])
        if denominator == 0:
            raise ZeroDivisionError("RATIO denominator is zero")
        return _as_number(values[0]) / denominator
    if op == "PERCENT_CHANGE":
        _require_arity(op, values, 2)
        new_value = _as_number(values[0])
        old_value = _as_number(values[1])
        if old_value == 0:
            raise ZeroDivisionError("PERCENT_CHANGE base is zero")
        return (new_value - old_value) / abs(old_value)
    if op == "GREATER_THAN":
        _require_arity(op, values, 2)
        return "yes" if _as_number(values[0]) > _as_number(values[1]) else "no"
    raise ValueError(f"unsupported canonical operator: {operator}")


def _evaluate_normalize(operation: str, values: list[float | str]) -> float | str:
    if operation == "divide":
        _require_arity(operation, values, 2)
        denominator = _as_number(values[1])
        if denominator == 0:
            raise ZeroDivisionError("normalize divide denominator is zero")
        return _as_number(values[0]) / denominator
    if operation == "multiply":
        product = 1.0
        for value in values:
            product *= _as_number(value)
        return product
    if operation == "negate":
        _require_arity(operation, values, 1)
        return -_as_number(values[0])
    if operation == "percent_to_ratio":
        _require_arity(operation, values, 1)
        return _as_number(values[0]) / 100.0
    if operation == "absolute_base_growth":
        _require_arity(operation, values, 2)
        numerator = _as_number(values[0])
        base = _as_number(values[1])
        if base == 0:
            raise ZeroDivisionError("absolute_base_growth base is zero")
        return numerator / abs(base)
    raise ValueError(f"unsupported canonical normalization: {operation}")


def _require_arity(name: str, values: list[float | str], expected: int) -> None:
    if len(values) < expected:
        raise ValueError(f"{name} requires at least {expected} operand(s)")


def _as_number(value: float | str) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean values are not numeric operands")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError as exc:
        raise TypeError(f"non-numeric canonical operand: {value!r}") from exc
