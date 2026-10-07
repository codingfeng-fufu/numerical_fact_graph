"""Bridge between JSONL real-data samples and the graph_numeric framework.

Converts samples from sum_dataset2.jsonl / count_600.jsonl into
AttributeValueGraph + OperatorPlan so the Executor can evaluate them.

Two plan modes:
  oracle    — uses gold field / threshold directly from the sample
  predicted — uses FieldGrounder + regex extraction from query_text

Handles both single-condition and multi-condition COUNT samples.
"""

from __future__ import annotations

import re
from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource, normalize_identifier
from graph_numeric.learning.field_grounder import FieldGrounder
from graph_numeric.operators.operator_plan import OperatorPlan, Slot

# Node attributes that are metadata, not numeric values
_NON_VALUE_ATTRS = frozenset({"industry", "type", "name", "id"})

# Condition operator patterns (order matters: longer / more specific first)
_OP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r">=|不低于|大于等于"), ">="),
    (re.compile(r"<=|不高于|小于等于"), "<="),
    (re.compile(r"(?<![<>!])>(?!=)|超过|大于|exceeds?|greater than"), ">"),
    (re.compile(r"(?<![<>!])<(?!=)|小于|less than"), "<"),
    (re.compile(r"==|等于|equals?"), "=="),
]

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def jsonl_to_graph(sample: dict[str, Any]) -> AttributeValueGraph:
    """Convert a JSONL sample's graph.nodes into an AttributeValueGraph.

    Each numeric node attribute becomes one AttributeValueToken.
    year=None because the real datasets don't carry a year dimension.
    """
    tokens: list[AttributeValueToken] = []
    for node_idx, node in enumerate(sample["graph"]["nodes"]):
        attrs: dict[str, Any] = node.get("attributes", {})
        company_name: str = node["name"]
        industry: str | None = attrs.get("industry")
        entity_id = normalize_identifier(company_name)
        src = TokenSource(
            document_id=sample.get("sample_id"),
            row=node_idx,
        )

        for field_name, raw_value in attrs.items():
            if field_name in _NON_VALUE_ATTRS:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=company_name,
                    field_name=field_name,
                    field_label=field_name.replace("_", " "),
                    value=value,
                    year=None,
                    industry=industry,
                    source=src,
                )
            )

    return AttributeValueGraph(tuple(tokens), source_name=sample.get("sample_id"))


# ---- Condition extraction ----

def extract_condition_from_query(query: str) -> tuple[str, float] | None:
    """Extract the first (op, threshold) from a query string.

    Handles English (>=, <=, ==, exceeds) and Chinese (大于等于, 等于, 超过 ...) forms.
    Returns None if no condition is found.
    """
    for op_re, canonical_op in _OP_PATTERNS:
        m = op_re.search(query)
        if m:
            rest = query[m.end():]
            num_m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)", rest)
            if num_m:
                raw = num_m.group(1).replace(",", ".")
                return canonical_op, float(raw)
    return None


def extract_threshold_from_query(query: str) -> float | None:
    """Return only the threshold value, ignoring the operator."""
    result = extract_condition_from_query(query)
    return result[1] if result else None


def extract_all_conditions_from_query(
    query: str, known_fields: list[str]
) -> list[tuple[str, str, float]]:
    """Extract ALL (field, op, threshold) triples from a multi-condition query.

    Splits on '且' or 'and', then matches field name + condition per segment.
    """
    segments = re.split(r"且|and", query, flags=re.IGNORECASE)
    results: list[tuple[str, str, float]] = []

    for seg in segments:
        cond = extract_condition_from_query(seg)
        if cond is None:
            continue
        op, value = cond
        # Find which known field is mentioned in this segment
        field = _match_field_in_text(seg, known_fields)
        if field:
            results.append((field, op, value))

    return results


def _match_field_in_text(text: str, fields: list[str]) -> str | None:
    """Return the first field name (or alias) found as a substring of text."""
    text_lower = text.lower()
    for f in fields:
        if f.lower() in text_lower or f.replace("_", " ").lower() in text_lower:
            return f
    return None


# ---- Direct count for multi-condition samples ----

def _apply_op(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == "==":
        return abs(value - threshold) < 1e-9
    return False


def count_multi_condition(
    conditions: list[dict[str, Any]], graph: AttributeValueGraph
) -> float:
    """Count distinct companies satisfying ALL conditions (AND logic)."""
    entity_ids = {t.entity_id for t in graph.tokens}
    count = 0
    for entity_id in entity_ids:
        # Build a {field: value} map for this entity
        entity_vals: dict[str, float] = {
            t.field_name: t.value
            for t in graph.tokens
            if t.entity_id == entity_id
        }
        if all(
            _apply_op(entity_vals.get(c["attribute"], float("nan")), c["op"], float(c["value"]))
            for c in conditions
        ):
            count += 1
    return float(count)


# ---- Helper ----

def _make_slot(surface: str, value: object, confidence: float = 1.0) -> Slot:
    return Slot(surface=surface, grounded_value=value, confidence=confidence)


def is_multi_condition(sample: dict[str, Any]) -> bool:
    return "conditions" in sample["task"]["target"]


# ---- Oracle plan builders ----

def oracle_sum_plan(sample: dict[str, Any]) -> OperatorPlan:
    """Build a gold SUM plan directly from sample metadata."""
    field: str = sample["task"]["target"]["attribute"]
    return OperatorPlan(
        operator="SUM",
        slots={"target_field": _make_slot(field, field)},
    )


def oracle_count_plan(sample: dict[str, Any]) -> OperatorPlan | None:
    """Build a gold COUNT plan. Returns None for multi-condition samples
    (caller should use count_multi_condition() directly instead)."""
    if is_multi_condition(sample):
        return None
    target = sample["task"]["target"]
    field: str = target["attribute"]
    condition = target.get("condition", {})
    op: str = condition.get("op", ">=")
    threshold: float = float(condition.get("value", 0))
    return OperatorPlan(
        operator="COUNT",
        slots={
            "count_target": _make_slot("company", "company"),
            "condition_field": _make_slot(field, field),
            "condition_op": _make_slot(op, op),
            "condition_threshold": _make_slot(str(threshold), threshold),
        },
    )


# ---- Predicted plan builders ----

def predicted_sum_plan(
    sample: dict[str, Any],
    graph: AttributeValueGraph,
    grounder: FieldGrounder,
) -> OperatorPlan:
    """Build a predicted SUM plan using FieldGrounder on query_text."""
    query: str = sample["task"]["query_text"]
    grounding = grounder.ground(query, list(graph.fields))
    industry: str | None = _extract_industry(query, graph)

    slots: dict[str, Slot] = {
        "target_field": _make_slot(query, grounding.field_name, grounding.confidence),
    }
    if industry:
        slots["industry"] = _make_slot(industry, industry, 0.9)

    return OperatorPlan(operator="SUM", slots=slots, confidence=grounding.confidence)


def predicted_count_plan(
    sample: dict[str, Any],
    graph: AttributeValueGraph,
    grounder: FieldGrounder,
) -> OperatorPlan | None:
    """Build a predicted single-condition COUNT plan.

    Returns None for multi-condition samples (caller handles separately).
    """
    if is_multi_condition(sample):
        return None

    query: str = sample["task"]["query_text"]
    grounding = grounder.ground(query, list(graph.fields))
    condition = extract_condition_from_query(query)
    op = condition[0] if condition else ">="
    threshold = condition[1] if condition else 0.0

    return OperatorPlan(
        operator="COUNT",
        slots={
            "count_target": _make_slot("company", "company", 0.9),
            "condition_field": _make_slot(query, grounding.field_name, grounding.confidence),
            "condition_op": _make_slot(op, op, 0.9 if condition else 0.3),
            "condition_threshold": _make_slot(str(threshold), threshold, 0.95 if condition else 0.1),
        },
        confidence=grounding.confidence,
    )


def predicted_count_multi(
    sample: dict[str, Any],
    graph: AttributeValueGraph,
) -> float:
    """Predicted multi-condition count: extract all conditions from query via regex."""
    query: str = sample["task"]["query_text"]
    fields = list(graph.fields)
    extracted = extract_all_conditions_from_query(query, fields)
    if not extracted:
        return 0.0
    # Rebuild as conditions dicts for count_multi_condition
    conds = [{"attribute": f, "op": o, "value": v} for f, o, v in extracted]
    return count_multi_condition(conds, graph)


# ---- Shared helper ----

def _extract_industry(query: str, graph: AttributeValueGraph) -> str | None:
    """Find the longest industry name that is a substring of the query."""
    query_upper = query.upper()
    candidates: list[str] = []
    seen: set[str] = set()
    for token in graph.tokens:
        ind = token.industry
        if ind and ind not in seen:
            seen.add(ind)
            if ind.upper() in query_upper:
                candidates.append(ind)
    return max(candidates, key=len) if candidates else None
