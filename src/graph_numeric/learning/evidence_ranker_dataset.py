from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Iterable

from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
    normalize_identifier,
    token_matches_year,
)
from graph_numeric.learning.evidence_ranker import (
    LeafTokenExample,
    extract_leaf_token_features,
    heuristic_score_from_features,
)
from graph_numeric.core.expression_plan import (
    EvidenceQuery,
    ExpressionNode,
    ExpressionPlan,
    parse_expression_plan,
)


_PROGRAM_NUMBER_RE = re.compile(r"(?<![A-Za-z_#\d])-?\d[\d,]*(?:\.\d+)?")
_CONSTANT_RE = re.compile(r"\bconst_-?\d[\d,]*(?:\.\d+)?\b", re.IGNORECASE)
_GENERIC_LEAF_TERMS = {
    "amount",
    "cost",
    "costs",
    "expense",
    "expenses",
    "income",
    "net",
    "revenue",
    "sale",
    "sales",
    "total",
}
_SEGMENT_ROLES = {"part", "left", "right", "numerator", "segment"}


def collect_evidence_leaves(plan: ExpressionPlan | None) -> list[EvidenceQuery]:
    if plan is None:
        return []

    leaves: list[EvidenceQuery] = []

    def visit(node: ExpressionNode) -> None:
        if node.evidence_query is not None:
            leaves.append(node.evidence_query)
        for child in node.children:
            visit(child)

    visit(plan.root)
    return leaves


def numeric_values_from_program(program: object) -> list[float]:
    if program is None:
        return []
    text = _CONSTANT_RE.sub(" ", str(program))
    return [
        float(match.group(0).replace(",", ""))
        for match in _PROGRAM_NUMBER_RE.finditer(text)
        if not _looks_like_context_year(match)
    ]


def target_values_for_evidence_leaves(
    *,
    program: object,
    leaves: list[EvidenceQuery],
    gold_answer: object | None = None,
) -> tuple[list[float], str]:
    program_values = numeric_values_from_program(program)
    if len(program_values) >= len(leaves):
        return program_values, "program"

    if _can_use_gold_answer_as_single_leaf_target(leaves):
        answer_values = numeric_values_from_program(gold_answer)
        if len(answer_values) == 1:
            return answer_values, "gold_answer_single_leaf"

    return program_values, "program"


def build_examples_for_sample(
    dataset: str,
    sample_id: str,
    question: str,
    graph: AttributeValueGraph,
    program: object,
    gold_answer: object | None = None,
) -> list[LeafTokenExample]:
    leaves = collect_evidence_leaves(parse_expression_plan(question))
    gold_values, _ = target_values_for_evidence_leaves(
        program=program,
        leaves=leaves,
        gold_answer=gold_answer,
    )
    if not leaves or len(gold_values) < len(leaves):
        return []

    examples: list[LeafTokenExample] = []
    for leaf_index, leaf in enumerate(leaves):
        gold_value = gold_values[leaf_index]
        positive_token_ids = {
            token.token_id
            for token in graph.tokens
            if _close_value(token.value, gold_value)
            and (
                leaf.time_surface is None
                or token_matches_year(token, leaf.time_surface)
            )
            and _token_supports_leaf(token, leaf)
        }
        if not positive_token_ids:
            return []

        leaf_sample_id = f"{dataset}:{sample_id}:leaf{leaf_index}"
        for token in graph.tokens:
            features = extract_leaf_token_features(
                query=question,
                leaf=leaf,
                token=token,
            )
            label = 1 if token.token_id in positive_token_ids else 0
            if label == 0 and _is_hard_negative(features):
                features["hard_negative"] = 1.0
            examples.append(
                LeafTokenExample(
                    query=question,
                    leaf=leaf,
                    token_id=token.token_id,
                    label=label,
                    features=features,
                    sample_id=leaf_sample_id,
                )
            )
    return examples


def write_jsonl(path: str | Path, examples: Iterable[LeafTokenExample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example_to_json(example), ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[LeafTokenExample]:
    input_path = Path(path)
    examples: list[LeafTokenExample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(example_from_json(json.loads(line)))
    return examples


def example_to_json(example: LeafTokenExample) -> dict[str, object]:
    return {
        "query": example.query,
        "leaf": example.leaf.to_dict(),
        "token_id": example.token_id,
        "label": example.label,
        "features": dict(example.features),
        "sample_id": example.sample_id,
    }


def example_from_json(payload: dict[str, object]) -> LeafTokenExample:
    _require_keys(payload, ("query", "leaf", "token_id", "label", "features", "sample_id"))
    leaf_payload = payload["leaf"]
    if not isinstance(leaf_payload, dict):
        raise ValueError("LeafTokenExample JSON requires a leaf object.")
    _require_keys(leaf_payload, ("field_surface", "time_surface", "entity_surface", "unit_surface", "role"))
    features_payload = payload["features"]
    if not isinstance(features_payload, dict):
        raise ValueError("LeafTokenExample JSON features must be an object.")
    label = int(payload["label"])
    if label not in {0, 1}:
        raise ValueError("LeafTokenExample JSON label must be 0 or 1.")

    return LeafTokenExample(
        query=str(payload["query"]),
        leaf=EvidenceQuery(
            field_surface=str(leaf_payload["field_surface"]),
            time_surface=_optional_string(leaf_payload.get("time_surface")),
            entity_surface=_optional_string(leaf_payload.get("entity_surface")),
            unit_surface=_optional_string(leaf_payload.get("unit_surface")),
            role=str(leaf_payload.get("role", "value")),
        ),
        token_id=str(payload["token_id"]),
        label=label,
        features={
            str(feature_name): float(feature_value)
            for feature_name, feature_value in features_payload.items()
        },
        sample_id=_optional_string(payload["sample_id"]),
    )


def _close_value(left: float, right: float) -> bool:
    if math.isclose(left, right, abs_tol=1e-6, rel_tol=1e-6):
        return True
    return math.isclose(left * 100.0, right, abs_tol=1e-6, rel_tol=1e-6)


def _is_hard_negative(features: dict[str, float]) -> bool:
    if features.get("year_match", 0.0) <= 0.0:
        return False
    return heuristic_score_from_features(features) >= 2.0


def _can_use_gold_answer_as_single_leaf_target(leaves: list[EvidenceQuery]) -> bool:
    if len(leaves) != 1:
        return False
    return normalize_identifier(leaves[0].role) == "value"


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _looks_like_context_year(match: re.Match[str]) -> bool:
    value = match.group(0).replace(",", "")
    if not re.fullmatch(r"(?:19|20)\d{2}", value):
        return False
    before = match.string[max(0, match.start() - 1):match.start()]
    after = match.string[match.end():match.end() + 1]
    return before == "(" and after == ")"


def _token_supports_leaf(token: AttributeValueToken, leaf: EvidenceQuery) -> bool:
    leaf_terms = _terms(leaf.field_surface)
    if not leaf_terms:
        return True
    token_terms = _terms(token.token_id)
    token_terms.update(_terms(token.entity_id))
    token_terms.update(_terms(token.company_name))
    token_terms.update(_terms(token.field_name))
    token_terms.update(_terms(token.field_label))
    token_terms.update(_terms(token.raw_label))
    if token.dimensions:
        for value in token.dimensions.values():
            token_terms.update(_terms(value))
    distinctive_leaf_terms = leaf_terms - _GENERIC_LEAF_TERMS
    role = normalize_identifier(leaf.role)
    has_segment = _dimension_text(token, "segment") is not None
    if role in _SEGMENT_ROLES and distinctive_leaf_terms:
        return bool(distinctive_leaf_terms & token_terms)
    if has_segment and distinctive_leaf_terms:
        return bool(distinctive_leaf_terms & token_terms)
    return bool(leaf_terms & token_terms)


def _terms(value: object | None) -> set[str]:
    if value is None:
        return set()
    return {
        term
        for term in normalize_identifier(str(value)).split("_")
        if term and not term.isdigit()
    }


def _require_keys(payload: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in payload:
            raise ValueError(f"LeafTokenExample JSON missing required field: {key}")


def _dimension_text(token: AttributeValueToken, key: str) -> str | None:
    if not token.dimensions:
        return None
    value = token.dimensions.get(key)
    if value is None:
        return None
    return str(value)
