from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph, field_aliases, normalize_identifier
from graph_numeric.learning.router import RoutingResult


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
COMPARISON_RE = re.compile(
    r">=|<=|(?<![<>!])>|(?<![<>!])<|不低于|不高于|大于|小于|超过|"
    r"at\s+least|at\s+most|greater\s+than|less\s+than|above|below",
    re.IGNORECASE,
)
UNIT_RE = re.compile(r"\b(?:usd|eur|rmb|cny|million|billion|thousand|%)\b|美元|人民币|万元|亿元")


@dataclass(frozen=True)
class HybridStrategySelection:
    strategy: str
    confidence: float
    reason: str
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "confidence": self.confidence,
            "reason": self.reason,
            "features": self.features,
        }


def select_hybrid_strategy(
    query: str,
    graph: AttributeValueGraph,
    *,
    routing: RoutingResult | None = None,
) -> HybridStrategySelection:
    """Choose the vector-scalar execution strategy for one query.

    The selector is intentionally lightweight and deterministic. It does not
    replace operator routing or numerical execution; it only exposes the risk
    boundary for whether semantic grounding, scalar pruning, or both should
    dominate the plan.
    """
    features = _strategy_features(query, graph, routing=routing)
    exact_fields = features["exact_field_mentions"]
    alias_fields = features["alias_field_mentions"]
    scalar_constraints = int(features["scalar_constraint_count"])
    filter_selectivity = float(features["filter_selectivity"])
    ambiguous_field_count = int(features["ambiguous_field_count"])
    unit_conflict_fields = features["unit_conflict_fields"]

    if alias_fields and scalar_constraints:
        return HybridStrategySelection(
            strategy="hybrid",
            confidence=0.88,
            reason="alias_field_grounding_with_scalar_constraints",
            features=features,
        )
    if ambiguous_field_count > 1 and scalar_constraints:
        return HybridStrategySelection(
            strategy="hybrid",
            confidence=0.84,
            reason="ambiguous_field_grounding_with_scalar_constraints",
            features=features,
        )
    if unit_conflict_fields and scalar_constraints:
        return HybridStrategySelection(
            strategy="hybrid",
            confidence=0.8,
            reason="unit_conflict_requires_grounding_and_filtering",
            features=features,
        )
    if exact_fields and scalar_constraints >= 2 and filter_selectivity <= 0.8:
        return HybridStrategySelection(
            strategy="scalar_first",
            confidence=0.82,
            reason="exact_field_and_selective_scalar_filters",
            features=features,
        )
    if exact_fields and scalar_constraints:
        return HybridStrategySelection(
            strategy="scalar_first",
            confidence=0.76,
            reason="exact_field_with_scalar_filters",
            features=features,
        )
    if exact_fields:
        return HybridStrategySelection(
            strategy="scalar_first",
            confidence=0.7,
            reason="exact_field_match_without_semantic_ambiguity",
            features=features,
        )
    if alias_fields:
        return HybridStrategySelection(
            strategy="vector_first",
            confidence=0.72,
            reason="alias_grounding_without_selective_filters",
            features=features,
        )
    if scalar_constraints >= 2 and filter_selectivity <= 0.5:
        return HybridStrategySelection(
            strategy="scalar_first",
            confidence=0.68,
            reason="highly_selective_scalar_filters",
            features=features,
        )
    return HybridStrategySelection(
        strategy="vector_first",
        confidence=0.62,
        reason="semantic_grounding_needed_before_scalar_execution",
        features=features,
    )


def _strategy_features(
    query: str,
    graph: AttributeValueGraph,
    *,
    routing: RoutingResult | None,
) -> dict[str, Any]:
    exact_fields, alias_fields = _field_mentions(query, graph)
    years = _year_mentions(query, graph)
    industries = _industry_mentions(query, graph)
    entities = _entity_mentions(query, graph)
    unit_mentions = _unit_mentions(query)
    has_comparison = bool(COMPARISON_RE.search(query))
    candidate_fields = sorted(set(exact_fields) | set(alias_fields))
    constrained_tokens = _apply_scalar_constraints(graph, years, industries, entities)
    total_tokens = max(len(graph.tokens), 1)
    candidate_evidence = [
        token
        for token in constrained_tokens
        if not candidate_fields or token.field_name in candidate_fields
    ]
    unit_conflict_fields = [
        field
        for field in candidate_fields
        if len({token.unit for token in graph.select(field_name=field) if token.unit}) > 1
    ]
    scalar_constraint_count = (
        min(len(years), 1)
        + min(len(industries), 1)
        + min(len(entities), 1)
        + int(has_comparison)
        + int(bool(unit_mentions))
    )
    return {
        "exact_field_mentions": exact_fields,
        "alias_field_mentions": alias_fields,
        "year_mentions": years,
        "industry_mentions": industries,
        "entity_mentions": entities,
        "unit_mentions": unit_mentions,
        "has_numeric_comparison": has_comparison,
        "scalar_constraint_count": scalar_constraint_count,
        "filter_selectivity": round(len(constrained_tokens) / total_tokens, 4),
        "candidate_evidence_count": len(candidate_evidence),
        "ambiguous_field_count": len(candidate_fields),
        "unit_conflict_fields": unit_conflict_fields,
        "routed_operator": routing.operator if routing is not None else None,
        "route_type": routing.route_type if routing is not None else None,
    }


def _field_mentions(query: str, graph: AttributeValueGraph) -> tuple[list[str], list[str]]:
    query_lower = query.lower()
    query_normalized = normalize_identifier(query)
    exact: list[str] = []
    alias: list[str] = []
    for field in graph.fields:
        field_lower = field.lower()
        field_phrase = field.replace("_", " ").lower()
        if field_lower in query_normalized or field_phrase in query_lower:
            exact.append(field)
            continue
        for alias_text in field_aliases(field):
            alias_lower = alias_text.lower()
            alias_normalized = normalize_identifier(alias_text)
            if alias_normalized in {field_lower, normalize_identifier(field_phrase)}:
                continue
            if _contains_phrase(query_lower, query_normalized, alias_lower, alias_normalized):
                alias.append(field)
                break
    return sorted(exact), sorted(alias)


def _contains_phrase(
    query_lower: str,
    query_normalized: str,
    alias_lower: str,
    alias_normalized: str,
) -> bool:
    if not alias_lower.strip():
        return False
    if " " in alias_lower:
        return alias_lower in query_lower or alias_normalized in query_normalized
    if re.search(rf"(?<![0-9a-zA-Z]){re.escape(alias_lower)}(?![0-9a-zA-Z])", query_lower):
        return True
    return alias_normalized in query_normalized.split("_")


def _year_mentions(query: str, graph: AttributeValueGraph) -> list[int]:
    graph_years = set(graph.years)
    years = sorted({int(match) for match in YEAR_RE.findall(query)})
    if graph_years:
        years = [year for year in years if year in graph_years]
    return years


def _industry_mentions(query: str, graph: AttributeValueGraph) -> list[str]:
    query_normalized = normalize_identifier(query)
    matches = []
    for industry in graph.industries:
        industry_normalized = normalize_identifier(industry)
        if industry_normalized and industry_normalized in query_normalized:
            matches.append(industry)
    return sorted(matches)


def _entity_mentions(query: str, graph: AttributeValueGraph) -> list[str]:
    query_normalized = normalize_identifier(query)
    entities = sorted({token.company_name for token in graph.tokens})
    matches = []
    for entity in entities:
        entity_normalized = normalize_identifier(entity)
        if entity_normalized and entity_normalized in query_normalized:
            matches.append(entity)
    return matches


def _unit_mentions(query: str) -> list[str]:
    return sorted({match.group(0).lower() for match in UNIT_RE.finditer(query)})


def _apply_scalar_constraints(
    graph: AttributeValueGraph,
    years: list[int],
    industries: list[str],
    entities: list[str],
):
    selected = graph.tokens
    if years:
        year_set = set(years)
        selected = tuple(token for token in selected if token.year in year_set)
    if industries:
        industry_set = set(industries)
        selected = tuple(token for token in selected if token.industry in industry_set)
    if entities:
        entity_set = set(entities)
        selected = tuple(token for token in selected if token.company_name in entity_set)
    return selected
