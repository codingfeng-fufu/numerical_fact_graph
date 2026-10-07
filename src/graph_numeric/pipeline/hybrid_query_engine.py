from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, field_aliases, normalize_identifier
from graph_numeric.extraction.document_extraction import EvidenceChunk, build_evidence_chunks
from graph_numeric.operators.operator_solvers import extract_condition
from graph_numeric.pipeline.strategy_selector import select_hybrid_strategy
from graph_numeric.core.unit_resolver import UnitIncompatibleError, UnitResolver


@dataclass(frozen=True)
class EvidenceVectorCandidate:
    rank: int
    chunk_id: str
    score: float
    token_ids: tuple[str, ...]
    entities: tuple[str, ...]
    industries: tuple[str, ...]
    fields: tuple[str, ...]
    years: tuple[int, ...]
    units: tuple[str, ...]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "token_ids": list(self.token_ids),
            "entities": list(self.entities),
            "industries": list(self.industries),
            "fields": list(self.fields),
            "years": list(self.years),
            "units": list(self.units),
            "text": self.text,
        }


@dataclass(frozen=True)
class ScalarPredicate:
    kind: str
    key: str
    op: str
    value: Any
    surface: str
    field: str | None = None
    unit: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "op": self.op,
            "value": self.value,
            "surface": self.surface,
            "field": self.field,
            "unit": self.unit,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ScalarPruningStep:
    predicate: ScalarPredicate
    before_count: int
    after_count: int
    removed_count: int
    selectivity: float
    candidate_chunk_ids: tuple[str, ...]
    rejected_chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate.to_dict(),
            "before_count": self.before_count,
            "after_count": self.after_count,
            "removed_count": self.removed_count,
            "selectivity": self.selectivity,
            "candidate_chunk_ids": list(self.candidate_chunk_ids),
            "rejected_chunk_ids": list(self.rejected_chunk_ids),
        }


@dataclass(frozen=True)
class ScalarPruningTrace:
    initial_count: int
    final_count: int
    predicates: tuple[ScalarPredicate, ...]
    steps: tuple[ScalarPruningStep, ...]
    final_chunk_ids: tuple[str, ...]
    candidate_reduction_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_count": self.initial_count,
            "final_count": self.final_count,
            "predicates": [predicate.to_dict() for predicate in self.predicates],
            "steps": [step.to_dict() for step in self.steps],
            "final_chunk_ids": list(self.final_chunk_ids),
            "candidate_reduction_ratio": self.candidate_reduction_ratio,
        }


@dataclass(frozen=True)
class CostBasedDecision:
    strategy: str
    confidence: float
    reason: str
    costs: dict[str, float]
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "confidence": self.confidence,
            "reason": self.reason,
            "costs": self.costs,
            "features": self.features,
        }


@dataclass(frozen=True)
class HybridQueryAnalysis:
    vector_candidates: tuple[EvidenceVectorCandidate, ...]
    scalar_trace: ScalarPruningTrace
    optimizer: CostBasedDecision
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_candidates": [candidate.to_dict() for candidate in self.vector_candidates],
            "scalar_pruning": self.scalar_trace.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "elapsed_ms": self.elapsed_ms,
        }


class EvidenceVectorIndex:
    """Deterministic sparse-vector index over evidence chunks.

    It is deliberately local and dependency-free so benchmarks can run offline.
    The production-facing interface mirrors a vector index: build over chunks,
    encode query text, return top-k candidates with similarity scores.
    """

    def __init__(self, chunks: Iterable[EvidenceChunk]) -> None:
        self.chunks = tuple(chunks)
        self._vectors = tuple(_text_vector(_chunk_index_text(chunk)) for chunk in self.chunks)
        self._norms = tuple(_norm(vector) for vector in self._vectors)

    @classmethod
    def from_graph(cls, graph: AttributeValueGraph) -> EvidenceVectorIndex:
        return cls(build_evidence_chunks(graph))

    def search(self, query: str, *, top_k: int = 5) -> tuple[EvidenceVectorCandidate, ...]:
        query_vector = _text_vector(query)
        query_norm = _norm(query_vector)
        scored = []
        for chunk, vector, norm in zip(self.chunks, self._vectors, self._norms, strict=True):
            score = _cosine(query_vector, query_norm, vector, norm)
            scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            EvidenceVectorCandidate(
                rank=index + 1,
                chunk_id=chunk.chunk_id,
                score=round(score, 6),
                token_ids=chunk.token_ids,
                entities=chunk.entities,
                industries=chunk.industries,
                fields=chunk.fields,
                years=chunk.years,
                units=chunk.units,
                text=chunk.text,
            )
            for index, (score, chunk) in enumerate(scored[:top_k])
        )


def analyze_hybrid_query(
    query: str,
    graph: AttributeValueGraph,
    *,
    top_k: int = 5,
) -> HybridQueryAnalysis:
    started = time.perf_counter()
    chunks = build_evidence_chunks(graph)
    vector_candidates = EvidenceVectorIndex(chunks).search(query, top_k=top_k)
    predicates = build_scalar_predicates(query, graph)
    scalar_trace = apply_scalar_predicates(chunks, graph.tokens, predicates)
    optimizer = optimize_hybrid_strategy(
        query,
        graph,
        vector_candidates=vector_candidates,
        scalar_trace=scalar_trace,
    )
    return HybridQueryAnalysis(
        vector_candidates=vector_candidates,
        scalar_trace=scalar_trace,
        optimizer=optimizer,
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 4),
    )


def build_scalar_predicates(query: str, graph: AttributeValueGraph) -> tuple[ScalarPredicate, ...]:
    predicates: list[ScalarPredicate] = []
    for year in _year_mentions(query, graph):
        predicates.append(ScalarPredicate("time", "year", "=", year, str(year), confidence=0.99))
    for industry in _industry_mentions(query, graph):
        predicates.append(ScalarPredicate("dimension", "industry", "=", industry, industry, confidence=0.95))
    for entity in _entity_mentions(query, graph):
        predicates.append(ScalarPredicate("dimension", "entity", "=", entity, entity, confidence=0.9))
    for field, source in _field_mentions(query, graph):
        predicates.append(ScalarPredicate("field", "field", "=", field, source, confidence=0.9))
    for unit in _unit_mentions(query):
        predicates.append(ScalarPredicate("unit", "unit", "contains", unit, unit, confidence=0.75))

    condition = extract_condition(query)
    if condition is not None:
        op, threshold, confidence, condition_unit = condition
        field = next((predicate.value for predicate in predicates if predicate.kind == "field"), None)
        surface = f"{op} {threshold}"
        if condition_unit is not None:
            surface = f"{surface} {condition_unit}"
        predicates.append(
            ScalarPredicate(
                "value",
                "value",
                op,
                threshold,
                surface,
                field=str(field) if field is not None else None,
                unit=condition_unit,
                confidence=confidence,
            )
        )
    return tuple(_dedupe_predicates(predicates))


def apply_scalar_predicates(
    chunks: Iterable[EvidenceChunk],
    tokens: Iterable[AttributeValueToken],
    predicates: Iterable[ScalarPredicate],
) -> ScalarPruningTrace:
    token_by_id = {token.token_id: token for token in tokens}
    candidates = tuple(chunks)
    initial_count = len(candidates)
    steps: list[ScalarPruningStep] = []
    predicate_tuple = tuple(predicates)
    for predicate in predicate_tuple:
        before = len(candidates)
        before_chunk_ids = tuple(chunk.chunk_id for chunk in candidates)
        candidates = tuple(
            chunk
            for chunk in candidates
            if _chunk_satisfies(chunk, token_by_id, predicate)
        )
        after = len(candidates)
        after_chunk_ids = tuple(chunk.chunk_id for chunk in candidates)
        steps.append(
            ScalarPruningStep(
                predicate=predicate,
                before_count=before,
                after_count=after,
                removed_count=before - after,
                selectivity=round(after / max(before, 1), 4),
                candidate_chunk_ids=after_chunk_ids,
                rejected_chunk_ids=tuple(
                    chunk_id for chunk_id in before_chunk_ids if chunk_id not in set(after_chunk_ids)
                ),
            )
        )
    final_count = len(candidates)
    reduction = 1.0 - (final_count / max(initial_count, 1))
    return ScalarPruningTrace(
        initial_count=initial_count,
        final_count=final_count,
        predicates=predicate_tuple,
        steps=tuple(steps),
        final_chunk_ids=tuple(chunk.chunk_id for chunk in candidates),
        candidate_reduction_ratio=round(reduction, 4),
    )


def optimize_hybrid_strategy(
    query: str,
    graph: AttributeValueGraph,
    *,
    vector_candidates: tuple[EvidenceVectorCandidate, ...],
    scalar_trace: ScalarPruningTrace,
) -> CostBasedDecision:
    selector = select_hybrid_strategy(query, graph)
    features = dict(selector.features)
    total = max(scalar_trace.initial_count, 1)
    final = scalar_trace.final_count
    top_k = len(vector_candidates)
    top_score = vector_candidates[0].score if vector_candidates else 0.0
    second_score = vector_candidates[1].score if len(vector_candidates) > 1 else 0.0
    vector_score_gap = max(top_score - second_score, 0.0)
    scalar_selectivity = final / total
    alias_count = len(features.get("alias_field_mentions") or [])
    exact_count = len(features.get("exact_field_mentions") or [])
    constraint_count = int(features.get("scalar_constraint_count") or 0)
    ambiguity = int(features.get("ambiguous_field_count") or 0)
    unit_conflicts = len(features.get("unit_conflict_fields") or [])
    entity_count = len(features.get("entity_mentions") or [])
    year_count = len(features.get("year_mentions") or [])

    vector_cost = 1.0 + 0.12 * top_k + 0.35 * constraint_count
    scalar_cost = 0.08 * total + 0.18 * final + 1.35 * alias_count + 0.4 * unit_conflicts
    if constraint_count == 0:
        scalar_cost += 0.8
    if top_score == 0:
        vector_cost += 0.8
    hybrid_cost = 0.58 * vector_cost + 0.58 * scalar_cost + 0.18 * ambiguity
    if alias_count and constraint_count:
        hybrid_cost -= 0.9
    if exact_count and scalar_selectivity <= 0.5:
        scalar_cost -= 0.35

    costs = {
        "vector_first": round(max(vector_cost, 0.01), 4),
        "scalar_first": round(max(scalar_cost, 0.01), 4),
        "hybrid": round(max(hybrid_cost, 0.01), 4),
    }
    strategy = min(costs, key=costs.get)
    if alias_count and constraint_count:
        strategy = "hybrid"
    elif exact_count and constraint_count and scalar_selectivity <= 0.55:
        strategy = "scalar_first"
    elif constraint_count == 0 and alias_count:
        strategy = "vector_first"

    sorted_costs = sorted(costs.values())
    margin = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else 0.0
    decision_features = {
        **features,
        "chunk_count": total,
        "vector_top_k": top_k,
        "vector_top_score": round(top_score, 6),
        "vector_score_gap": round(vector_score_gap, 6),
        "scalar_predicate_count": len(scalar_trace.predicates),
        "scalar_final_count": final,
        "scalar_selectivity": round(scalar_selectivity, 4),
        "field_alias_count": alias_count,
        "exact_field_count": exact_count,
        "unit_conflict_count": unit_conflicts,
        "entity_mention_count": entity_count,
        "year_mention_count": year_count,
        "candidate_reduction_ratio": scalar_trace.candidate_reduction_ratio,
    }
    return CostBasedDecision(
        strategy=strategy,
        confidence=round(min(0.95, 0.62 + margin * 0.16 + abs(0.5 - scalar_selectivity) * 0.22), 4),
        reason=_decision_reason(strategy, alias_count, exact_count, constraint_count, scalar_selectivity),
        costs=costs,
        features=decision_features,
    )


def _chunk_index_text(chunk: EvidenceChunk) -> str:
    field_text = " ".join(
        " ".join((field, field.replace("_", " "), *field_aliases(field)))
        for field in chunk.fields
    )
    return " ".join(
        [
            chunk.text,
            " ".join(chunk.entities),
            " ".join(chunk.industries),
            field_text,
            " ".join(str(year) for year in chunk.years),
            " ".join(chunk.units),
        ]
    )


def _text_vector(text: str) -> Counter[str]:
    normalized = normalize_identifier(text)
    tokens = [token for token in normalized.split("_") if token]
    tokens.extend(_char_ngrams(normalized, n=3))
    return Counter(tokens)


def _char_ngrams(text: str, *, n: int) -> list[str]:
    compact = text.replace("_", "")
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[index:index + n] for index in range(len(compact) - n + 1)]


def _norm(vector: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(
    left: Counter[str],
    left_norm: float,
    right: Counter[str],
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    return dot / (left_norm * right_norm)


def _year_mentions(query: str, graph: AttributeValueGraph) -> list[int]:
    years = sorted({int(value) for value in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", query)})
    graph_years = set(graph.years)
    return [year for year in years if not graph_years or year in graph_years]


def _industry_mentions(query: str, graph: AttributeValueGraph) -> list[str]:
    query_normalized = normalize_identifier(query)
    return [
        industry
        for industry in graph.industries
        if normalize_identifier(industry) in query_normalized
    ]


def _entity_mentions(query: str, graph: AttributeValueGraph) -> list[str]:
    query_normalized = normalize_identifier(query)
    entities = sorted({token.company_name for token in graph.tokens})
    return [
        entity
        for entity in entities
        if normalize_identifier(entity) in query_normalized
    ]


def _field_mentions(query: str, graph: AttributeValueGraph) -> list[tuple[str, str]]:
    query_lower = query.lower()
    query_normalized = normalize_identifier(query)
    mentions: list[tuple[str, str]] = []
    for field in graph.fields:
        field_phrase = field.replace("_", " ").lower()
        if field in query_normalized or field_phrase in query_lower:
            mentions.append((field, field))
            continue
        for alias in field_aliases(field):
            alias_lower = alias.lower()
            alias_normalized = normalize_identifier(alias)
            if alias_normalized in {field, normalize_identifier(field_phrase)}:
                continue
            if (
                alias_lower in query_lower
                or alias_normalized in query_normalized.split("_")
                or alias_normalized in query_normalized
            ):
                mentions.append((field, alias))
                break
    return mentions


def _unit_mentions(query: str) -> list[str]:
    return sorted({
        match.group(0).lower()
        for match in re.finditer(r"\b(?:usd|eur|rmb|cny|%)\b|美元|人民币|元", query, re.IGNORECASE)
    })


def _dedupe_predicates(predicates: list[ScalarPredicate]) -> list[ScalarPredicate]:
    seen: set[tuple[str, str, str, str, str | None, str | None]] = set()
    deduped: list[ScalarPredicate] = []
    for predicate in predicates:
        key = (
            predicate.kind,
            predicate.key,
            predicate.op,
            str(predicate.value),
            predicate.field,
            predicate.unit,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(predicate)
    return deduped


def _chunk_satisfies(
    chunk: EvidenceChunk,
    token_by_id: dict[str, AttributeValueToken],
    predicate: ScalarPredicate,
) -> bool:
    if predicate.kind == "time":
        return int(predicate.value) in set(chunk.years)
    if predicate.key == "industry":
        return str(predicate.value) in set(chunk.industries)
    if predicate.key == "entity":
        return str(predicate.value) in set(chunk.entities)
    if predicate.kind == "field":
        return str(predicate.value) in set(chunk.fields)
    if predicate.kind == "unit":
        return any(_unit_compatible(value, str(predicate.value)) for value in chunk.units)
    if predicate.kind == "value":
        return any(
            _token_satisfies_value(token_by_id[token_id], predicate)
            for token_id in chunk.token_ids
            if token_id in token_by_id
        )
    return True


def _token_satisfies_value(token: AttributeValueToken, predicate: ScalarPredicate) -> bool:
    if predicate.field is not None and token.field_name != predicate.field:
        return False
    threshold = float(predicate.value)
    value = _value_in_predicate_unit(token, predicate.unit)
    if value is None:
        return False
    if predicate.op == ">":
        return value > threshold
    if predicate.op == ">=":
        return value >= threshold
    if predicate.op == "<":
        return value < threshold
    if predicate.op == "<=":
        return value <= threshold
    if predicate.op == "==":
        return value == threshold
    return False


def _value_in_predicate_unit(token: AttributeValueToken, requested_unit: str | None) -> float | None:
    if requested_unit is None:
        return token.value
    if not token.unit:
        return None
    resolver = UnitResolver()
    try:
        return resolver.normalize(
            token.value,
            resolver.detect(str(token.unit)),
            resolver.detect(requested_unit),
        )
    except UnitIncompatibleError:
        return None


def _unit_compatible(token_unit: str, requested_unit: str) -> bool:
    resolver = UnitResolver()
    token_info = resolver.detect(token_unit)
    requested_info = resolver.detect(requested_unit)
    if requested_info.unit_category == "unknown":
        return normalize_identifier(requested_unit) in normalize_identifier(token_unit)
    if token_info.unit_category != requested_info.unit_category:
        return False
    if (
        requested_info.unit_category == "money"
        and token_info.currency_code
        and requested_info.currency_code
        and token_info.currency_code != requested_info.currency_code
    ):
        return False
    return True


def _decision_reason(
    strategy: str,
    alias_count: int,
    exact_count: int,
    constraint_count: int,
    scalar_selectivity: float,
) -> str:
    if strategy == "hybrid" and alias_count and constraint_count:
        return "semantic_alias_grounding_with_scalar_pruning"
    if strategy == "scalar_first" and exact_count and scalar_selectivity <= 0.55:
        return "exact_fields_with_selective_scalar_predicates"
    if strategy == "vector_first" and constraint_count == 0:
        return "semantic_grounding_without_selective_scalar_predicates"
    return "lowest_estimated_hybrid_query_cost"
