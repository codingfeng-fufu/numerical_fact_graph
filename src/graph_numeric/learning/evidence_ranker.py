from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from graph_numeric.core.attribute_graph import (
    AttributeValueToken,
    field_aliases,
    normalize_identifier,
    token_matches_year,
)
from graph_numeric.core.expression_plan import EvidenceQuery


@dataclass(frozen=True)
class LeafTokenExample:
    query: str
    leaf: EvidenceQuery
    token_id: str
    label: int
    features: dict[str, float]
    sample_id: str | None = None


@dataclass(frozen=True)
class RankedEvidenceToken:
    token_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class SafeArbitrationResult:
    ranked: list[RankedEvidenceToken]
    trace: dict[str, Any]


@dataclass(frozen=True)
class EvidenceRankerMetrics:
    example_count: int
    query_count: int
    top1_accuracy: float
    top3_recall: float
    mean_reciprocal_rank: float
    role_metrics: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "query_count": self.query_count,
            "top1_accuracy": self.top1_accuracy,
            "top3_recall": self.top3_recall,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "role_metrics": self.role_metrics,
        }


def evaluate_rankings(
    grouped: Mapping[str, tuple[Sequence[LeafTokenExample], Sequence[RankedEvidenceToken]]]
) -> EvidenceRankerMetrics:
    example_count = 0
    query_count = 0
    top1_hits = 0
    top3_hits = 0
    reciprocal_ranks: list[float] = []
    role_totals: dict[str, int] = {}
    role_top1_hits: dict[str, int] = {}

    for examples, ranked in grouped.values():
        example_count += len(examples)
        query_count += 1
        positives = [example for example in examples if example.label == 1]
        if not positives:
            continue

        positive_token_ids = {example.token_id for example in positives}
        positive_roles = {
            normalize_identifier(example.leaf.role)
            for example in positives
            if normalize_identifier(example.leaf.role)
        }
        positive_ranks = [
            index
            for index, token in enumerate(ranked, start=1)
            if token.token_id in positive_token_ids
        ]
        best_rank = min(positive_ranks) if positive_ranks else None
        if best_rank == 1:
            top1_hits += 1
        if best_rank is not None and best_rank <= 3:
            top3_hits += 1
        reciprocal_ranks.append(1.0 / float(best_rank) if best_rank is not None else 0.0)

        for role in positive_roles:
            role_totals[role] = role_totals.get(role, 0) + 1
            if best_rank == 1:
                role_top1_hits[role] = role_top1_hits.get(role, 0) + 1

    role_metrics = {
        role: {
            "top1_accuracy": role_top1_hits.get(role, 0) / total if total else 0.0,
        }
        for role, total in sorted(role_totals.items())
    }
    evaluated_queries = len(reciprocal_ranks)
    return EvidenceRankerMetrics(
        example_count=example_count,
        query_count=query_count,
        top1_accuracy=(top1_hits / evaluated_queries) if evaluated_queries else 0.0,
        top3_recall=(top3_hits / evaluated_queries) if evaluated_queries else 0.0,
        mean_reciprocal_rank=(sum(reciprocal_ranks) / evaluated_queries)
        if evaluated_queries
        else 0.0,
        role_metrics=role_metrics,
    )


FEATURE_NAMES = [
    "year_match",
    "field_alias_overlap",
    "dimension_key_overlap",
    "dimension_value_overlap",
    "source_column_overlap",
    "field_specificity",
    "leaf_token_overlap",
    "query_token_overlap",
    "field_phrase_coverage",
    "row_label_overlap",
    "segment_overlap",
    "segment_present",
    "total_like",
    "unit_present",
    "same_source_sentence",
    "role_prefers_segment",
    "role_prefers_total",
    "role_dimension_consistency",
    "role_total_consistency",
    "heuristic_score",
]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "compared",
    "did",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}

_SEGMENT_ROLES = {"part", "segment", "numerator"}
_TOTAL_ROLES = {"whole", "total", "denominator"}
_TOTAL_TERMS = {"aggregate", "all", "consolidated", "overall", "total", "totals"}


def extract_leaf_token_features(
    query: str,
    leaf: EvidenceQuery,
    token: AttributeValueToken,
) -> dict[str, float]:
    leaf_field_terms = _terms(leaf.field_surface)
    leaf_terms = set(leaf_field_terms)
    leaf_terms.update(_terms(leaf.entity_surface))
    leaf_terms.update(_terms(leaf.unit_surface))
    query_terms = _terms(query)
    token_terms = _token_terms(token)

    row_label = _dimension_text(token, "row_label") or token.raw_label or token.field_label
    row_label_terms = _terms(row_label)
    segment = _dimension_text(token, "segment")
    segment_terms = _terms(segment)
    source_terms = _terms(token.source.text_excerpt if token.source is not None else None)
    source_column_terms = _terms(token.source.column if token.source is not None else None)
    unit_terms = _terms(token.unit)
    token_field_terms = _terms(token.field_name)
    token_field_phrase_terms = set(token_field_terms)
    token_field_phrase_terms.update(_terms(token.field_label))
    token_field_phrase_terms.update(_terms(token.raw_label))
    token_field_phrase_terms.update(source_column_terms)
    dimension_key_terms: set[str] = set()
    dimension_value_terms: set[str] = set()
    if token.dimensions:
        for key, value in token.dimensions.items():
            dimension_key_terms.update(_terms(key))
            dimension_value_terms.update(_terms(value))
    token_field_phrase_terms.update(row_label_terms)
    token_field_phrase_terms.update(dimension_value_terms)

    alias_scores = [
        _overlap_ratio(_terms(alias), leaf_field_terms)
        for alias in (*field_aliases(token.field_name), token.field_name, token.field_label)
        if _terms(alias)
    ]
    role = normalize_identifier(leaf.role)
    segment_present = 1.0 if segment_terms else 0.0
    dimension_present = 1.0 if dimension_value_terms else 0.0
    total_like = 1.0 if _is_total_like(token) else 0.0
    is_total_dimension = bool(dimension_value_terms & _TOTAL_TERMS)

    features = {
        "year_match": 1.0
        if leaf.time_surface is not None and token_matches_year(token, leaf.time_surface)
        else 0.0,
        "field_alias_overlap": max(alias_scores, default=0.0),
        "dimension_key_overlap": _overlap_ratio(leaf_field_terms, dimension_key_terms),
        "dimension_value_overlap": _overlap_ratio(
            leaf_field_terms,
            dimension_value_terms,
        ),
        "source_column_overlap": _overlap_ratio(
            leaf_field_terms,
            source_column_terms,
        ),
        "field_specificity": min(1.0, len(token_field_terms) / 4.0)
        if token_field_terms
        else 0.0,
        "leaf_token_overlap": _overlap_ratio(leaf_terms, token_terms),
        "query_token_overlap": _overlap_ratio(query_terms, token_terms),
        "field_phrase_coverage": _overlap_ratio(
            leaf_field_terms,
            token_field_phrase_terms,
        ),
        "row_label_overlap": _overlap_ratio(leaf_field_terms, row_label_terms),
        "segment_overlap": _overlap_ratio(leaf_field_terms, segment_terms),
        "segment_present": segment_present,
        "total_like": total_like,
        "unit_present": 1.0 if unit_terms else 0.0,
        "same_source_sentence": 1.0
        if token.source is not None
        and token.source.row is None
        and _overlap_ratio(leaf_field_terms, source_terms) > 0.0
        else 0.0,
        "role_prefers_segment": 1.0
        if role in _SEGMENT_ROLES and segment_present
        else 0.0,
        "role_prefers_total": 1.0 if role in _TOTAL_ROLES and total_like else 0.0,
        "role_dimension_consistency": 1.0
        if role in _SEGMENT_ROLES and dimension_present and not is_total_dimension
        else 0.0,
        "role_total_consistency": 1.0
        if role in _TOTAL_ROLES and (total_like or is_total_dimension)
        else 0.0,
    }
    features["heuristic_score"] = heuristic_score_from_features(features)
    features["hard_negative"] = 0.0
    return features


def heuristic_score_from_features(features: Mapping[str, float]) -> float:
    if "heuristic_score" in features:
        return float(features["heuristic_score"])
    return float(
        2.0 * features.get("year_match", 0.0)
        + 1.5 * features.get("leaf_token_overlap", 0.0)
        + 1.0 * features.get("field_alias_overlap", 0.0)
        + 0.8 * features.get("segment_overlap", 0.0)
        + 0.6 * features.get("role_prefers_segment", 0.0)
        + 0.6 * features.get("role_prefers_total", 0.0)
    )


def structural_support_score_from_features(features: Mapping[str, float]) -> float:
    return float(
        1.6 * features.get("source_column_overlap", 0.0)
        + 1.2 * features.get("dimension_value_overlap", 0.0)
        + 1.0 * features.get("role_dimension_consistency", 0.0)
        + 1.0 * features.get("role_total_consistency", 0.0)
        + 0.8 * features.get("same_source_sentence", 0.0)
        + 0.6 * features.get("field_alias_overlap", 0.0)
        + 0.5 * features.get("leaf_token_overlap", 0.0)
        + 0.5 * features.get("field_phrase_coverage", 0.0)
        + 0.4 * features.get("query_token_overlap", 0.0)
        - 0.4 * features.get("dimension_key_overlap", 0.0)
        - 0.3 * features.get("row_label_overlap", 0.0)
    )


def vectorize_feature_dicts(
    feature_rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    if not feature_rows:
        return np.zeros((0, len(feature_names)), dtype=np.float32)
    return np.asarray(
        [
            [float(row.get(feature_name, 0.0)) for feature_name in feature_names]
            for row in feature_rows
        ],
        dtype=np.float32,
    )


def rank_by_heuristic(examples: Sequence[LeafTokenExample]) -> list[RankedEvidenceToken]:
    scored = [
        (
            heuristic_score_from_features(example.features),
            example.token_id,
        )
        for example in examples
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        RankedEvidenceToken(token_id=token_id, score=float(score), rank=index)
        for index, (score, token_id) in enumerate(scored, start=1)
    ]


def apply_safe_arbitration(
    examples: Sequence[LeafTokenExample],
    heuristic_ranked: Sequence[RankedEvidenceToken],
    ranker_ranked: Sequence[RankedEvidenceToken],
    *,
    policy: str = "safe_v1",
    structural_margin: float = 0.1,
    ranker_score_margin: float = 0.05,
) -> list[RankedEvidenceToken]:
    if not heuristic_ranked:
        return list(ranker_ranked)
    if not ranker_ranked:
        return list(heuristic_ranked)

    heuristic_top = heuristic_ranked[0]
    ranker_top = ranker_ranked[0]
    if heuristic_top.token_id == ranker_top.token_id:
        return list(ranker_ranked)
    if policy not in {"safe_v1", "safe_v2", "safe_v3", "safe_v4", "safe_v5"}:
        raise ValueError("policy must be safe_v1, safe_v2, safe_v3, safe_v4, or safe_v5")

    feature_by_token = {example.token_id: example.features for example in examples}
    ranker_score_by_token = {token.token_id: token.score for token in ranker_ranked}
    heuristic_features = feature_by_token.get(heuristic_top.token_id, {})
    ranker_features = feature_by_token.get(ranker_top.token_id, {})
    structural_delta = (
        heuristic_score_from_features(ranker_features)
        - heuristic_score_from_features(heuristic_features)
    )
    ranker_score_delta = (
        ranker_top.score
        - float(ranker_score_by_token.get(heuristic_top.token_id, float("-inf")))
    )
    if structural_delta >= structural_margin and ranker_score_delta >= ranker_score_margin:
        return list(ranker_ranked)
    if (
        policy == "safe_v2"
        and _has_weak_structural_anchor(heuristic_features)
        and structural_delta >= -0.1
        and ranker_score_delta >= 0.1
    ):
        return list(ranker_ranked)
    if policy == "safe_v3" and _safe_v3_allows_override(
        heuristic_features,
        ranker_features,
        structural_delta=structural_delta,
        ranker_score_delta=ranker_score_delta,
    ):
        return list(ranker_ranked)
    if policy == "safe_v4" and _safe_v4_allows_override(
        heuristic_features,
        ranker_features,
        structural_delta=structural_delta,
        ranker_score_delta=ranker_score_delta,
    ):
        return list(ranker_ranked)
    if policy == "safe_v5" and _safe_v5_allows_override(
        heuristic_features,
        ranker_features,
        structural_delta=structural_delta,
        ranker_score_delta=ranker_score_delta,
    ):
        return list(ranker_ranked)
    return list(heuristic_ranked)


def apply_safe_arbitration_with_trace(
    examples: Sequence[LeafTokenExample],
    heuristic_ranked: Sequence[RankedEvidenceToken],
    ranker_ranked: Sequence[RankedEvidenceToken],
    *,
    policy: str = "safe_v5",
    structural_margin: float = 0.1,
    ranker_score_margin: float = 0.05,
) -> SafeArbitrationResult:
    selected = apply_safe_arbitration(
        examples,
        heuristic_ranked,
        ranker_ranked,
        policy=policy,
        structural_margin=structural_margin,
        ranker_score_margin=ranker_score_margin,
    )
    return SafeArbitrationResult(
        ranked=selected,
        trace=_safe_arbitration_trace(
            examples,
            heuristic_ranked,
            ranker_ranked,
            selected,
            policy=policy,
        ),
    )


def _safe_arbitration_trace(
    examples: Sequence[LeafTokenExample],
    heuristic_ranked: Sequence[RankedEvidenceToken],
    ranker_ranked: Sequence[RankedEvidenceToken],
    selected: Sequence[RankedEvidenceToken],
    *,
    policy: str,
) -> dict[str, Any]:
    heuristic_top = heuristic_ranked[0] if heuristic_ranked else None
    ranker_top = ranker_ranked[0] if ranker_ranked else None
    selected_top = selected[0] if selected else None
    feature_by_token = {example.token_id: example.features for example in examples}
    heuristic_features = (
        feature_by_token.get(heuristic_top.token_id, {})
        if heuristic_top is not None
        else {}
    )
    ranker_features = (
        feature_by_token.get(ranker_top.token_id, {})
        if ranker_top is not None
        else {}
    )
    if heuristic_top is None:
        decision = "accepted"
        reason = "heuristic_empty"
    elif ranker_top is None:
        decision = "rejected"
        reason = "ranker_empty"
    elif heuristic_top.token_id == ranker_top.token_id:
        decision = "tied"
        reason = "same_top_candidate"
    elif selected_top is not None and selected_top.token_id == ranker_top.token_id:
        decision = "accepted"
        reason = f"{policy}_guard_passed"
    else:
        decision = "rejected"
        reason = f"{policy}_guard_rejected"
    return {
        "policy": policy,
        "decision": decision,
        "reason": reason,
        "heuristic_top": _ranked_token_payload(heuristic_top),
        "ranker_top": _ranked_token_payload(ranker_top),
        "selected": _selected_token_payload(
            selected_top,
            decision=decision,
            ranker_top=ranker_top,
        ),
        "guard_metrics": _safe_arbitration_guard_metrics(
            heuristic_features,
            ranker_features,
            heuristic_top=heuristic_top,
            ranker_top=ranker_top,
        ),
        "top_candidates": [_ranked_token_payload(row) for row in selected[:5]],
    }


def _selected_token_payload(
    token: RankedEvidenceToken | None,
    *,
    decision: str,
    ranker_top: RankedEvidenceToken | None,
) -> dict[str, Any] | None:
    if token is None:
        return None
    source = "heuristic_guarded"
    if (
        ranker_top is not None
        and token.token_id == ranker_top.token_id
        and decision in {"accepted", "tied"}
    ):
        source = "ranker_safe_v5" if decision == "accepted" else "same_top_candidate"
    return {**_ranked_token_payload(token), "source": source}


def _safe_arbitration_guard_metrics(
    heuristic_features: Mapping[str, float],
    ranker_features: Mapping[str, float],
    *,
    heuristic_top: RankedEvidenceToken | None,
    ranker_top: RankedEvidenceToken | None,
) -> dict[str, float | bool]:
    ranker_score_delta = 0.0
    if heuristic_top is not None and ranker_top is not None:
        ranker_score_delta = float(ranker_top.score) - float(heuristic_top.score)
    return {
        "structural_delta": heuristic_score_from_features(ranker_features)
        - heuristic_score_from_features(heuristic_features),
        "support_delta": structural_support_score_from_features(ranker_features)
        - structural_support_score_from_features(heuristic_features),
        "ranker_score_delta": ranker_score_delta,
        "field_phrase_coverage_delta": float(
            ranker_features.get("field_phrase_coverage", 0.0)
        )
        - float(heuristic_features.get("field_phrase_coverage", 0.0)),
        "leaf_token_overlap_delta": float(ranker_features.get("leaf_token_overlap", 0.0))
        - float(heuristic_features.get("leaf_token_overlap", 0.0)),
        "dimension_only_gain_blocked": bool(
            _has_strong_field_anchor(heuristic_features)
            and _is_dimension_only_gain(heuristic_features, ranker_features)
        ),
    }


def _ranked_token_payload(token: RankedEvidenceToken | None) -> dict[str, Any]:
    if token is None:
        return {}
    return {"token_id": token.token_id, "score": token.score, "rank": token.rank}


def _safe_v3_allows_override(
    heuristic_features: Mapping[str, float],
    ranker_features: Mapping[str, float],
    *,
    structural_delta: float,
    ranker_score_delta: float,
) -> bool:
    support_delta = (
        structural_support_score_from_features(ranker_features)
        - structural_support_score_from_features(heuristic_features)
    )
    if _has_strong_field_anchor(heuristic_features) and _is_dimension_only_gain(
        heuristic_features,
        ranker_features,
    ):
        return False
    if support_delta >= 0.4 and structural_delta >= -0.05 and ranker_score_delta >= 0.07:
        return True
    if (
        _has_weak_structural_anchor(heuristic_features)
        and support_delta >= 0.0
        and structural_delta >= -0.05
        and ranker_score_delta >= 0.08
    ):
        return True
    return False


def _safe_v4_allows_override(
    heuristic_features: Mapping[str, float],
    ranker_features: Mapping[str, float],
    *,
    structural_delta: float,
    ranker_score_delta: float,
) -> bool:
    if _safe_v3_allows_override(
        heuristic_features,
        ranker_features,
        structural_delta=structural_delta,
        ranker_score_delta=ranker_score_delta,
    ):
        return True
    support_delta = (
        structural_support_score_from_features(ranker_features)
        - structural_support_score_from_features(heuristic_features)
    )
    if _has_strong_field_anchor(heuristic_features) and _is_dimension_only_gain(
        heuristic_features,
        ranker_features,
    ):
        return False
    phrase_floor = min(
        float(heuristic_features.get("field_phrase_coverage", 0.0)),
        float(ranker_features.get("field_phrase_coverage", 0.0)),
    )
    return (
        phrase_floor >= 0.8
        and support_delta >= 0.4
        and structural_delta >= 0.0
        and ranker_score_delta >= 0.06
    )


def _safe_v5_allows_override(
    heuristic_features: Mapping[str, float],
    ranker_features: Mapping[str, float],
    *,
    structural_delta: float,
    ranker_score_delta: float,
) -> bool:
    if _safe_v4_allows_override(
        heuristic_features,
        ranker_features,
        structural_delta=structural_delta,
        ranker_score_delta=ranker_score_delta,
    ):
        return True
    support_delta = (
        structural_support_score_from_features(ranker_features)
        - structural_support_score_from_features(heuristic_features)
    )
    if _has_strong_field_anchor(heuristic_features) and _is_dimension_only_gain(
        heuristic_features,
        ranker_features,
    ):
        return False
    return (
        support_delta >= 0.08
        and structural_delta >= 0.0
        and ranker_score_delta >= 0.0
        and float(ranker_features.get("field_phrase_coverage", 0.0))
        >= float(heuristic_features.get("field_phrase_coverage", 0.0))
        and float(ranker_features.get("leaf_token_overlap", 0.0))
        >= float(heuristic_features.get("leaf_token_overlap", 0.0))
    )


def _has_strong_field_anchor(features: Mapping[str, float]) -> bool:
    return (
        float(features.get("field_alias_overlap", 0.0)) >= 0.8
        and float(features.get("source_column_overlap", 0.0)) > 0.0
        and (
            float(features.get("role_dimension_consistency", 0.0)) > 0.0
            or float(features.get("role_total_consistency", 0.0)) > 0.0
        )
    )


def _is_dimension_only_gain(
    heuristic_features: Mapping[str, float],
    ranker_features: Mapping[str, float],
) -> bool:
    positive_gain_features = [
        "source_column_overlap",
        "same_source_sentence",
        "field_alias_overlap",
        "leaf_token_overlap",
        "query_token_overlap",
        "role_dimension_consistency",
        "role_total_consistency",
    ]
    has_non_dimension_gain = any(
        float(ranker_features.get(feature_name, 0.0))
        > float(heuristic_features.get(feature_name, 0.0))
        for feature_name in positive_gain_features
    )
    return (
        float(ranker_features.get("dimension_value_overlap", 0.0))
        > float(heuristic_features.get("dimension_value_overlap", 0.0))
        and not has_non_dimension_gain
    )


def _has_weak_structural_anchor(features: Mapping[str, float]) -> bool:
    return (
        float(features.get("field_alias_overlap", 0.0)) < 0.5
        and float(features.get("row_label_overlap", 0.0)) < 0.5
    )


class SklearnEvidenceRanker:
    def __init__(
        self,
        model: LogisticRegression | None = None,
        feature_names: Sequence[str] = FEATURE_NAMES,
        heuristic_blend_weight: float = 0.0,
    ) -> None:
        self.model = model or LogisticRegression(
            class_weight="balanced",
            random_state=17,
            max_iter=1000,
        )
        self.feature_names = tuple(feature_names)
        self.heuristic_blend_weight = float(heuristic_blend_weight)

    def fit(self, examples: Sequence[LeafTokenExample]) -> SklearnEvidenceRanker:
        if not examples:
            raise ValueError("SklearnEvidenceRanker.fit requires at least one example.")
        labels = np.asarray([int(example.label) for example in examples], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("SklearnEvidenceRanker.fit requires binary labels {0, 1}.")
        x_rows = vectorize_feature_dicts(
            [example.features for example in examples],
            feature_names=self.feature_names,
        )
        self.model.fit(x_rows, labels, sample_weight=self.sample_weights(examples))
        return self

    def fit_pairwise(self, examples: Sequence[LeafTokenExample]) -> SklearnEvidenceRanker:
        if not examples:
            raise ValueError("SklearnEvidenceRanker.fit_pairwise requires at least one example.")
        x_rows, labels = pairwise_training_rows(
            examples,
            feature_names=self.feature_names,
        )
        if len(labels) == 0:
            raise ValueError("SklearnEvidenceRanker.fit_pairwise requires pairwise comparable examples.")
        self.model.fit(x_rows, labels)
        return self

    @staticmethod
    def sample_weights(examples: Sequence[LeafTokenExample]) -> np.ndarray:
        if not examples:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(
            [
                3.0
                if int(example.label) == 0
                and float(example.features.get("hard_negative", 0.0)) > 0.0
                else 1.0
                for example in examples
            ],
            dtype=np.float32,
        )

    def score(self, examples: Sequence[LeafTokenExample]) -> list[RankedEvidenceToken]:
        if not examples:
            return []
        self._require_fitted("scoring")
        x_rows = vectorize_feature_dicts(
            [example.features for example in examples],
            feature_names=self.feature_names,
        )
        probabilities = self.model.predict_proba(x_rows)[:, 1]
        scored = [
            (
                float(probability)
                + self.heuristic_blend_weight
                * heuristic_score_from_features(example.features),
                example.token_id,
            )
            for probability, example in zip(probabilities, examples, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RankedEvidenceToken(token_id=token_id, score=score, rank=index)
            for index, (score, token_id) in enumerate(scored, start=1)
        ]

    def save(self, path: str) -> None:
        self._require_fitted("saving")
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "heuristic_blend_weight": self.heuristic_blend_weight,
            },
            path,
        )

    def _require_fitted(self, action: str) -> None:
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            raise ValueError(f"SklearnEvidenceRanker must be fit before {action}.") from exc

    @classmethod
    def load(cls, path: str) -> SklearnEvidenceRanker:
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            feature_names=payload["feature_names"],
            heuristic_blend_weight=payload.get("heuristic_blend_weight", 0.0),
        )


def pairwise_training_rows(
    examples: Sequence[LeafTokenExample],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[LeafTokenExample]] = {}
    for index, example in enumerate(examples):
        group_key = example.sample_id or f"__ungrouped_{index}"
        grouped.setdefault(group_key, []).append(example)

    rows: list[list[float]] = []
    labels: list[int] = []
    for group_examples in grouped.values():
        positives = [example for example in group_examples if int(example.label) == 1]
        negatives = [example for example in group_examples if int(example.label) == 0]
        for positive in positives:
            positive_vector = _feature_vector(positive.features, feature_names)
            for negative in negatives:
                negative_vector = _feature_vector(negative.features, feature_names)
                delta = [
                    positive_value - negative_value
                    for positive_value, negative_value in zip(
                        positive_vector,
                        negative_vector,
                        strict=True,
                    )
                ]
                rows.append(delta)
                labels.append(1)
                rows.append([-value for value in delta])
                labels.append(0)
    if not rows:
        return (
            np.zeros((0, len(feature_names)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def _feature_vector(
    features: Mapping[str, float],
    feature_names: Sequence[str],
) -> list[float]:
    return [float(features.get(feature_name, 0.0)) for feature_name in feature_names]


def _terms(value: object | None) -> set[str]:
    if value is None:
        return set()
    normalized = normalize_identifier(str(value))
    return {
        term
        for term in normalized.split("_")
        if term and term not in _STOPWORDS and not term.isdigit()
    }


def _token_terms(token: AttributeValueToken) -> set[str]:
    terms: set[str] = set()
    values: list[object | None] = [
        token.token_id,
        token.entity_id,
        token.company_name,
        token.field_name,
        token.field_label,
        token.industry,
        token.unit,
        token.canonical_concept_id,
        token.raw_label,
        *(token.external_concept_ids or ()),
    ]
    if token.source is not None:
        values.extend(
            [
                token.source.document_id,
                token.source.table,
                token.source.column,
                token.source.text_excerpt,
            ]
        )
    if token.dimensions:
        values.extend(token.dimensions.keys())
        values.extend(token.dimensions.values())
    for value in values:
        terms.update(_terms(value))
    return terms


def _overlap_ratio(reference_terms: set[str], candidate_terms: set[str]) -> float:
    if not reference_terms or not candidate_terms:
        return 0.0
    return len(reference_terms & candidate_terms) / len(reference_terms)


def _is_total_like(token: AttributeValueToken) -> bool:
    if _dimension_text(token, "segment") is None:
        return True
    values: list[object | None] = [
        token.token_id,
        token.entity_id,
        token.company_name,
        token.field_name,
        token.field_label,
        token.raw_label,
    ]
    if token.dimensions:
        values.extend(token.dimensions.values())
    terms: set[str] = set()
    for value in values:
        terms.update(_terms(value))
    return bool(terms & _TOTAL_TERMS)


def _dimension_text(token: AttributeValueToken, key: str) -> str | None:
    if not token.dimensions:
        return None
    value = token.dimensions.get(key)
    if value is None:
        return None
    return str(value)
