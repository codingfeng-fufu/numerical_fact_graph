from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken
from graph_numeric.learning.evidence_ranker import (
    LeafTokenExample,
    SklearnEvidenceRanker,
    apply_safe_arbitration_with_trace,
    extract_leaf_token_features,
    rank_by_heuristic,
)
from graph_numeric.core.expression_plan import EvidenceQuery, parse_expression_plan
from graph_numeric.learning.field_grounder import FieldGrounder
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot

DEFAULT_SAFEV5_RANKER_MODEL = Path(
    "models/graph_numeric/"
    "evidence_ranker_train2000_exprplan11_qualifier1_pairwise_blend03_20260622.joblib"
)


@dataclass(frozen=True)
class EvidenceArbitrationConfig:
    mode: str = "safe_v5"
    policy: str = "safe_v5"
    model_path: Path = DEFAULT_SAFEV5_RANKER_MODEL
    max_candidates_per_leaf: int = 12


def apply_runtime_evidence_arbitration(
    *,
    query: str,
    graph: AttributeValueGraph,
    plan: OperatorPlan | CompositeOperatorPlan | None,
    config: EvidenceArbitrationConfig,
    field_grounder: FieldGrounder,
) -> tuple[OperatorPlan | CompositeOperatorPlan | None, dict[str, Any]]:
    if config.mode == "off":
        return plan, _base_trace(config, enabled=False, status="disabled")
    if plan is None or isinstance(plan, CompositeOperatorPlan):
        return plan, {
            **_base_trace(config, enabled=True, status="not_applicable"),
            "warnings": ["operator_plan_not_supported"],
        }

    expression = parse_expression_plan(query)
    leaves = _collect_leaves(expression)
    if not leaves:
        return plan, {
            **_base_trace(config, enabled=True, status="not_applicable"),
            "warnings": ["expression_plan_missing_or_leafless"],
        }

    ranker, warning = _load_ranker_safely(config.model_path)
    if ranker is None:
        return plan, {
            **_base_trace(config, enabled=True, status="unavailable"),
            "warnings": [warning or "ranker_unavailable"],
        }

    decisions: list[dict[str, Any]] = []
    updated_plan = plan
    for index, leaf in enumerate(leaves):
        examples, token_by_id = _examples_for_leaf(
            query=query,
            leaf=leaf,
            graph=graph,
            max_candidates=config.max_candidates_per_leaf,
        )
        if not examples:
            decisions.append(_leaf_unavailable_trace(index, leaf, "no_candidates"))
            continue

        heuristic_ranked = rank_by_heuristic(examples)
        ranker_ranked = ranker.score(examples)
        arbitration = apply_safe_arbitration_with_trace(
            examples,
            heuristic_ranked,
            ranker_ranked,
            policy=config.policy,
        )
        decision = {
            "leaf_id": f"leaf{index}",
            "role": leaf.role,
            "field_surface": leaf.field_surface,
            "time_surface": leaf.time_surface,
            **arbitration.trace,
        }
        decisions.append(decision)

        if config.mode == "safe_v5" and arbitration.ranked:
            updated_plan = _apply_selected_token_to_plan(
                updated_plan,
                leaf=leaf,
                selected_token=token_by_id.get(arbitration.ranked[0].token_id),
            )

    return updated_plan, {
        **_base_trace(config, enabled=True, status="applied"),
        "decisions": decisions,
        "warnings": [],
    }


def _base_trace(
    config: EvidenceArbitrationConfig,
    *,
    enabled: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "mode": config.mode,
        "policy": config.policy,
        "status": status,
        "model_path": str(config.model_path),
        "decisions": [],
        "warnings": [],
    }


@lru_cache(maxsize=4)
def _load_ranker_cached(path: str) -> SklearnEvidenceRanker:
    return SklearnEvidenceRanker.load(path)


def _load_ranker_safely(path: Path) -> tuple[SklearnEvidenceRanker | None, str | None]:
    try:
        return _load_ranker_cached(str(path)), None
    except Exception as exc:
        return None, f"ranker_load_failed:{exc}"


def _collect_leaves(expression: Any) -> list[EvidenceQuery]:
    if expression is None:
        return []
    leaves: list[EvidenceQuery] = []

    def visit(node: Any) -> None:
        evidence_query = getattr(node, "evidence_query", None)
        if evidence_query is not None:
            leaves.append(evidence_query)
        for child in getattr(node, "children", []) or []:
            visit(child)

    visit(expression.root)
    return leaves


def _examples_for_leaf(
    *,
    query: str,
    leaf: EvidenceQuery,
    graph: AttributeValueGraph,
    max_candidates: int,
) -> tuple[list[LeafTokenExample], dict[str, AttributeValueToken]]:
    scored: list[tuple[float, str, AttributeValueToken, dict[str, float]]] = []
    for token in graph.tokens:
        features = extract_leaf_token_features(query=query, leaf=leaf, token=token)
        score = float(features.get("heuristic_score", 0.0))
        if score <= 0.0:
            continue
        scored.append((score, token.token_id, token, features))
    scored.sort(key=lambda item: (-item[0], item[1]))

    examples: list[LeafTokenExample] = []
    token_by_id: dict[str, AttributeValueToken] = {}
    for _, _, token, features in scored[:max_candidates]:
        token_by_id[token.token_id] = token
        examples.append(
            LeafTokenExample(
                query=query,
                leaf=leaf,
                token_id=token.token_id,
                label=0,
                features=features,
                sample_id=f"runtime:{leaf.role}:{leaf.field_surface}:{leaf.time_surface}",
            )
        )
    return examples, token_by_id


def _leaf_unavailable_trace(
    index: int,
    leaf: EvidenceQuery,
    reason: str,
) -> dict[str, Any]:
    return {
        "leaf_id": f"leaf{index}",
        "role": leaf.role,
        "field_surface": leaf.field_surface,
        "time_surface": leaf.time_surface,
        "decision": "unavailable",
        "reason": reason,
        "heuristic_top": {},
        "ranker_top": {},
        "selected": None,
        "guard_metrics": {},
        "top_candidates": [],
    }


def _apply_selected_token_to_plan(
    plan: OperatorPlan,
    *,
    leaf: EvidenceQuery,
    selected_token: AttributeValueToken | None,
) -> OperatorPlan:
    if selected_token is None or plan.operator != "SHARE":
        return plan

    slots = dict(plan.slots or {})
    role = (leaf.role or "").lower()
    if role in {"part", "numerator", "left", "segment"}:
        slots["numerator_token_ids"] = Slot(
            surface=leaf.field_surface,
            grounded_value=[selected_token.token_id],
            confidence=0.95,
        )
        slots["numerator_field"] = Slot(
            leaf.field_surface,
            selected_token.field_name,
            0.95,
        )
    elif role in {"whole", "denominator", "right", "total"}:
        slots["denominator_token_ids"] = Slot(
            surface=leaf.field_surface,
            grounded_value=[selected_token.token_id],
            confidence=0.95,
        )
        slots["denominator_field"] = Slot(
            leaf.field_surface,
            selected_token.field_name,
            0.95,
        )
    else:
        return plan

    trace = {
        **(plan.trace or {}),
        "runtime_evidence_arbitration_applied": True,
    }
    return dataclasses.replace(plan, slots=slots, trace=trace)
