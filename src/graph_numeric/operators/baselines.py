from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph, field_aliases, normalize_identifier
from graph_numeric.operators.executor import execute
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan
from graph_numeric.pipeline.pipeline import run_operator_pipeline
from graph_numeric.learning.router import RoutingResult


SCALAR_SLOT_KEYS = {
    "year",
    "industry",
    "entity",
    "entities",
    "left_entity",
    "right_entity",
    "selected_token_ids",
}


@dataclass(frozen=True)
class BaselineResult:
    strategy: str
    answer: float | None
    operator: str | None
    selected_field: str | None
    selected_entity: str | None
    selected_token_ids: tuple[str, ...]
    evidence_complete: bool
    verification_passed: bool
    metadata: dict[str, Any]


def run_vector_only_baseline(query: str, graph: AttributeValueGraph) -> BaselineResult:
    """Semantic grounding path with scalar slots removed before execution."""
    try:
        grounded = run_operator_pipeline(query, graph)
        if grounded.plan is None or isinstance(grounded.plan, CompositeOperatorPlan):
            raise ValueError("vector-only baseline only supports single-step plans")
        plan = _strip_scalar_slots(grounded.plan)
        result = execute(graph, plan)
        selected_entities = _selected_entities(result.selected_tokens)
        removed = tuple(
            key for key in (grounded.plan.slots or {}) if key in SCALAR_SLOT_KEYS
        )
        return BaselineResult(
            strategy="vector_only",
            answer=float(result.answer),
            operator=plan.operator,
            selected_field=_first_plan_field(plan),
            selected_entity=selected_entities[0] if len(selected_entities) == 1 else None,
            selected_token_ids=tuple(token.token_id for token in result.selected_tokens),
            evidence_complete=bool(result.metadata.get("evidence_trace_complete")),
            verification_passed=bool(result.checks) and all(result.checks.values()),
            metadata={
                "removed_scalar_slots": sorted(removed),
                "source_pipeline_status": grounded.status,
                "source_strategy": _source_strategy(grounded.to_dict()),
                "records_used": list(result.metadata.get("records_used", [])),
            },
        )
    except Exception as exc:
        return _failed_result("vector_only", str(exc))


def run_scalar_only_baseline(query: str, graph: AttributeValueGraph) -> BaselineResult:
    """Exact canonical field matching with limited alias support by design."""
    exact_fields = _exact_fields(query, graph)
    if not exact_fields:
        return _failed_result(
            "scalar_only",
            "no exact canonical field mention",
        )
    operator = _scalar_operator(query, exact_fields)
    if operator is None:
        return _failed_result(
            "scalar_only",
            "no scalar-only operator pattern",
            selected_field=exact_fields[0],
        )
    routing = RoutingResult(
        operator=operator,
        confidence=0.6,
        intent="scalar_exact_match",
        route_type="scalar_only",
        fallback_operators=[],
        trace={
            "hybrid_strategy": "scalar_first",
            "exact_fields": list(exact_fields),
        },
    )
    try:
        pipeline = run_operator_pipeline(
            query,
            graph,
            routing=routing,
            enable_fallback=False,
        )
        return _result_from_pipeline(
            strategy="scalar_only",
            pipeline_dict=pipeline.to_dict(),
            plan=pipeline.plan,
        )
    except Exception as exc:
        return _failed_result("scalar_only", str(exc), selected_field=exact_fields[0])


def run_retrieval_calculator_baseline(query: str, graph: AttributeValueGraph) -> BaselineResult:
    """Ground field semantically, then run deterministic arithmetic without scalar filters."""
    try:
        grounded = run_operator_pipeline(query, graph)
        if grounded.plan is None or isinstance(grounded.plan, CompositeOperatorPlan):
            raise ValueError("retrieval-calculator baseline only supports single-step plans")
        plan = _strip_scalar_slots(grounded.plan, keep_entity=False)
        result = execute(graph, plan)
        selected_entities = _selected_entities(result.selected_tokens)
        return BaselineResult(
            strategy="retrieval_calculator",
            answer=float(result.answer),
            operator=plan.operator,
            selected_field=_first_plan_field(plan),
            selected_entity=selected_entities[0] if len(selected_entities) == 1 else None,
            selected_token_ids=tuple(token.token_id for token in result.selected_tokens),
            evidence_complete=bool(result.metadata.get("evidence_trace_complete")),
            verification_passed=False,
            metadata={
                "scalar_filters_enforced": False,
                "deterministic_arithmetic": True,
                "source_pipeline_status": grounded.status,
                "records_used": list(result.metadata.get("records_used", [])),
            },
        )
    except Exception as exc:
        return _failed_result("retrieval_calculator", str(exc))


def run_plan_without_verifier_baseline(query: str, graph: AttributeValueGraph) -> BaselineResult:
    """Use full planning/execution but report verifier as advisory, not enforced."""
    try:
        pipeline = run_operator_pipeline(query, graph)
        payload = pipeline.to_dict()
        result = _result_from_pipeline(
            strategy="plan_without_verifier",
            pipeline_dict=payload,
            plan=pipeline.plan,
        )
        return dataclasses.replace(
            result,
            verification_passed=False,
            metadata={
                **result.metadata,
                "verifier_enforced": False,
                "verifier_advisory_passed": bool(
                    (payload.get("verification_report") or {}).get("passed")
                ),
            },
        )
    except Exception as exc:
        return _failed_result("plan_without_verifier", str(exc))


def _result_from_pipeline(
    *,
    strategy: str,
    pipeline_dict: dict[str, Any],
    plan: OperatorPlan | CompositeOperatorPlan | None,
) -> BaselineResult:
    metadata = pipeline_dict.get("execution_metadata") or {}
    records = metadata.get("records_used") or []
    entities = _selected_entities_from_records(records)
    verification = pipeline_dict.get("verification_report") or {}
    return BaselineResult(
        strategy=strategy,
        answer=(
            float(pipeline_dict["answer"])
            if isinstance(pipeline_dict.get("answer"), (int, float))
            else None
        ),
        operator=(
            str(pipeline_dict["selected_operator"])
            if pipeline_dict.get("selected_operator") is not None
            else None
        ),
        selected_field=_first_plan_field(plan),
        selected_entity=entities[0] if len(entities) == 1 else None,
        selected_token_ids=tuple(str(token_id) for token_id in metadata.get("selected_token_ids") or ()),
        evidence_complete=bool(metadata.get("evidence_trace_complete")),
        verification_passed=bool(verification.get("passed")),
        metadata={
            "status": pipeline_dict.get("status"),
            "records_used": records,
            "failure_categories": verification.get("failure_categories") or [],
        },
    )


def _strip_scalar_slots(
    plan: OperatorPlan,
    *,
    keep_entity: bool = False,
) -> OperatorPlan:
    removed_keys = {
        key for key in SCALAR_SLOT_KEYS if not (keep_entity and key in {"entity", "entities"})
    }
    return dataclasses.replace(
        plan,
        slots={
            key: value
            for key, value in (plan.slots or {}).items()
            if key not in removed_keys
        },
        trace={
            **(plan.trace or {}),
            "strategy_mutation": "removed_scalar_constraints",
            "removed_scalar_slots": sorted(
                key for key in (plan.slots or {}) if key in removed_keys
            ),
        },
    )


def _exact_fields(query: str, graph: AttributeValueGraph) -> tuple[str, ...]:
    query_lower = query.lower()
    query_normalized = normalize_identifier(query)
    fields = []
    for field in graph.fields:
        if field in query_lower or field.replace("_", " ") in query_lower or field in query_normalized:
            fields.append(field)
    return tuple(sorted(fields))


def _scalar_operator(query: str, exact_fields: tuple[str, ...]) -> str | None:
    normalized = normalize_identifier(query)
    if "count" in normalized or "how_many" in normalized or "多少" in query:
        return "COUNT"
    if "margin" in normalized or "利润率" in query:
        return "MARGIN" if "net_profit" in exact_fields else None
    if "growth" in normalized or "增长" in query:
        return "GROWTH"
    if "trend" in normalized or "趋势" in query:
        return "TREND"
    if "difference" in normalized or "gap" in normalized or "差" in query:
        return "DIFFERENCE"
    if "ratio" in normalized or "比率" in query:
        return "RATIO"
    if "lowest" in normalized or "smallest" in normalized or "minimum" in normalized or "最低" in query:
        return "ARGMIN"
    if "highest" in normalized or "largest" in normalized or "top" in normalized or "maximum" in normalized or "最高" in query:
        return "ARGMAX"
    if "what_is" in normalized or "lookup" in normalized or "是多少" in query:
        return "LOOKUP"
    if "total" in normalized or "sum" in normalized or "reported" in normalized or "总和" in query:
        return "SUM"
    return None


def _first_plan_field(plan: OperatorPlan | CompositeOperatorPlan | None) -> str | None:
    fields = _plan_fields(plan)
    return fields[0] if fields else None


def _plan_fields(plan: OperatorPlan | CompositeOperatorPlan | None) -> list[str]:
    if isinstance(plan, CompositeOperatorPlan):
        fields: list[str] = []
        for step in plan.steps:
            fields.extend(_plan_fields(step))
        return sorted(set(fields))
    if not isinstance(plan, OperatorPlan):
        return []
    fields = []
    for key, value in (plan.slots or {}).items():
        if key.endswith("_field") or key in {"target_field", "condition_field"}:
            grounded = getattr(value, "grounded_value", value)
            if grounded is not None:
                fields.append(str(grounded))
    return sorted(set(fields))


def _selected_entities(tokens: tuple[Any, ...]) -> list[str]:
    entities: list[str] = []
    for token in tokens:
        company = getattr(token, "company_name", None)
        if company is None:
            continue
        company_text = str(company)
        if company_text not in entities:
            entities.append(company_text)
    return entities


def _selected_entities_from_records(records: list[dict[str, Any]]) -> list[str]:
    entities: list[str] = []
    for record in records:
        company = record.get("company_name")
        if company is None:
            continue
        company_text = str(company)
        if company_text not in entities:
            entities.append(company_text)
    return entities


def _source_strategy(payload: dict[str, Any]) -> str | None:
    selector = (
        payload.get("routing", {})
        .get("trace", {})
        .get("strategy_selector", {})
    )
    if isinstance(selector, dict) and selector.get("strategy"):
        return str(selector["strategy"])
    return None


def _failed_result(
    strategy: str,
    error: str,
    *,
    selected_field: str | None = None,
) -> BaselineResult:
    return BaselineResult(
        strategy=strategy,
        answer=None,
        operator=None,
        selected_field=selected_field,
        selected_entity=None,
        selected_token_ids=(),
        evidence_complete=False,
        verification_passed=False,
        metadata={"error": error},
    )
