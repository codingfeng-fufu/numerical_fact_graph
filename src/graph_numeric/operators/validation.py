from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from graph_numeric.core.attribute_graph import AttributeValueGraph
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot
from graph_numeric.operators.operator_registry import OPERATOR_REGISTRY
from graph_numeric.core.unit_resolver import UnitResolver


@dataclass(frozen=True)
class PreconditionReport:
    status: str
    operator: str
    predicates: dict[str, bool] = field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()
    invalid_slots: tuple[str, ...] = ()
    field_not_found: tuple[str, ...] = ()
    records_found: int = 0
    unit_warnings: tuple[str, ...] = ()
    blocking_violations: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "operator": self.operator,
            "predicates": dict(self.predicates),
            "missing_slots": list(self.missing_slots),
            "invalid_slots": list(self.invalid_slots),
            "field_not_found": list(self.field_not_found),
            "records_found": self.records_found,
            "unit_warnings": list(self.unit_warnings),
            "blocking_violations": list(self.blocking_violations),
            "details": self.details,
        }


def validate_preconditions(
    graph: AttributeValueGraph,
    plan: OperatorPlan | CompositeOperatorPlan,
) -> PreconditionReport:
    if isinstance(plan, CompositeOperatorPlan):
        return _validate_composite(graph, plan)
    return _validate_operator(graph, plan)


def _validate_composite(
    graph: AttributeValueGraph,
    plan: CompositeOperatorPlan,
) -> PreconditionReport:
    blocking: list[str] = []
    details: dict[str, Any] = {"steps": []}
    records_found = 0
    if not plan.steps:
        blocking.append("composite_plan_empty")

    seen_step_ids: set[str] = set()
    for index, step in enumerate(plan.steps):
        step_id = _step_id(step, index)
        step_report = _validate_operator(
            graph,
            step,
            dependency_slots=set((step.depends_on or {}).keys()),
        )
        details["steps"].append({"step_id": step_id, **step_report.to_dict()})
        records_found += step_report.records_found
        blocking.extend(f"{step_id}:{item}" for item in step_report.blocking_violations)

        refs = [
            str(value).split(".", 1)[0].lstrip("$")
            for value in (step.depends_on or {}).values()
            if isinstance(value, str) and value.startswith("$")
        ]
        bad_refs = [ref for ref in refs if ref not in seen_step_ids]
        if bad_refs:
            blocking.append(f"{step_id}:dependency_ref_unavailable")
        seen_step_ids.add(step_id)

    status = "ok" if not blocking else "precondition_violation"
    predicates = _composite_predicates(blocking)
    return PreconditionReport(
        status=status,
        operator="COMPOSITE",
        predicates=predicates,
        records_found=records_found,
        blocking_violations=tuple(blocking),
        details=details,
    )


def _validate_operator(
    graph: AttributeValueGraph,
    plan: OperatorPlan,
    *,
    dependency_slots: set[str] | None = None,
) -> PreconditionReport:
    dependency_slots = dependency_slots or set()
    missing = tuple(sorted(plan.missing_required_slots() - dependency_slots))
    invalid = tuple(sorted(_invalid_slots(plan)))
    fields = _field_slot_values(plan)
    graph_fields = set(graph.fields)
    field_not_found = tuple(sorted(field for field in fields if field not in graph_fields))
    records = _estimate_records(graph, plan)
    unit_warnings = tuple(_unit_warnings(graph, plan, fields))
    records_missing = (
        _requires_records(plan.operator)
        and not missing
        and not field_not_found
        and records <= 0
    )
    insufficient_factors = (
        OPERATOR_REGISTRY.canonical_executor_operator(plan.operator) == "PRODUCT"
        and not missing
        and not field_not_found
        and records < 2
    )
    unit_incompatible = _has_cross_currency_warning(unit_warnings)
    predicates = {
        "required_slots": not missing,
        "slot_values": not invalid,
        "field_exists": not field_not_found,
        "records_found": not records_missing,
        "sufficient_factors": not insufficient_factors,
        "unit_compatible": not unit_incompatible,
    }

    blocking: list[str] = []
    if not predicates["required_slots"]:
        blocking.append("missing_required_slots")
    if not predicates["slot_values"]:
        blocking.append("invalid_slots")
    if not predicates["field_exists"]:
        blocking.append("field_not_found")
    if not predicates["records_found"]:
        blocking.append("records_not_found")
    if not predicates["sufficient_factors"]:
        blocking.append("insufficient_factors")
    if not predicates["unit_compatible"]:
        blocking.append("unit_incompatible")

    status = "ok" if not blocking else "precondition_violation"
    return PreconditionReport(
        status=status,
        operator=plan.operator,
        predicates=predicates,
        missing_slots=missing,
        invalid_slots=invalid,
        field_not_found=field_not_found,
        records_found=records,
        unit_warnings=unit_warnings,
        blocking_violations=tuple(blocking),
        details={
            "field_slots": fields,
            "graph_fields": sorted(graph_fields),
            "proposal_token_bindings": _slot_value(plan, "proposal_token_bindings"),
        },
    )


def _invalid_slots(plan: OperatorPlan) -> set[str]:
    invalid: set[str] = set()
    for key, value in (plan.slots or {}).items():
        if isinstance(value, Slot) and value.grounded_value is None:
            invalid.add(key)
        elif isinstance(value, Slot) and value.confidence <= 0:
            invalid.add(key)
    for key in ("k", "horizon", "from_time", "to_time", "year"):
        value = _slot_value(plan, key)
        if value is None:
            continue
        try:
            int(value)
        except (TypeError, ValueError):
            invalid.add(key)
    return invalid


def _field_slot_values(plan: OperatorPlan) -> tuple[str, ...]:
    keys = (
        "target_field",
        "condition_field",
        "numerator_field",
        "denominator_field",
        "left_field",
        "right_field",
    )
    values: list[str] = []
    for key in keys:
        value = _slot_value(plan, key)
        if value is not None and value != "__step_result__":
            values.append(str(value))
    numerator_fields = _slot_value(plan, "numerator_fields")
    if isinstance(numerator_fields, (list, tuple, set, frozenset)):
        values.extend(str(field) for field in numerator_fields)
    target_fields = _slot_value(plan, "target_fields")
    if isinstance(target_fields, (list, tuple, set, frozenset)):
        values.extend(str(field) for field in target_fields)
    factor_fields = _slot_value(plan, "factor_fields")
    if isinstance(factor_fields, (list, tuple, set, frozenset)):
        values.extend(str(field) for field in factor_fields)
    return tuple(dict.fromkeys(values))


def _estimate_records(graph: AttributeValueGraph, plan: OperatorPlan) -> int:
    proposal_token_ids = _proposal_bound_token_ids(plan)
    if proposal_token_ids:
        return len(_tokens_by_ids(graph, proposal_token_ids))
    factor_token_ids = _slot_value(plan, "factor_token_ids")
    if OPERATOR_REGISTRY.canonical_executor_operator(plan.operator) == "PRODUCT" and factor_token_ids:
        return len(_tokens_by_ids(graph, factor_token_ids))
    fields = _field_slot_values(plan)
    if not fields:
        return 0
    year = _maybe_int(_slot_value(plan, "year"))
    industry = _slot_value(plan, "industry")
    entity = _slot_value(plan, "entity")
    entities = _slot_value(plan, "entities")
    if plan.operator in {"RATIO", "SHARE", "MARGIN"}:
        return _estimate_ratio_records(graph, plan, year, industry, entity, entities)
    if plan.operator in {"DIFFERENCE", "COMPARE"} and (
        _slot_value(plan, "left_field") is not None and _slot_value(plan, "right_field") is not None
    ):
        return _estimate_difference_field_records(graph, plan, year, industry, entity, entities)
    total = 0
    for field in fields:
        selected = graph.select(
            field_name=field,
            year=year,
            industry=str(industry) if industry is not None else None,
        )
        selected = _filter_dimensions(selected, _slot_value(plan, "dimension_filters"))
        if plan.operator in {"RATIO", "SHARE", "MARGIN"}:
            if field == _slot_value(plan, "numerator_field"):
                selected = _filter_dimensions(
                    selected,
                    _slot_value(plan, "numerator_dimension_filters"),
                )
            if field == _slot_value(plan, "denominator_field"):
                selected = _filter_dimensions(
                    selected,
                    _slot_value(plan, "denominator_dimension_filters"),
                )
        if entity is not None:
            selected = tuple(t for t in selected if _matches_entity(t, entity))
        if entities is not None:
            selected = _filter_entities(selected, entities)
        total += len(selected)
    return total


def _estimate_difference_field_records(
    graph: AttributeValueGraph,
    plan: OperatorPlan,
    year: int | None,
    industry: object | None,
    entity: object | None,
    entities: object | None,
) -> int:
    total = 0
    for field_key in ("left_field", "right_field"):
        field = _slot_value(plan, field_key)
        if field is None:
            continue
        selected = graph.select(
            field_name=str(field),
            year=year,
            industry=str(industry) if industry is not None else None,
        )
        selected = _filter_dimensions(selected, _slot_value(plan, "dimension_filters"))
        if entity is not None:
            selected = tuple(t for t in selected if _matches_entity(t, entity))
        if entities is not None:
            selected = _filter_entities(selected, entities)
        total += len(selected)
    return total


def _composite_predicates(blocking: list[str]) -> dict[str, bool]:
    return {
        "required_slots": not any(
            "missing_required_slots" in item or item == "composite_plan_empty"
            for item in blocking
        ),
        "slot_values": not any("invalid_slots" in item for item in blocking),
        "field_exists": not any("field_not_found" in item for item in blocking),
        "records_found": not any("records_not_found" in item for item in blocking),
        "sufficient_factors": not any("insufficient_factors" in item for item in blocking),
        "unit_compatible": not any("unit_incompatible" in item for item in blocking),
        "dependencies_available": not any("dependency_ref_unavailable" in item for item in blocking),
    }


def _estimate_ratio_records(
    graph: AttributeValueGraph,
    plan: OperatorPlan,
    year: int | None,
    industry: object | None,
    entity: object | None,
    entities: object | None,
) -> int:
    numerator_token_ids = _slot_value(plan, "numerator_token_ids")
    denominator_token_ids = _slot_value(plan, "denominator_token_ids")
    if numerator_token_ids or denominator_token_ids:
        return (
            len(_tokens_by_ids(graph, numerator_token_ids))
            + len(_tokens_by_ids(graph, denominator_token_ids))
        )
    total = 0
    numerator_fields = _slot_value(plan, "numerator_fields")
    for field_key, dimension_key in (
        ("numerator_field", "numerator_dimension_filters"),
        ("denominator_field", "denominator_dimension_filters"),
    ):
        field = _slot_value(plan, field_key)
        fields = [field]
        if field_key == "numerator_field" and isinstance(numerator_fields, (list, tuple, set, frozenset)):
            fields = list(numerator_fields)
        if not fields or fields == [None]:
            continue
        selected = tuple(
            token
            for selected_field in fields
            for token in graph.select(
                field_name=str(selected_field),
                year=year,
                industry=str(industry) if industry is not None else None,
            )
        )
        selected = _filter_dimensions(selected, _slot_value(plan, "dimension_filters"))
        selected = _filter_dimensions(selected, _slot_value(plan, dimension_key))
        if entity is not None:
            selected = tuple(t for t in selected if _matches_entity(t, entity))
        if entities is not None:
            selected = _filter_entities(selected, entities)
        total += len(selected)
    return total


def _tokens_by_ids(
    graph: AttributeValueGraph,
    token_ids: object | None,
) -> tuple[object, ...]:
    if token_ids is None:
        return ()
    if isinstance(token_ids, str):
        wanted = {token_ids}
    elif isinstance(token_ids, (list, tuple, set, frozenset)):
        wanted = {str(token_id) for token_id in token_ids}
    else:
        return ()
    return tuple(token for token in graph.tokens if token.token_id in wanted)


def _unit_warnings(
    graph: AttributeValueGraph,
    plan: OperatorPlan,
    fields: tuple[str, ...],
) -> list[str]:
    proposal_token_ids = _proposal_bound_token_ids(plan)
    if proposal_token_ids:
        tokens = _tokens_by_ids(graph, proposal_token_ids)
        units = [UnitResolver().detect(str(token.unit)) for token in tokens if token.unit]
        canonical = OPERATOR_REGISTRY.canonical_executor_operator(plan.operator)
        return UnitResolver().check_compatibility(canonical, units)
    if not fields:
        return []
    year = _maybe_int(_slot_value(plan, "year"))
    industry = _slot_value(plan, "industry")
    tokens = []
    for field in fields:
        tokens.extend(
            graph.select(
                field_name=field,
                year=year,
                industry=str(industry) if industry is not None else None,
            )
        )
    units = [UnitResolver().detect(str(token.unit)) for token in tokens if token.unit]
    if not units:
        plan_unit = _slot_value(plan, "unit")
        if plan_unit is not None:
            units = [UnitResolver().detect(str(plan_unit))]
    canonical = OPERATOR_REGISTRY.canonical_executor_operator(plan.operator)
    return UnitResolver().check_compatibility(canonical, units)


def _proposal_bound_token_ids(plan: OperatorPlan) -> list[str]:
    bindings = _slot_value(plan, "proposal_token_bindings")
    if not isinstance(bindings, Mapping):
        return []
    return list(
        dict.fromkeys(
            str(token_id)
            for token_ids in bindings.values()
            if isinstance(token_ids, (list, tuple, set, frozenset))
            for token_id in token_ids
        )
    )


def _requires_records(operator: str) -> bool:
    return OPERATOR_REGISTRY.canonical_executor_operator(operator) in {
        "SUM",
        "COUNT",
        "AVG",
        "MAX",
        "MIN",
        "ARGMAX",
        "ARGMIN",
        "RATIO",
        "GROWTH",
        "DIFFERENCE",
        "LOOKUP",
        "YEAR_LIST",
        "TOP_K",
        "TREND",
        "PREDICT",
        "PRODUCT",
    }


def _has_cross_currency_warning(warnings: tuple[str, ...]) -> bool:
    return any("Cross-currency" in warning or "Mismatched money currencies" in warning for warning in warnings)


def _slot_value(plan: OperatorPlan, key: str) -> object | None:
    value = (plan.slots or {}).get(key)
    if isinstance(value, Slot):
        return value.grounded_value
    return value


def _maybe_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _matches_entity(token: object, entity: object) -> bool:
    entity_str = str(entity)
    company = str(getattr(token, "company_name", ""))
    entity_id = str(getattr(token, "entity_id", ""))
    return (
        entity_id == entity_str
        or company == entity_str
        or _entity_text_matches(company, entity_str)
        or entity_id.startswith(f"{entity_str}:")
    )


def _entity_text_matches(company: str, entity: str) -> bool:
    company_terms = _entity_terms(company)
    entity_terms = _entity_terms(entity)
    if not company_terms or not entity_terms:
        return False
    if len(entity_terms) == 1:
        return next(iter(entity_terms)) in company_terms
    return entity_terms.issubset(company_terms)


def _entity_terms(value: str) -> set[str]:
    import re

    suffixes = {
        "and",
        "the",
        "for",
        "from",
        "with",
        "year",
        "years",
        "total",
        "net",
        "sales",
        "revenue",
        "income",
        "operating",
        "gross",
        "research",
        "development",
        "cash",
        "equivalents",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "company",
        "co",
        "ltd",
        "limited",
        "plc",
        "com",
        "class",
        "common",
        "stock",
    }
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if (len(term) >= 3 or term.isdigit()) and term not in suffixes
    }


def _filter_entities(tokens: tuple[object, ...], entities: object) -> tuple[object, ...]:
    if isinstance(entities, str):
        entity_values = [entities]
    elif isinstance(entities, (list, tuple, set, frozenset)):
        entity_values = list(entities)
    else:
        entity_values = [entities]
    return tuple(token for token in tokens if any(_matches_entity(token, entity) for entity in entity_values))


def _filter_dimensions(tokens: tuple[object, ...], dimensions: object | None) -> tuple[object, ...]:
    if not dimensions or not isinstance(dimensions, Mapping):
        return tokens
    selected = tokens
    for key, value in dimensions.items():
        if not any(str(key) in (getattr(token, "dimensions", None) or {}) for token in selected):
            continue
        selected = tuple(
            token
            for token in selected
            if _dimension_value_matches(
                (getattr(token, "dimensions", None) or {}).get(str(key)),
                value,
            )
        )
    return selected


def _dimension_value_matches(actual: object | None, expected: object | None) -> bool:
    if actual is None or expected is None:
        return False
    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    actual_terms = _entity_terms(actual_text)
    expected_terms = _entity_terms(expected_text)
    if not actual_terms or not expected_terms:
        return False
    return expected_terms.issubset(actual_terms) or actual_terms.issubset(expected_terms)


def _step_id(step: OperatorPlan, index: int) -> str:
    value = (step.trace or {}).get("step_id") if step.trace else None
    return str(value) if value else f"s{index + 1}"
