from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Sequence

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, field_aliases, normalize_identifier
from graph_numeric.learning.field_grounder import FieldGrounder, FieldGroundingResult
from graph_numeric.learning.forecasting import DEFAULT_FORECAST_SELECTOR_CONFIG, forecast_profile
from graph_numeric.operators.operator_registry import PUBLIC_OPERATORS
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot
from graph_numeric.core.query_cleaning import strip_negative_distractors
from graph_numeric.learning.router import RoutingResult
from graph_numeric.core.unit_resolver import UnitResolver


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
FY_RE = re.compile(r"\b(?:fy|fiscal\s+year|fiscal)\s*'?(\d{2})\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

CONDITION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r">=|不低于|大于等于|at\s+or\s+above|at\s+least|no\s+less\s+than", re.IGNORECASE), ">="),
    (re.compile(r"<=|不高于|小于等于|at\s+or\s+below|at\s+most|no\s+more\s+than", re.IGNORECASE), "<="),
    (re.compile(r"(?<![<>!])>(?!=)|超过|大于|exceeds?|greater\s+than|above|over|cleared", re.IGNORECASE), ">"),
    (re.compile(r"(?<![<>!])<(?!=)|小于|less\s+than|below|under", re.IGNORECASE), "<"),
    (re.compile(r"==|等于|equals?|equal\s+to", re.IGNORECASE), "=="),
]

SUPPORTED_SOLVER_OPERATORS = set(PUBLIC_OPERATORS)


class OperatorSolver:
    """Build grounded OperatorPlan objects from routed queries.

    The solver layer owns slot grounding. Routers choose operators, executors
    only run structured plans.
    """

    def __init__(
        self,
        *,
        field_grounder: FieldGrounder | None = None,
        unit_resolver: UnitResolver | None = None,
    ) -> None:
        self.field_grounder = field_grounder or FieldGrounder.from_lexical()
        self.unit_resolver = unit_resolver or UnitResolver()

    def solve(
        self,
        query: str,
        graph: AttributeValueGraph,
        routing: RoutingResult,
        *,
        operator: str | None = None,
    ) -> OperatorPlan:
        selected_operator = operator or routing.operator
        effective_routing = (
            replace(routing, operator=selected_operator, confidence=1.0, route_type="oracle_operator")
            if operator is not None and operator != routing.operator
            else routing
        )

        base = self._base_slots(query, graph, selected_operator)
        if selected_operator in ("SUM", "AVG", "MAX", "MIN"):
            slots = self._solve_single_field(query, graph, base)
        elif selected_operator == "PRODUCT":
            slots = self._solve_product(query, graph, base)
        elif selected_operator == "COUNT":
            slots = self._solve_count(query, graph, base)
        elif selected_operator in ("RATIO", "SHARE", "MARGIN"):
            slots = self._solve_ratio(query, graph, base, selected_operator)
        elif selected_operator in ("GROWTH", "PERCENT_CHANGE"):
            slots = self._solve_growth(query, graph, base)
        elif selected_operator in ("ARGMAX", "ARGMIN"):
            asks_for_year = _asks_for_year_output(query)
            asks_for_entity = (not asks_for_year) and _asks_for_entity_output(query)
            entity_type = "year" if asks_for_year else ("entity" if asks_for_entity else "company")
            slots = {
                **base,
                "target_entity_type": _slot(
                    entity_type,
                    entity_type,
                    0.9,
                ),
                "target_field": self._field_slot(query, graph),
            }
            if asks_for_year:
                slots["output_slot"] = _slot("year", "year", 0.95)
            elif asks_for_entity:
                slots["output_slot"] = _slot("entity", "entity", 0.95)
            else:
                entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
                if entity is not None:
                    slots["entity"] = _slot(entity, entity, 0.85)
                entities = resolve_entity_group(query, graph, _slot_grounded(base, "industry"))
                if entities:
                    slots["entities"] = entities
        elif selected_operator == "LOOKUP":
            slots = self._solve_lookup(query, graph, base)
        elif selected_operator == "YEAR_LIST":
            slots = self._solve_year_list(query, graph, base)
        elif selected_operator in ("DIFFERENCE", "COMPARE"):
            slots = self._solve_difference(query, graph, base, selected_operator)
        elif selected_operator in ("TOP_K", "RANK"):
            slots = self._solve_top_k(query, graph, base)
        elif selected_operator == "TREND":
            slots = self._solve_trend(query, graph, base)
        elif selected_operator in ("PREDICT", "FORECAST"):
            slots = self._solve_predict(query, graph, base)
        else:
            slots = self._solve_compat(selected_operator, query, graph, base)

        return OperatorPlan(
            operator=selected_operator,
            slots=slots,
            confidence=effective_routing.confidence,
            trace={
                "routing": effective_routing.to_dict(),
                "solver": "operator_solvers.v2",
            },
        )

    def _base_slots(
        self,
        query: str,
        graph: AttributeValueGraph,
        operator: str | None = None,
    ) -> dict[str, Slot]:
        slots: dict[str, Slot] = {}
        year = resolve_year_for_operator(query, operator)
        if year is not None and _year_is_dimension_value(query, graph, year):
            year = None
        industry = resolve_industry(query, graph)
        if year is not None:
            slots["year"] = _slot(str(year), year, 0.99)
        if industry is not None:
            slots["industry"] = _slot(industry, industry, 0.95)
        unit = extract_requested_unit(query)
        if unit is not None:
            slots["unit"] = _slot(unit, unit, 0.85)
        dimensions = resolve_dimension_filters(query, graph)
        if dimensions:
            slots["dimension_filters"] = _slot(str(dimensions), dimensions, 0.85)
        return slots

    def _field_slot(self, query: str, graph: AttributeValueGraph) -> Slot:
        clean_query = strip_negative_distractors(query)
        review_field = resolve_review_hardening_field(clean_query, graph.fields)
        if review_field is not None:
            return _slot(query, review_field, 0.92)
        grounding = self.field_grounder.ground(clean_query, graph.fields)
        return _slot(query, grounding.field_name, grounding.confidence)

    def _solve_single_field(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        slots = {**base, "target_field": self._field_slot(query, graph)}
        parallel_fields = resolve_explicit_parallel_sum_fields(query, graph.fields)
        if parallel_fields:
            slots["target_field"] = _slot(query, parallel_fields[0], 0.95)
            slots["target_fields"] = _slot(query, parallel_fields, 0.95)
        entities = resolve_entity_group(query, graph, _slot_grounded(base, "industry"))
        if entities:
            slots["entities"] = entities
        return slots

    def _solve_count(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        condition = extract_condition(query)
        op, threshold, condition_conf, condition_unit = (
            condition if condition is not None else (">=", 1000.0, 0.3, None)
        )
        slots = {
            **base,
            "count_target": _slot("company", "company", 0.9),
            "condition_field": self._field_slot(query, graph),
            "condition_op": _slot(op, op, 0.9 if condition is not None else 0.3),
            "condition_threshold": _slot(str(threshold), threshold, condition_conf),
        }
        if condition_unit is not None:
            slots["condition_unit"] = _slot(condition_unit, condition_unit, 0.85)
        entities = resolve_entity_group(query, graph, _slot_grounded(base, "industry"))
        if entities:
            slots["entities"] = entities
        return slots

    def _solve_product(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        product_base = {key: value for key, value in base.items() if key not in {"year", "entity"}}
        factor_tokens = _resolve_product_factor_tokens(query, graph)
        constant_slots = _drop_constants_already_represented_by_tokens(
            resolve_question_constant_slots(query),
            factor_tokens,
        )
        slots = {**product_base}
        if factor_tokens:
            slots["factor_token_ids"] = _slot(
                "product factors",
                [token.token_id for token in factor_tokens],
                0.85,
            )
            slots["factor_fields"] = _slot(
                "product factor fields",
                [token.field_name for token in factor_tokens],
                0.75,
            )
        else:
            slots["target_field"] = self._field_slot(query, graph)
        if constant_slots:
            slots["constant_slot"] = _slot("question_text constants", constant_slots, 0.9)
        return slots

    def _solve_ratio(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
        operator: str = "RATIO",
    ) -> dict[str, Slot]:
        if operator == "SHARE":
            numerator, denominator, num_conf, den_conf = resolve_share_fields(
                query, graph.fields, self.field_grounder
            )
        elif operator == "MARGIN":
            numerator, denominator, num_conf, den_conf = resolve_margin_fields(
                query, graph.fields, self.field_grounder
            )
        else:
            numerator, denominator, num_conf, den_conf = resolve_ratio_fields(
                query,
                graph.fields,
                self.field_grounder,
            )
        share_dimensions = (
            resolve_share_of_total_dimensions(query, graph) if operator == "SHARE" else None
        )
        explicit_total_row_dimensions = (
            resolve_explicit_total_row_dimensions(query, graph) if operator == "SHARE" else None
        )
        if share_dimensions is not None:
            numerator_dimension, denominator_dimension = share_dimensions
            if not _is_total_like_field(denominator):
                denominator = numerator
                den_conf = max(den_conf, num_conf, 0.9)
        elif explicit_total_row_dimensions is not None:
            numerator_dimension = None
            denominator_dimension = explicit_total_row_dimensions
        slots = {
            **base,
            "numerator_field": _slot(query, numerator, num_conf),
            "denominator_field": _slot(query, denominator, den_conf),
        }
        numerator_fields = resolve_multi_numerator_fields(query, graph.fields, numerator, denominator)
        if operator in {"RATIO", "SHARE"} and numerator_fields is not None:
            slots["numerator_fields"] = numerator_fields
        aggregate_token_ids = (
            resolve_aggregate_share_token_ids(
                query,
                graph,
                numerator,
                denominator,
                year=_slot_grounded(base, "year"),
                industry=_slot_grounded(base, "industry"),
            )
            if operator in {"RATIO", "SHARE"} and numerator_fields is None
            else None
        )
        if aggregate_token_ids is not None:
            numerator_token_ids, denominator_token_ids = aggregate_token_ids
            slots["numerator_token_ids"] = _slot(
                "same table aggregate share numerator",
                numerator_token_ids,
                0.92,
            )
            slots["denominator_token_ids"] = _slot(
                "same table aggregate share denominator",
                denominator_token_ids,
                0.92,
            )
            slots.pop("dimension_filters", None)
            slots.pop("industry", None)
            share_dimensions = None
            explicit_total_row_dimensions = None
        elif operator == "SHARE" and numerator == denominator:
            same_field_token_ids = resolve_same_field_share_token_ids(
                query,
                graph,
                numerator,
                year=_slot_grounded(base, "year"),
                industry=_slot_grounded(base, "industry"),
            )
            if same_field_token_ids is not None:
                numerator_token_ids, denominator_token_ids = same_field_token_ids
                slots["numerator_token_ids"] = _slot(
                    "same field share numerator",
                    numerator_token_ids,
                    0.9,
                )
                slots["denominator_token_ids"] = _slot(
                    "same field share denominator",
                    denominator_token_ids,
                    0.9,
                )
                slots.pop("dimension_filters", None)
                slots.pop("industry", None)
        ratio_fields = numerator_fields or [numerator, denominator]
        if not _dimension_filter_keeps_fields(graph, slots, ratio_fields):
            slots.pop("dimension_filters", None)
        if operator == "SHARE" and _should_use_next_total_row(query, denominator, share_dimensions):
            slots["denominator_row_selector"] = _slot("next_total_after_numerator", "next_total_after_numerator", 0.8)
        if share_dimensions is not None or explicit_total_row_dimensions is not None:
            slots.pop("dimension_filters", None)
            if numerator_dimension is not None:
                slots["numerator_dimension_filters"] = _slot(
                    str(numerator_dimension),
                    numerator_dimension,
                    0.9,
                )
            if denominator_dimension is not None:
                slots["denominator_dimension_filters"] = _slot(
                    str(denominator_dimension),
                    denominator_dimension,
                    0.9,
                )
        entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
        has_role_token_binding = (
            "numerator_token_ids" in slots and "denominator_token_ids" in slots
        )
        if entity is not None and not has_role_token_binding:
            slots["entity"] = _slot(entity, entity, 0.85)
        return slots

    def _solve_growth(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        fiscal_periods = resolve_fiscal_period_end_years(query)
        years = resolve_years(query)
        if len(fiscal_periods) >= 2:
            if re.search(r"\bfrom\b", query, re.IGNORECASE):
                to_year, from_year = fiscal_periods[0], fiscal_periods[-1]
            else:
                from_year, to_year = fiscal_periods[0], fiscal_periods[-1]
            from_conf = to_conf = 0.9
        elif len(years) >= 2:
            from_year, to_year = years[0], years[-1]
            from_conf = to_conf = 0.85
        elif len(years) == 1:
            to_year = years[0]
            from_year = to_year - 1
            from_conf, to_conf = 0.45, 0.85
        else:
            to_year = graph.years[-1] if graph.years else 2024
            from_year = to_year - 1
            from_conf = to_conf = 0.4
        growth_base = {key: value for key, value in base.items() if key != "year"}
        slots = {
            **growth_base,
            "target_field": self._field_slot(query, graph),
            "from_time": _slot(str(from_year), from_year, from_conf),
            "to_time": _slot(str(to_year), to_year, to_conf),
        }
        entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
        if entity is not None and str(entity) not in {str(from_year), str(to_year)}:
            slots["entity"] = _slot(entity, entity, 0.85)
        return slots

    def _solve_lookup(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
        return {
            **base,
            "entity": _slot(entity or "", entity, 0.85 if entity else 0.0),
            "target_field": self._field_slot(query, graph),
        }

    def _solve_year_list(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        year_list_base = {key: value for key, value in base.items() if key != "year"}
        return {
            **year_list_base,
            "target_field": self._field_slot(query, graph),
        }

    def _solve_difference(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
        operator: str = "DIFFERENCE",
    ) -> dict[str, Slot]:
        years = resolve_years(query)
        difference_base = {key: value for key, value in base.items() if key != "year"} if len(years) >= 2 else base
        field_pair = resolve_difference_field_pair(query, graph.fields)
        target_field_slot = (
            _slot(query, field_pair[0], 0.95)
            if field_pair is not None
            else self._field_slot(query, graph)
        )
        slots = {**difference_base, "target_field": target_field_slot}
        if field_pair is not None:
            left_field, right_field = field_pair
            slots["left_field"] = _slot(left_field, left_field, 0.95)
            slots["right_field"] = _slot(right_field, right_field, 0.95)
            if not _dimension_filter_keeps_field_pair(graph, slots, left_field, right_field):
                slots.pop("dimension_filters", None)
        if len(years) >= 2:
            from_year, to_year = resolve_change_direction_years(query) or (years[0], years[-1])
            slots["left_time"] = _slot(str(to_year), to_year, 0.9)
            slots["right_time"] = _slot(str(from_year), from_year, 0.9)
        left, right = (
            (None, None)
            if len(years) >= 2
            else resolve_two_entities(query, graph, _slot_grounded(base, "industry"))
        )
        if left is not None:
            slots["left_entity"] = _slot(left, left, 0.85)
        if right is not None:
            slots["right_entity"] = _slot(right, right, 0.85)
        if field_pair is None:
            left_dimension, right_dimension = resolve_two_dimension_values(query, graph)
            if left_dimension is not None and right_dimension is not None:
                slots["left_dimension_filter"] = _slot(str(left_dimension), left_dimension, 0.9)
                slots["right_dimension_filter"] = _slot(str(right_dimension), right_dimension, 0.9)
        entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
        time_entities = {str(year) for year in years}
        if entity is not None and str(entity) not in time_entities and not (left is not None and right is not None):
            slots["entity"] = _slot(entity, entity, 0.85)
        if operator == "COMPARE":
            slots["comparison_type"] = _slot("difference", "difference", 0.6)
        return slots

    def _solve_top_k(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        order = extract_order(query)
        k = extract_k(query)
        return {
            **base,
            "target_entity_type": _slot("company", "company", 0.9),
            "target_field": self._field_slot(query, graph),
            "k": _slot(str(k), k, 0.9),
            "order": _slot(order, order, 0.9),
        }

    def _solve_trend(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        years = resolve_years(query)
        anchor_year = graph.years[-1] if graph.years else 2024
        if len(years) >= 2:
            from_year, to_year = years[0], years[-1]
            from_conf = to_conf = 0.85
        else:
            from_year, to_year = anchor_year - 3, anchor_year
            from_conf = to_conf = 0.45
        trend_base = {key: value for key, value in base.items() if key != "year"}
        slots = {
            **trend_base,
            "target_field": self._field_slot(query, graph),
            "from_time": _slot(str(from_year), from_year, from_conf),
            "to_time": _slot(str(to_year), to_year, to_conf),
        }
        if re.search(r"\btrend\b", query, re.IGNORECASE) and len(years) >= 2:
            slots["trend_metric"] = _slot("annual_delta", "annual_delta", 0.8)
        entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
        if entity is not None:
            slots["entity"] = _slot(entity, entity, 0.85)
        return slots

    def _solve_predict(
        self,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        horizon = resolve_last_year(query)
        if horizon is None:
            horizon = (graph.years[-1] + 1) if graph.years else 2025
            horizon_conf = 0.4
        else:
            horizon_conf = 0.85

        history_years = [year for year in graph.years if year < horizon]
        if len(history_years) >= 2:
            start_year = history_years[0] if len(history_years) <= 4 else history_years[-4]
            end_year = history_years[-1]
            history_conf = 0.8
        else:
            start_year = horizon - 4
            end_year = horizon - 1
            history_conf = 0.4

        entity = resolve_entity(query, graph, _slot_grounded(base, "industry"))
        predict_base = {key: value for key, value in base.items() if key != "year"}
        return {
            **predict_base,
            "target_field": self._field_slot(query, graph),
            "entity": _slot(entity or "", entity, 0.85 if entity else 0.0),
            "history_range": _slot(
                f"{start_year}-{end_year}",
                f"{start_year}-{end_year}",
                history_conf,
            ),
            "horizon": _slot(str(horizon), horizon, horizon_conf),
            "method": _slot("selector", "selector", 0.7),
            "selector_config": _slot(
                str(DEFAULT_FORECAST_SELECTOR_CONFIG["name"]),
                forecast_profile("default"),
                0.7,
            ),
            "forecast_profile": _slot(
                str(DEFAULT_FORECAST_SELECTOR_CONFIG["name"]),
                str(DEFAULT_FORECAST_SELECTOR_CONFIG["name"]),
                0.7,
            ),
        }

    def _solve_compat(
        self,
        operator: str,
        query: str,
        graph: AttributeValueGraph,
        base: dict[str, Slot],
    ) -> dict[str, Slot]:
        """Compatibility path for existing non-MVP executor operators."""
        slots = {**base}
        if operator in ("AVG", "MAX", "MIN", "DIFFERENCE", "LOOKUP", "TOP_K", "TREND", "PREDICT"):
            slots["target_field"] = self._field_slot(query, graph)
        if operator == "LOOKUP":
            entity = resolve_entity(query, graph)
            slots["entity"] = _slot(entity or "", entity, 0.85 if entity else 0.0)
        elif operator == "DIFFERENCE":
            left, right = resolve_two_entities(query, graph)
            if left is not None:
                slots["left_entity"] = _slot(left, left, 0.85)
            if right is not None:
                slots["right_entity"] = _slot(right, right, 0.85)
        elif operator == "TOP_K":
            slots["target_entity_type"] = _slot("company", "company", 0.9)
            k = extract_k(query)
            slots["k"] = _slot(str(k), k, 0.9)
            slots["order"] = _slot("descending", "descending", 0.9)
        elif operator == "TREND":
            from_year, to_year = resolve_two_years(query)
            anchor_year = resolve_year(query) or (graph.years[-1] if graph.years else 2024)
            slots["from_time"] = _slot(str(from_year or anchor_year - 3), from_year or anchor_year - 3, 0.8 if from_year else 0.4)
            slots["to_time"] = _slot(str(to_year or anchor_year), to_year or anchor_year, 0.8 if to_year else 0.4)
        elif operator == "PREDICT":
            entity = resolve_entity(query, graph)
            horizon = resolve_year(query) or 2025
            slots["entity"] = _slot(entity or "", entity, 0.85 if entity else 0.0)
            slots["history_range"] = _slot(f"{horizon - 4}-{horizon - 1}", f"{horizon - 4}-{horizon - 1}", 0.7)
            slots["horizon"] = _slot(str(horizon), horizon, 0.8)
        return slots

    def _solve_composite_rank_then_margin(
        self,
        query: str,
        graph: AttributeValueGraph,
        routing: RoutingResult,
    ) -> CompositeOperatorPlan:
        year = resolve_year(query)
        industry = resolve_industry(query, graph)
        rank_field = _rank_field(query, graph.fields, self.field_grounder)
        numerator, denominator, num_conf, den_conf = _margin_fields_after_rank(
            query,
            graph.fields,
            self.field_grounder,
            rank_field.field_name,
        )
        order = _composite_order(routing)
        first_operator = "ARGMIN" if order == "ascending" else "ARGMAX"
        step1_slots: dict[str, Slot] = {
            "target_entity_type": _slot("company", "company", 0.9),
            "target_field": _slot(query, rank_field.field_name, rank_field.confidence),
        }
        step2_slots: dict[str, Slot] = {
            "numerator_field": _slot(query, numerator, num_conf),
            "denominator_field": _slot(query, denominator, den_conf),
        }
        if year is not None:
            step1_slots["year"] = _slot(str(year), year, 0.99)
            step2_slots["year"] = _slot(str(year), year, 0.99)
        if industry is not None:
            step1_slots["industry"] = _slot(industry, industry, 0.95)
            step2_slots["industry"] = _slot(industry, industry, 0.95)
        steps = (
            OperatorPlan(
                operator=first_operator,
                slots=step1_slots,
                confidence=0.9,
                trace={"step_id": "s1", "role": "rank_entity", "routing": routing.to_dict()},
            ),
            OperatorPlan(
                operator="MARGIN",
                slots=step2_slots,
                depends_on={"entity": "$s1.entity"},
                confidence=0.9,
                trace={"step_id": "s2", "role": "compute_margin", "routing": routing.to_dict()},
            ),
        )
        return CompositeOperatorPlan(
            steps=steps,
            confidence=routing.confidence,
            trace={"solver": "operator_solvers.composite.v1", "routing": routing.to_dict()},
        )

    def _solve_composite_topk_then_sum(
        self,
        query: str,
        graph: AttributeValueGraph,
        routing: RoutingResult,
    ) -> CompositeOperatorPlan:
        base = self._base_slots(query, graph)
        rank_field = _rank_field(query, graph.fields, self.field_grounder)
        sum_field = _sum_field_after_topk(query, graph.fields, self.field_grounder, rank_field.field_name)
        k = extract_k(query)
        order = extract_order(query)
        step1_slots = {
            **base,
            "target_entity_type": _slot("company", "company", 0.9),
            "target_field": _slot(query, rank_field.field_name, rank_field.confidence),
            "k": _slot(str(k), k, 0.9),
            "order": _slot(order, order, 0.9),
        }
        step2_slots = {
            **base,
            "target_field": _slot(query, sum_field.field_name, sum_field.confidence),
        }
        steps = (
            OperatorPlan(
                operator="TOP_K",
                slots=step1_slots,
                confidence=0.9,
                trace={"step_id": "s1", "role": "select_entities", "routing": routing.to_dict()},
            ),
            OperatorPlan(
                operator="SUM",
                slots=step2_slots,
                depends_on={"entities": "$s1.entities"},
                confidence=0.9,
                trace={"step_id": "s2", "role": "aggregate_selected_entities", "routing": routing.to_dict()},
            ),
        )
        return CompositeOperatorPlan(
            steps=steps,
            confidence=routing.confidence,
            trace={"solver": "operator_solvers.composite.v1", "routing": routing.to_dict()},
        )

    def _solve_composite_rank_then_lookup(
        self,
        query: str,
        graph: AttributeValueGraph,
        routing: RoutingResult,
    ) -> CompositeOperatorPlan:
        base = self._base_slots(query, graph)
        rank_field = _rank_field(query, graph.fields, self.field_grounder)
        lookup_field = _lookup_field_after_rank(query, graph.fields, self.field_grounder, rank_field.field_name)
        order = _composite_order(routing)
        first_operator = "ARGMIN" if order == "ascending" else "ARGMAX"
        step1_slots = {
            **base,
            "target_entity_type": _slot("company", "company", 0.9),
            "target_field": _slot(query, rank_field.field_name, rank_field.confidence),
        }
        step2_slots = {
            **base,
            "target_field": _slot(query, lookup_field.field_name, lookup_field.confidence),
        }
        steps = (
            OperatorPlan(
                operator=first_operator,
                slots=step1_slots,
                confidence=0.9,
                trace={"step_id": "s1", "role": "rank_entity", "routing": routing.to_dict()},
            ),
            OperatorPlan(
                operator="LOOKUP",
                slots=step2_slots,
                depends_on={"entity": "$s1.entity"},
                confidence=0.9,
                trace={"step_id": "s2", "role": "lookup_selected_entity", "routing": routing.to_dict()},
            ),
        )
        return CompositeOperatorPlan(
            steps=steps,
            confidence=routing.confidence,
            trace={"solver": "operator_solvers.composite.v1", "routing": routing.to_dict()},
        )


def resolve_question_constant_slots(query: str) -> list[dict[str, object]]:
    constants: list[dict[str, object]] = []
    seen: set[tuple[str, float, str]] = set()

    def add(surface: str, value: float, role: str) -> None:
        key = (normalize_identifier(surface), float(value), role)
        if key in seen:
            return
        seen.add(key)
        constants.append(
            {
                "surface": surface,
                "value": float(value),
                "source": "question_text",
                "role": role,
            }
        )

    for match in re.finditer(
        r"\b(?:all\s+)?(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+quarters?\b",
        query,
        re.IGNORECASE,
    ):
        value = _constant_number_value(match.group("number"))
        if value is not None:
            add(match.group(0).strip(), value, "multiplicative_factor")

    for match in re.finditer(r"([一二两三四五六七八九十])\s*个?\s*季度", query):
        value = _constant_number_value(match.group(1))
        if value is not None:
            add(match.group(0).strip(), value, "multiplicative_factor")

    for match in re.finditer(r"\bper\s+(?P<number>\d+(?:\.\d+)?)\s+shares?\b", query, re.IGNORECASE):
        value = _constant_number_value(match.group("number"))
        if value is not None:
            add(match.group(0).strip(), value, "unit_base")
    return constants


def _drop_constants_already_represented_by_tokens(
    constants: list[dict[str, object]],
    tokens: Sequence[AttributeValueToken],
) -> list[dict[str, object]]:
    if not constants or not tokens:
        return constants
    kept: list[dict[str, object]] = []
    for constant in constants:
        if _constant_represented_by_any_token(constant, tokens):
            continue
        kept.append(constant)
    return kept


def _constant_represented_by_any_token(
    constant: dict[str, object],
    tokens: Sequence[AttributeValueToken],
) -> bool:
    if constant.get("role") != "multiplicative_factor":
        return False
    try:
        value = float(constant.get("value"))
    except (TypeError, ValueError):
        return False
    constant_terms = _normalized_word_set(str(constant.get("surface") or ""))
    for token in tokens:
        try:
            token_value = float(token.value)
        except (TypeError, ValueError):
            continue
        if abs(token_value - value) > 1e-12:
            continue
        token_terms = _product_token_terms(token)
        if constant_terms & token_terms:
            return True
    return False


def _constant_number_value(value: str) -> float | None:
    normalized = value.strip().lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if normalized in words:
        return float(words[normalized])
    try:
        return float(normalized)
    except ValueError:
        return None


def _resolve_product_factor_tokens(
    query: str,
    graph: AttributeValueGraph,
) -> tuple[AttributeValueToken, ...]:
    numeric_tokens = tuple(
        token
        for token in graph.tokens
        if isinstance(token.value, (int, float)) and not _token_looks_like_year_constant(token)
    )
    if len(numeric_tokens) <= 4:
        return numeric_tokens

    query_terms = _product_query_terms(query)
    if not query_terms:
        return ()
    scored: list[tuple[float, int, AttributeValueToken]] = []
    for index, token in enumerate(numeric_tokens):
        token_terms = _product_token_terms(token)
        overlap = query_terms & token_terms
        if not overlap:
            continue
        score = float(len(overlap))
        if token.unit and any(term in str(token.unit).lower() for term in ("share", "per share", "usd", "$")):
            score += 0.3
        scored.append((score, index, token))
    if len(scored) < 2:
        return ()
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(token for _, _, token in scored[:3])


def _token_looks_like_year_constant(token: AttributeValueToken) -> bool:
    try:
        value = float(token.value)
    except (TypeError, ValueError):
        return False
    if 1900 <= value <= 2100 and value.is_integer():
        unit = str(token.unit or "").lower()
        label = normalize_identifier(" ".join(str(part or "") for part in (token.field_name, token.field_label, token.raw_label)))
        return "year" in unit or "year" in label or not unit
    return False


def _product_query_terms(query: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "all",
        "did",
        "for",
        "had",
        "how",
        "if",
        "in",
        "many",
        "much",
        "number",
        "of",
        "same",
        "the",
        "total",
        "was",
        "were",
        "what",
    }
    return {
        term
        for term in normalize_identifier(query).split("_")
        if len(term) >= 3 and term not in stopwords
    }


def _product_token_terms(token: AttributeValueToken) -> set[str]:
    parts = [
        token.field_name,
        token.field_label,
        token.raw_label,
        token.unit,
        token.company_name,
        token.entity_id,
    ]
    if token.dimensions:
        parts.extend(str(value) for value in token.dimensions.values())
    return {
        term
        for term in normalize_identifier(" ".join(str(part or "") for part in parts)).split("_")
        if len(term) >= 3
    }


def solve_operator_plan(
    query: str,
    graph: AttributeValueGraph,
    routing: RoutingResult,
    *,
    field_grounder: FieldGrounder | None = None,
    unit_resolver: UnitResolver | None = None,
    operator: str | None = None,
) -> OperatorPlan:
    return OperatorSolver(
        field_grounder=field_grounder,
        unit_resolver=unit_resolver,
    ).solve(query, graph, routing, operator=operator)


def solve_composite_operator_plan(
    query: str,
    graph: AttributeValueGraph,
    routing: RoutingResult,
    *,
    field_grounder: FieldGrounder | None = None,
    unit_resolver: UnitResolver | None = None,
) -> CompositeOperatorPlan:
    solver = OperatorSolver(
        field_grounder=field_grounder,
        unit_resolver=unit_resolver,
    )
    case = (routing.trace or {}).get("composite_case")
    if case == "arg_rank_then_margin":
        return solver._solve_composite_rank_then_margin(query, graph, routing)
    if case == "topk_then_sum":
        return solver._solve_composite_topk_then_sum(query, graph, routing)
    if case == "arg_rank_then_lookup":
        return solver._solve_composite_rank_then_lookup(query, graph, routing)
    raise ValueError(f"Unsupported composite route case: {case!r}")


def extract_condition(query: str) -> tuple[str, float, float, str | None] | None:
    for pattern, op in CONDITION_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        number_match = NUMBER_RE.search(query, match.end())
        if number_match is None:
            text_threshold = _textual_threshold(query[match.end():])
            if text_threshold is None:
                return None
            threshold, unit = text_threshold
            return op, threshold, 0.9, unit
        raw = number_match.group(0).replace(",", "")
        unit = _unit_after_number(query, number_match.end())
        return op, float(raw), 0.95, unit
    return None


def resolve_review_hardening_field(query: str, fields: Sequence[str]) -> str | None:
    normalized = normalize_identifier(query)
    tokens = set(normalized.split("_"))
    field_set = set(fields)
    if "top_line" in normalized or "topline" in tokens or "turnover" in tokens or "sales" in tokens or "revenue" in tokens:
        return "revenue" if "revenue" in field_set else None
    if (
        "bottom_line" in normalized
        or "bottomline" in tokens
        or "net_earnings" in normalized
        or "profit_after_tax" in normalized
    ):
        return "net_profit" if "net_profit" in field_set else None
    if "workforce" in tokens or "headcount" in tokens or "employees" in tokens:
        return "employees" if "employees" in field_set else None
    if "ebitda" in tokens:
        return "ebitda" if "ebitda" in field_set else None
    return None


def extract_requested_unit(query: str) -> str | None:
    lowered = query.lower()
    if re.search(r"\bin\s+employees\b|\bin\s+headcount\b|\bin\s+workforce\b", lowered):
        return "employees"
    if re.search(r"\bin\s+percent\b|\bin\s+percentage\b|\bin\s+%\b", lowered):
        return "percent"
    money_match = re.search(
        r"\bin\s+((?:million|billion|thousand)\s+)?(usd|eur|cny|rmb|dollars?)\b",
        lowered,
        re.IGNORECASE,
    )
    if money_match:
        scale = (money_match.group(1) or "").strip()
        currency = money_match.group(2).upper()
        if currency == "DOLLARS":
            currency = "USD"
        unit = f"{scale} {currency}".strip()
        return unit
    return None


def resolve_dimension_filters(
    query: str,
    graph: AttributeValueGraph,
) -> dict[str, str]:
    dimensions = _dimension_values(graph)
    filters: dict[str, str] = {}
    normalized_query = normalize_identifier(query)
    compact_query = normalized_query.replace("_", "")
    for key, values in dimensions.items():
        matches = [
            (_dimension_match_score(value, normalized_query, compact_query), value)
            for value in values
            if _dimension_match_score(value, normalized_query, compact_query) > 0
        ]
        if not matches:
            continue
        matches.sort(reverse=True)
        if len(matches) == 1 or matches[0][0] > matches[1][0]:
            filters[key] = matches[0][1]
    return filters


def resolve_two_dimension_values(
    query: str,
    graph: AttributeValueGraph,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    dimensions = _dimension_values(graph)
    normalized_query = normalize_identifier(query)
    compact_query = normalized_query.replace("_", "")
    for key, values in dimensions.items():
        matches = [
            (_dimension_match_score(value, normalized_query, compact_query), value)
            for value in values
            if _dimension_match_score(value, normalized_query, compact_query) > 0
        ]
        matches.sort(reverse=True)
        if len(matches) >= 2:
            return {key: matches[0][1]}, {key: matches[1][1]}
    return None, None


def resolve_difference_field_pair(
    query: str,
    fields: Sequence[str],
) -> tuple[str, str] | None:
    field_set = set(fields)
    normalized = normalize_identifier(query)
    if (
        "basic" in field_set
        and "diluted" in field_set
        and "basic" in normalized
        and "diluted" in normalized
        and re.search(r"\bdifference\b", query, re.IGNORECASE)
    ):
        return "basic", "diluted"
    mentions = field_mentions(query, fields)
    if len(mentions) >= 2 and re.search(r"\bdifference\b|\bgap\b", query, re.IGNORECASE):
        return mentions[0][2], mentions[1][2]
    return None


def _dimension_filter_keeps_field_pair(
    graph: AttributeValueGraph,
    slots: dict[str, Slot],
    left_field: str,
    right_field: str,
) -> bool:
    return _dimension_filter_keeps_fields(graph, slots, [left_field, right_field])


def _dimension_filter_keeps_fields(
    graph: AttributeValueGraph,
    slots: dict[str, Slot],
    fields: Sequence[str],
) -> bool:
    dimensions = _slot_grounded(slots, "dimension_filters")
    if not isinstance(dimensions, dict) or not dimensions:
        return True
    year = _slot_grounded(slots, "year")
    industry = _slot_grounded(slots, "industry")
    try:
        year_value = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_value = None
    for field in fields:
        tokens = graph.select(
            field_name=field,
            year=year_value,
            industry=str(industry) if industry is not None else None,
        )
        if not tokens:
            return True
        if not any(_token_matches_dimensions(token, dimensions) for token in tokens):
            return False
    return True


def _token_matches_dimensions(token: object, dimensions: dict[str, str]) -> bool:
    token_dimensions = getattr(token, "dimensions", None) or {}
    for key, expected in dimensions.items():
        if str(key) not in token_dimensions:
            continue
        if not _dimension_value_matches(token_dimensions.get(str(key)), expected):
            return False
    return True


def _dimension_value_matches(actual: object | None, expected: object | None) -> bool:
    if actual is None or expected is None:
        return False
    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    actual_norm = normalize_identifier(actual_text)
    expected_norm = normalize_identifier(expected_text)
    if actual_norm == expected_norm:
        return True
    actual_core = "_".join(_dimension_core_parts(actual_text))
    expected_core = "_".join(_dimension_core_parts(expected_text))
    if not actual_core or not expected_core:
        return False
    return expected_core in actual_core or actual_core in expected_core


def resolve_share_of_total_dimensions(
    query: str,
    graph: AttributeValueGraph,
) -> tuple[dict[str, str], dict[str, str]] | None:
    if not re.search(r"\b(?:share|proportion|percentage)\b", query, re.IGNORECASE):
        return None
    if not re.search(r"\b(?:from|of)\b", query, re.IGNORECASE):
        return None
    dimensions = _dimension_values(graph)
    if not dimensions:
        return None
    normalized_query = normalize_identifier(query)
    compact_query = normalized_query.replace("_", "")
    for key, values in dimensions.items():
        total_value = next(
            (value for value in values if _is_total_like_dimension_value(value)),
            None,
        )
        if total_value is None:
            continue
        part_matches = [
            (_dimension_match_score(value, normalized_query, compact_query), value)
            for value in values
            if value != total_value
            and _dimension_match_score(value, normalized_query, compact_query) > 0
        ]
        if not part_matches:
            continue
        part_matches.sort(reverse=True)
        if len(part_matches) > 1 and part_matches[0][0] == part_matches[1][0]:
            continue
        return {key: part_matches[0][1]}, {key: total_value}
    return None


def resolve_explicit_total_row_dimensions(
    query: str,
    graph: AttributeValueGraph,
) -> dict[str, str] | None:
    if not re.search(r"\b(?:total|overall)\b", query, re.IGNORECASE):
        return None
    if not any(_token_is_aggregate_like(token) for token in graph.tokens):
        return None
    return {"is_total_row": "true"}


def _is_total_like_field(field_name: object) -> bool:
    normalized = normalize_identifier(str(field_name))
    return normalized == "total" or normalized.startswith("total_") or normalized.endswith("_total")


def _is_total_like_dimension_value(value: object) -> bool:
    normalized = normalize_identifier(str(value))
    return normalized == "total" or normalized.startswith("total_")


def _should_use_next_total_row(
    query: str,
    denominator: object,
    share_dimensions: tuple[dict[str, str], dict[str, str]] | None,
) -> bool:
    if share_dimensions is None:
        return False
    if not _is_total_like_field(denominator):
        return False
    return bool(re.search(r"\bbefore\b", query, re.IGNORECASE))


def _year_is_dimension_value(
    query: str,
    graph: AttributeValueGraph,
    year: int,
) -> bool:
    if graph.years:
        return False
    year_text = str(year)
    normalized_query = normalize_identifier(query)
    compact_query = normalized_query.replace("_", "")
    for values in _dimension_values(graph).values():
        for value in values:
            if year_text not in str(value):
                continue
            if _dimension_match_score(str(value), normalized_query, compact_query) > 0:
                return True
    return False


def _dimension_match_score(value: str, normalized_query: str, compact_query: str) -> int:
    normalized_value = normalize_identifier(value)
    compact_value = normalized_value.replace("_", "")
    if compact_value and compact_value in compact_query:
        return len(compact_value)
    core_parts = _dimension_core_parts(value)
    if len(core_parts) < 2:
        return 0
    core = "_".join(core_parts)
    compact_core = core.replace("_", "")
    if compact_core and compact_core in compact_query:
        return len(compact_core)
    query_parts = normalized_query.split("_")
    if _dimension_parts_appear_in_order(core_parts, query_parts):
        return len(compact_core) - 1
    return 0


def _dimension_core_parts(value: str) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", str(value).lower())
    normalized = normalize_identifier(text)
    stopwords = {
        "m",
        "mm",
        "bn",
        "b",
        "k",
        "gbp",
        "usd",
        "eur",
        "cny",
        "rmb",
        "million",
        "millions",
        "billion",
        "billions",
        "thousand",
        "thousands",
    }
    return [part for part in normalized.split("_") if part and part not in stopwords]


def _dimension_parts_appear_in_order(parts: list[str], query_parts: list[str]) -> bool:
    position = 0
    for part in parts:
        try:
            position = query_parts.index(part, position) + 1
        except ValueError:
            return False
    return True


def resolve_ratio_fields(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, str, float, float]:
    percent_of_fields = _resolve_percent_of_ratio_fields(query, fields, grounder)
    if percent_of_fields is not None:
        return percent_of_fields

    explicit_ratio_fields = _resolve_explicit_ratio_of_to_fields(query, fields, grounder)
    if explicit_ratio_fields is not None:
        return explicit_ratio_fields

    mentions = field_mentions(query, fields)
    if len(mentions) >= 2:
        return mentions[0][2], mentions[1][2], 0.95, 0.95

    grounding = grounder.ground(query, fields)
    ranked = sorted(grounding.scores.items(), key=lambda item: item[1], reverse=True)
    numerator = grounding.field_name
    denominator = "revenue" if "revenue" in fields and numerator != "revenue" else None
    if denominator is None:
        denominator = next((field for field, _ in ranked if field != numerator), numerator)
        den_conf = float(dict(ranked).get(denominator, 0.4))
    else:
        den_conf = 0.7
    return numerator, denominator, grounding.confidence, den_conf


def _resolve_percent_of_ratio_fields(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, str, float, float] | None:
    denominator_first = re.search(
        r"\bwhat\s+percent(?:age)?\s+of\s+(?P<denominator>.+?)\s+"
        r"(?:were|was|are|is|where)\s+(?P<numerator>.+)",
        query,
        re.IGNORECASE,
    )
    if denominator_first is not None:
        numerator_text = _strip_ratio_question_text(denominator_first.group("numerator"))
        denominator_text = _strip_ratio_question_text(denominator_first.group("denominator"))
        if numerator_text and denominator_text:
            numerator, numerator_conf = _ground_ratio_phrase(numerator_text, fields, grounder)
            denominator, denominator_conf = _ground_ratio_phrase(denominator_text, fields, grounder)
            if numerator != denominator:
                return numerator, denominator, numerator_conf, denominator_conf

    match = re.search(
        r"\bpercent(?:age)?\s+of\s+(?P<numerator>.+?)\s+"
        r"(?P<link>as\s+part\s+of|to)\s+(?P<denominator>.+)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    numerator_text = _strip_ratio_question_text(match.group("numerator"))
    denominator_text = _strip_ratio_question_text(match.group("denominator"))
    if not numerator_text or not denominator_text:
        return None

    numerator, numerator_conf = _ground_ratio_phrase(numerator_text, fields, grounder)
    denominator, denominator_conf = _ground_ratio_phrase(denominator_text, fields, grounder)
    if numerator == denominator:
        return None
    return numerator, denominator, numerator_conf, denominator_conf


def _resolve_explicit_ratio_of_to_fields(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, str, float, float] | None:
    match = re.search(
        r"\bratio\s+of\s+(?P<numerator>.+?)\s+to\s+(?P<denominator>.+)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    numerator_text = match.group("numerator").strip(" ?.,")
    denominator_text = match.group("denominator").strip(" ?.,")
    if not numerator_text or not denominator_text:
        return None
    denominator_mentions = field_mentions(denominator_text, fields)
    if denominator_mentions:
        denominator = denominator_mentions[0][2]
        denominator_conf = 0.95
    else:
        denominator_grounding = grounder.ground(denominator_text, fields)
        denominator = denominator_grounding.field_name
        denominator_conf = denominator_grounding.confidence
    numerator_mentions = field_mentions(numerator_text, fields)
    if numerator_mentions:
        numerator = numerator_mentions[0][2]
        numerator_conf = 0.95
    else:
        numerator_grounding = grounder.ground(numerator_text, fields)
        numerator = numerator_grounding.field_name
        numerator_conf = numerator_grounding.confidence
    return numerator, denominator, numerator_conf, denominator_conf


def _strip_ratio_question_text(text: str) -> str:
    cleaned = text.strip(" ?.,")
    cleaned = re.sub(
        r"^\s*(?:the\s+|a\s+|an\s+)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:in|during|for|as\s+of)\s+(?:fiscal\s+)?(?:year\s+)?(?:19|20)\d{2}\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ?.,")


def _ground_ratio_phrase(
    phrase: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, float]:
    mentions = field_mentions(phrase, fields)
    if mentions:
        return mentions[0][2], 0.95
    grounding = grounder.ground(phrase, fields)
    return grounding.field_name, grounding.confidence


def resolve_multi_numerator_fields(
    query: str,
    fields: Sequence[str],
    numerator: str,
    denominator: str,
) -> list[str] | None:
    if not re.search(r"\bdue\s+(?:19|20)\d{2}\b", query, re.IGNORECASE):
        return None
    if not re.search(r"\b(?:and|,)\b", query, re.IGNORECASE):
        return None
    query_prefix = re.split(r"\bto\b|\bover\b|\bdivided\s+by\b", query, maxsplit=1, flags=re.IGNORECASE)[0]
    years = YEAR_RE.findall(query_prefix)
    if len(years) < 2:
        return None
    mentioned = [
        field
        for _, _, field, _ in field_mentions(query_prefix, fields)
        if field != denominator
    ]
    selected: list[str] = []
    for year in years:
        match = next(
            (
                field
                for field in fields
                if field != denominator
                and re.search(rf"(?:^|_){re.escape(year)}(?:_|$)", field)
                and re.search(r"senior_notes|notes", field)
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    if len(selected) < 2 and len(mentioned) >= 2:
        selected = mentioned
    selected = list(dict.fromkeys(selected))
    if len(selected) < 2 or numerator not in selected:
        return None
    return selected


def resolve_explicit_parallel_sum_fields(
    query: str,
    fields: Sequence[str],
) -> list[str] | None:
    if not re.search(r"\b(?:and|plus|together|combined)\b|,", query, re.IGNORECASE):
        return None
    if not re.search(r"\b(?:total|sum|combined|together)\b", query, re.IGNORECASE):
        return None
    mentions = [field for _, _, field, _ in field_mentions(query, fields)]
    deduped = list(dict.fromkeys(mentions))
    deduped = _expand_elided_parallel_sum_fields(query, fields, deduped)
    if len(deduped) < 2:
        return None
    return deduped


def _expand_elided_parallel_sum_fields(
    query: str,
    fields: Sequence[str],
    mentioned_fields: list[str],
) -> list[str]:
    if not mentioned_fields:
        return mentioned_fields
    normalized_query = normalize_identifier(query).replace("_", " ")
    selected = list(mentioned_fields)
    for anchor_field in mentioned_fields:
        anchor_aliases = _field_surface_candidates(anchor_field)
        for field in fields:
            if field in selected:
                continue
            for anchor_alias in anchor_aliases:
                anchor_terms = _normalized_terms(anchor_alias)
                if len(anchor_terms) < 3:
                    continue
                for candidate_alias in _field_surface_candidates(field):
                    candidate_terms = _normalized_terms(candidate_alias)
                    common_len = _common_prefix_len(anchor_terms, candidate_terms)
                    if common_len < 2 or common_len >= len(candidate_terms):
                        continue
                    suffix_terms = candidate_terms[common_len:]
                    if len(suffix_terms) < 2:
                        continue
                    suffix = " ".join(suffix_terms)
                    if re.search(rf"\b{re.escape(suffix)}\b", normalized_query):
                        selected.append(field)
                        break
                if field in selected:
                    break
    return list(dict.fromkeys(selected))


def _field_surface_candidates(field: str) -> tuple[str, ...]:
    return (field, field.replace("_", " "), *field_aliases(field))


def _normalized_terms(text: str) -> list[str]:
    return normalize_identifier(text).replace("_", " ").split()


def _common_prefix_len(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    for left_term, right_term in zip(left, right):
        if left_term != right_term:
            break
        count += 1
    return count


def resolve_share_fields(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, str, float, float]:
    explicit_percentage_fields = _resolve_explicit_percentage_share_fields(query, fields, grounder)
    if explicit_percentage_fields is not None:
        return explicit_percentage_fields

    percent_of_fields = _resolve_percent_of_ratio_fields(query, fields, grounder)
    if percent_of_fields is not None:
        return percent_of_fields

    mentions = field_mentions(query, fields)
    if len(mentions) >= 2:
        if re.search(r"\bwhat\s+share\s+of\b", query, re.IGNORECASE):
            return mentions[1][2], mentions[0][2], 0.95, 0.95
        return mentions[0][2], mentions[1][2], 0.95, 0.95
    return resolve_ratio_fields(query, fields, grounder)


def resolve_same_field_share_token_ids(
    query: str,
    graph: AttributeValueGraph,
    field: str,
    *,
    year: object | None = None,
    industry: object | None = None,
) -> tuple[list[str], list[str]] | None:
    selected_year = int(year) if isinstance(year, int) else None
    selected_industry = str(industry) if industry is not None else None
    candidates = graph.select(
        field_name=field,
        year=selected_year,
        industry=selected_industry,
    )
    if len(candidates) < 2:
        return None

    field_terms = _normalized_word_set(field.replace("_", " "))
    query_terms = _normalized_word_set(query)
    residual_terms = {
        term
        for term in query_terms - field_terms - _SHARE_QUALIFIER_STOPWORDS
        if not term.isdigit()
    }
    if not residual_terms:
        return None

    scored: list[tuple[float, int, str, AttributeValueToken]] = []
    for index, token in enumerate(candidates):
        label_terms = _normalized_word_set(_token_primary_descriptor(token))
        evidence_terms = _normalized_word_set(_token_secondary_descriptor(token))
        qualifier_terms = label_terms - field_terms
        numerator_overlap = len(residual_terms & label_terms) * 3 + len(residual_terms & evidence_terms)
        broadness = len(qualifier_terms)
        scored.append((float(numerator_overlap), broadness, str(token.token_id), token))

    numerator_row = max(scored, key=lambda item: (item[0], -item[1], item[2]))
    if numerator_row[0] <= 0:
        return None
    numerator = numerator_row[3]
    denominator_rows = [row for row in scored if row[3].token_id != numerator.token_id]
    if not denominator_rows:
        return None
    denominator = min(
        denominator_rows,
        key=lambda item: (item[1], -abs(float(item[3].value)), item[2]),
    )[3]
    return [numerator.token_id], [denominator.token_id]


def resolve_aggregate_share_token_ids(
    query: str,
    graph: AttributeValueGraph,
    numerator_field: str,
    denominator_field: str,
    *,
    year: object | None = None,
    industry: object | None = None,
) -> tuple[list[str], list[str]] | None:
    if not _query_requests_aggregate_denominator(query):
        return None
    selected_year = int(year) if isinstance(year, int) else None
    selected_industry = str(industry) if industry is not None else None
    numerator_candidates = graph.select(
        field_name=numerator_field,
        year=selected_year,
        industry=selected_industry,
    )
    denominator_candidates = graph.select(
        field_name=denominator_field,
        year=selected_year,
        industry=selected_industry,
    )
    aggregate_denominators = tuple(
        token for token in denominator_candidates if _token_is_aggregate_like(token)
    )
    if (
        selected_industry is not None
        and (
            not numerator_candidates
            or not aggregate_denominators
            or not _has_table_candidate_pair(numerator_candidates, aggregate_denominators)
        )
    ):
        unfiltered_numerator_candidates = graph.select(
            field_name=numerator_field,
            year=selected_year,
            industry=None,
        )
        unfiltered_denominator_candidates = graph.select(
            field_name=denominator_field,
            year=selected_year,
            industry=None,
        )
        unfiltered_aggregate_denominators = tuple(
            token for token in unfiltered_denominator_candidates if _token_is_aggregate_like(token)
        )
        if (
            not numerator_candidates
            or not aggregate_denominators
            or _has_table_candidate_pair(unfiltered_numerator_candidates, unfiltered_aggregate_denominators)
        ):
            numerator_candidates = unfiltered_numerator_candidates
            denominator_candidates = unfiltered_denominator_candidates
            aggregate_denominators = unfiltered_aggregate_denominators
    if not numerator_candidates or not aggregate_denominators:
        return None
    if _query_requests_aggregate_numerator(query, numerator_field, numerator_candidates):
        part_numerators = tuple(
            token for token in numerator_candidates if _token_is_aggregate_like(token)
        ) or numerator_candidates
    else:
        part_numerators = tuple(
            token for token in numerator_candidates if not _token_is_aggregate_like(token)
        ) or numerator_candidates

    pairs = [
        (
            _aggregate_share_pair_score(query, numerator, denominator),
            numerator,
            denominator,
        )
        for numerator in part_numerators
        for denominator in aggregate_denominators
        if numerator.token_id != denominator.token_id
    ]
    if not pairs:
        return None
    same_table_pairs = [
        pair for pair in pairs if _tokens_share_table_scope(pair[1], pair[2])
    ]
    if same_table_pairs:
        pairs = same_table_pairs
    score, numerator, denominator = max(
        pairs,
        key=lambda item: (
            item[0],
            _token_source_row_index(item[1]),
            _token_source_row_index(item[2]),
            str(item[1].token_id),
            str(item[2].token_id),
        ),
    )
    if score <= 0:
        return None
    return [numerator.token_id], [denominator.token_id]


def _query_requests_period_column(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:less\s+than|more\s+than|within|due|period|thereafter|current|q[1-4])\b",
            query,
            re.IGNORECASE,
        )
    )


def _query_requests_aggregate_denominator(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:total|overall|aggregate|as\s+a\s+percentage\s+of|percent(?:age)?\s+of)\b",
            query,
            re.IGNORECASE,
        )
    )


def _aggregate_share_pair_score(
    query: str,
    numerator: AttributeValueToken,
    denominator: AttributeValueToken,
) -> float:
    score = 3.0
    if _tokens_share_table_scope(numerator, denominator):
        score += 12.0
    if _tokens_share_column_scope(numerator, denominator):
        score += 4.0
        if _query_requests_period_column(query) and not _token_column_is_total(denominator):
            score += 10.0
        if _query_requests_period_column(query) and _token_column_is_total(denominator):
            score -= 8.0
    if _tokens_share_column_scope(numerator, denominator) and _token_column_is_total(numerator):
        score += 6.0
    if _tokens_share_row_scope(numerator, denominator):
        score += 8.0
    if _token_is_table_provenance(numerator) and _token_is_table_provenance(denominator):
        score += 2.0
    if numerator.year is not None and denominator.year == numerator.year:
        score += 3.0
    query_terms = _normalized_word_set(query) - _SHARE_QUALIFIER_STOPWORDS - {"total", "overall", "aggregate"}
    numerator_terms = _normalized_word_set(_token_primary_descriptor(numerator))
    numerator_terms |= _normalized_word_set(_token_secondary_descriptor(numerator))
    denominator_terms = _normalized_word_set(_token_primary_descriptor(denominator))
    score += 3.0 * len(query_terms & numerator_terms)
    score += 0.5 * len(query_terms & denominator_terms)
    if _token_is_aggregate_like(denominator):
        score += 3.0
    if _token_is_aggregate_like(numerator):
        score -= 8.0
        score += _aggregate_magnitude_tiebreaker(numerator)
        score += _aggregate_magnitude_tiebreaker(denominator)
    return score


def _has_table_candidate_pair(
    numerators: Sequence[AttributeValueToken],
    denominators: Sequence[AttributeValueToken],
) -> bool:
    return any(_token_is_table_provenance(numerator) for numerator in numerators) and any(
        _token_is_table_provenance(denominator) for denominator in denominators
    )


def _tokens_share_table_scope(
    left: AttributeValueToken,
    right: AttributeValueToken,
) -> bool:
    left_table = _token_table_scope(left)
    right_table = _token_table_scope(right)
    return bool(left_table and right_table and left_table == right_table)


def _aggregate_magnitude_tiebreaker(token: AttributeValueToken) -> float:
    try:
        value = abs(float(token.value))
    except (TypeError, ValueError):
        return 0.0
    return min(value, 1_000_000.0) / 1_000_000.0


def _token_table_scope(token: AttributeValueToken) -> str | None:
    dimensions = token.dimensions or {}
    for key in ("table_id", "table"):
        value = dimensions.get(key)
        if value:
            return str(value)
    if token.source is not None and token.source.table:
        return str(token.source.table)
    return None


def _tokens_share_column_scope(
    left: AttributeValueToken,
    right: AttributeValueToken,
) -> bool:
    left_column = left.source.column if left.source is not None else None
    right_column = right.source.column if right.source is not None else None
    return bool(left_column and right_column and str(left_column) == str(right_column))


def _token_column_is_total(token: AttributeValueToken) -> bool:
    dimensions = token.dimensions or {}
    column = dimensions.get("column_label")
    if not column and token.source is not None:
        column = token.source.column
    return normalize_identifier(str(column or "")) == "total"


def _token_is_table_provenance(token: AttributeValueToken) -> bool:
    dimensions = token.dimensions or {}
    if dimensions.get("provenance_channel") == "table":
        return True
    return token.source is not None and bool(token.source.table)


def _tokens_share_row_scope(
    left: AttributeValueToken,
    right: AttributeValueToken,
) -> bool:
    left_row = left.source.row if left.source is not None else None
    right_row = right.source.row if right.source is not None else None
    return left_row is not None and right_row is not None and int(left_row) == int(right_row)


def _token_source_row_index(token: AttributeValueToken) -> int:
    if token.source is None or token.source.row is None:
        return 10**6
    return int(token.source.row)


def _token_is_aggregate_like(token: AttributeValueToken) -> bool:
    if token.is_aggregate:
        return True
    dimensions = token.dimensions or {}
    if _truthy_dimension(dimensions.get("is_aggregate")) or _truthy_dimension(dimensions.get("is_total_row")):
        return True
    text = " ".join(
        str(value)
        for value in (
            token.raw_label,
            dimensions.get("row_label"),
            dimensions.get("row_path"),
            token.company_name,
        )
        if value is not None
    )
    normalized = normalize_identifier(text)
    return bool(re.search(r"(?:^|_)total(?:_|$)|(?:^|_)aggregate(?:_|$)|(?:^|_)overall(?:_|$)", normalized))


def _truthy_dimension(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _query_requests_aggregate_numerator(
    query: str,
    numerator_field: str,
    numerator_candidates: Sequence[AttributeValueToken],
) -> bool:
    if _query_mentions_nonaggregate_row(query, numerator_candidates):
        return False
    normalized_query = normalize_identifier(query).replace("_", " ")
    for surface in _field_surface_candidates(numerator_field):
        if _total_prefix_window_matches_field(normalized_query, surface):
            return True
        normalized_surface = normalize_identifier(surface).replace("_", " ")
        if not normalized_surface:
            continue
        if re.search(rf"\btotal\s+{re.escape(normalized_surface)}\b", normalized_query):
            return True
    return False


def _total_prefix_window_matches_field(normalized_query: str, field_surface: str) -> bool:
    field_terms = [
        term
        for term in normalize_identifier(field_surface).replace("_", " ").split()
        if term not in _SHARE_QUALIFIER_STOPWORDS and term not in {"total", "amount"}
    ]
    if not field_terms:
        return False
    required = min(2, len(field_terms))
    query_terms = normalized_query.split()
    for index, term in enumerate(query_terms):
        if term != "total":
            continue
        window = set(query_terms[index + 1 : index + 7])
        if len(set(field_terms) & window) >= required:
            return True
    return False


def _query_mentions_nonaggregate_row(
    query: str,
    candidates: Sequence[AttributeValueToken],
) -> bool:
    normalized_query = normalize_identifier(query).replace("_", " ")
    for token in candidates:
        if _token_is_aggregate_like(token):
            continue
        for value in (
            token.raw_label,
            (token.dimensions or {}).get("row_label"),
            token.company_name,
        ):
            if not value:
                continue
            normalized_value = normalize_identifier(str(value)).replace("_", " ")
            if normalized_value and re.search(rf"\b{re.escape(normalized_value)}\b", normalized_query):
                return True
    return False


_SHARE_QUALIFIER_STOPWORDS = {
    "a",
    "an",
    "are",
    "as",
    "at",
    "by",
    "december",
    "for",
    "is",
    "of",
    "on",
    "portion",
    "share",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "year",
}


def _token_primary_descriptor(token: AttributeValueToken) -> str:
    parts = [
        token.field_label,
        token.raw_label,
        " ".join(str(value) for value in (token.dimensions or {}).values()),
    ]
    return " ".join(part for part in parts if part)


def _token_secondary_descriptor(token: AttributeValueToken) -> str:
    source_text = token.source.text_excerpt if token.source is not None else None
    parts = [
        token.field_name.replace("_", " "),
        token.company_name,
        source_text,
    ]
    return " ".join(str(part) for part in parts if part)


def _normalized_word_set(text: str) -> set[str]:
    normalized = normalize_identifier(text).replace("_", " ")
    return {term for term in normalized.split() if term}


def _resolve_explicit_percentage_share_fields(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, str, float, float] | None:
    match = re.search(
        r"(?P<numerator>.+?)\bas\s+a\s+percentage\s+of\s+(?P<denominator>.+)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None

    numerator_text = _strip_share_question_prefix(match.group("numerator"))
    denominator_text = _strip_share_question_suffix(match.group("denominator"))
    if not numerator_text or not denominator_text:
        return None

    numerator_mentions = field_mentions(numerator_text, fields)
    denominator_mentions = field_mentions(denominator_text, fields)
    if numerator_mentions:
        numerator = numerator_mentions[0][2]
        numerator_conf = 0.95
    else:
        numerator_grounding = grounder.ground(numerator_text, fields)
        numerator = numerator_grounding.field_name
        numerator_conf = numerator_grounding.confidence

    if denominator_mentions:
        total_denominator = next(
            (field for _, _, field, _ in denominator_mentions if _is_total_like_field(field)),
            None,
        )
        denominator = total_denominator or denominator_mentions[0][2]
        denominator_conf = 0.95 if total_denominator is not None else 0.9
    else:
        denominator_grounding = grounder.ground(denominator_text, fields)
        denominator = denominator_grounding.field_name
        denominator_conf = denominator_grounding.confidence
    semantic_denominator = _semantic_total_denominator_field(denominator_text, fields)
    if semantic_denominator is not None:
        denominator = semantic_denominator
        denominator_conf = max(denominator_conf, 0.9)

    if numerator == denominator:
        return None
    return numerator, denominator, numerator_conf, denominator_conf


def _strip_share_question_prefix(text: str) -> str:
    cleaned = re.sub(
        r"^\s*(?:what\s+(?:is|was)\s+)?(?:the\s+)?(?:amount\s+of\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ?.,")


def _strip_share_question_suffix(text: str) -> str:
    cleaned = text.strip(" ?.,")
    cleaned = re.sub(r"^\s*total\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _semantic_total_denominator_field(text: str, fields: Sequence[str]) -> str | None:
    normalized = normalize_identifier(text)
    if "remuneration" in normalized and "total_amount" in fields:
        return "total_amount"
    if "amount" in normalized and "total_amount" in fields:
        return "total_amount"
    return None


def resolve_margin_fields(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> tuple[str, str, float, float]:
    candidate_fields = [
        field for field in fields
        if not re.search(r"(?:^|_)(?:margin|growth|rate|turnover)(?:_|$)", field)
    ]
    if "revenue" not in candidate_fields and "revenue" in fields:
        candidate_fields.append("revenue")

    mentions = field_mentions(query, candidate_fields)
    if len(mentions) >= 2:
        return mentions[0][2], mentions[1][2], 0.95, 0.95
    if len(mentions) == 1 and mentions[0][2] != "revenue" and "revenue" in fields:
        return mentions[0][2], "revenue", 0.9, 0.75

    profit_candidates = (
        "net_profit",
        "operating_profit",
        "gross_profit",
        "profit",
        "ebitda",
    )
    numerator = next((field for field in profit_candidates if field in candidate_fields), None)
    denominator = "revenue" if "revenue" in candidate_fields else None
    if numerator is not None and denominator is not None and numerator != denominator:
        return numerator, denominator, 0.75, 0.75
    return resolve_ratio_fields(query, fields, grounder)


def field_mentions(query: str, fields: Sequence[str]) -> list[tuple[int, int, str, str]]:
    normalized_query = normalize_identifier(query).replace("_", "")
    best_by_field: dict[str, tuple[int, int, str, str]] = {}
    for field in fields:
        aliases = (field, field.replace("_", " "), *field_aliases(field))
        for alias in aliases:
            normalized_alias = normalize_identifier(alias).replace("_", "")
            if not normalized_alias:
                continue
            position = normalized_query.find(normalized_alias)
            if position < 0:
                continue
            hit = (position, -len(normalized_alias), field, alias)
            current = best_by_field.get(field)
            if current is None or hit < current:
                best_by_field[field] = hit
    return sorted(best_by_field.values())


def resolve_year(query: str) -> int | None:
    years = resolve_years(query)
    return years[0] if years else None


def resolve_year_for_operator(query: str, operator: str | None) -> int | None:
    years = resolve_years(query)
    if not years:
        return None
    canonical = "GROWTH" if operator == "PERCENT_CHANGE" else operator
    if canonical in {"GROWTH", "TREND"}:
        return None
    if canonical == "YEAR_LIST":
        return None
    if canonical in {"COUNT", "SUM", "AVG", "MAX", "MIN", "ARGMAX", "ARGMIN", "TOP_K", "RANK"}:
        if _query_requests_cross_year_set(query) or (
            canonical == "AVG" and _query_mentions_parallel_years(query)
        ) or (
            canonical == "SUM" and len(years) >= 3 and _query_mentions_parallel_years(query)
        ):
            return None
        return years[-1]
    if canonical == "LOOKUP" and re.search(r"\bas\s+of\b", query, re.IGNORECASE):
        return years[-1]
    return years[0]


def resolve_years(query: str) -> list[int]:
    years: list[int] = []
    for match in YEAR_RE.finditer(query):
        start, end = match.span(1)
        if end < len(query) - 1 and query[end] == "." and query[end + 1].isdigit():
            continue
        if start > 0 and query[start - 1] == ".":
            continue
        if re.search(r"\bdue\s+$|\bdue\s+(?:19|20)\d{2}(?:\s*(?:,|and)\s*)$", query[:start], re.IGNORECASE):
            continue
        years.append(int(match.group(1)))
    for match in FY_RE.finditer(query):
        year = int(match.group(1))
        years.append(2000 + year if year < 70 else 1900 + year)
    years = list(dict.fromkeys(years))
    return years


def resolve_change_direction_years(query: str) -> tuple[int, int] | None:
    match = re.search(
        r"\b(?:in|for|during)\s+(19\d{2}|20\d{2})\b[^.?!]{0,80}\bcompar(?:e|ed|ing)?\s+(?:to|with)\s+(19\d{2}|20\d{2})\b",
        query,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(2)), int(match.group(1))
    match = re.search(
        r"\bbetween\s+(19\d{2}|20\d{2})\b\s+and\s+(19\d{2}|20\d{2})\b",
        query,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(
        r"\bin\s+(19\d{2}|20\d{2})\b.*\bfrom\s+(19\d{2}|20\d{2})\b",
        query,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(2)), int(match.group(1))
    match = re.search(
        r"\bfrom\s+(?:[A-Za-z\s,]+)?(19\d{2}|20\d{2})\b.*\bto\s+(19\d{2}|20\d{2})\b",
        query,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def resolve_fiscal_period_end_years(query: str) -> list[int]:
    years: list[int] = []
    for match in re.finditer(
        r"\b(19\d{2}|20\d{2})\s*/\s*(19\d{2}|20\d{2})\b",
        query,
    ):
        years.append(int(match.group(2)))
    return years


def _query_requests_cross_year_set(query: str) -> bool:
    return bool(
        re.search(
            r"从\s*20\d{2}\s*(?:到|至|-|–|—)\s*20\d{2}"
            r"|between\s+20\d{2}\s+and\s+20\d{2}"
            r"|20\d{2}\s*(?:-|–|—|to|至|到)\s*20\d{2}"
            r"|across\s+years|over\s+years|各年|所有年份|期间",
            query,
            re.IGNORECASE,
        )
    )


def _query_mentions_parallel_years(query: str) -> bool:
    return bool(re.search(r"\b20\d{2}\b\s+and\s+\b20\d{2}\b", query, re.IGNORECASE))


def resolve_last_year(query: str) -> int | None:
    years = resolve_years(query)
    range_members = {
        year
        for pair in re.findall(r"(20\d{2})\s*(?:[-–—至到]|to)\s*(20\d{2})", query, re.IGNORECASE)
        for year in pair
    }
    candidates = [year for year in years if str(year) not in range_members]
    if candidates:
        return candidates[-1]
    return years[-1] if years else None


def resolve_two_years(query: str) -> tuple[int | None, int | None]:
    matches = YEAR_RE.findall(query)
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    if len(matches) == 1:
        return int(matches[0]), None
    return None, None


def resolve_industry(query: str, graph: AttributeValueGraph) -> str | None:
    normalized_query = normalize_identifier(query).replace("_", "")
    matches: list[tuple[int, str]] = []
    for industry in graph.industries:
        normalized_industry = normalize_identifier(industry).replace("_", "")
        if normalized_industry and normalized_industry in normalized_query:
            matches.append((len(normalized_industry), industry))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    if len(graph.industries) == 1 and not _has_unmatched_industry_hint(query, graph):
        return graph.industries[0]
    return None


def resolve_entity(
    query: str,
    graph: AttributeValueGraph,
    industry: object | None = None,
) -> str | None:
    query_lower = query.lower()
    query_terms = _entity_query_terms(query)
    if industry is not None:
        query_terms = query_terms - _entity_terms(str(industry))
    candidates: list[str] = []
    partial_candidates: list[str] = []
    seen: set[str] = set()
    for token in graph.tokens:
        if industry is not None and token.industry != str(industry):
            continue
        name = token.company_name
        if name in seen:
            continue
        seen.add(name)
        if name.lower() in query_lower or _entity_name_matches_query(name, query_lower, query_terms):
            candidates.append(name)
        elif _entity_terms(name) & query_terms:
            partial_candidates.append(name)
    if not candidates and industry is not None and len(partial_candidates) == 1:
        return partial_candidates[0]
    return max(candidates, key=len) if candidates else None


def resolve_entity_group(
    query: str,
    graph: AttributeValueGraph,
    industry: object | None = None,
) -> list[str]:
    query_terms = _entity_query_terms(query)
    if industry is not None:
        query_terms = query_terms - _entity_terms(str(industry))
    if not query_terms:
        return []
    exact = resolve_entity(query, graph, industry)
    if exact is not None:
        return []
    entities: list[str] = []
    seen: set[str] = set()
    for token in graph.tokens:
        if industry is not None and token.industry != str(industry):
            continue
        name = token.company_name
        if name in seen:
            continue
        seen.add(name)
        terms = _entity_terms(name)
        if terms and terms & query_terms:
            entities.append(name)
    return entities if len(entities) >= 2 else []


def resolve_two_entities(
    query: str,
    graph: AttributeValueGraph,
    industry: object | None = None,
) -> tuple[str | None, str | None]:
    query_lower = query.lower()
    query_terms = _entity_query_terms(query)
    if industry is not None:
        query_terms = query_terms - _entity_terms(str(industry))
    candidates: list[str] = []
    seen: set[str] = set()
    for token in graph.tokens:
        if industry is not None and token.industry != str(industry):
            continue
        name = token.company_name
        if name in seen:
            continue
        seen.add(name)
        if name.lower() in query_lower or _entity_name_matches_query(name, query_lower, query_terms):
            candidates.append(name)
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    if candidates:
        return candidates[0], None
    return None, None


def _entity_query_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if (len(term) >= 3 or term.isdigit()) and term not in _ENTITY_STOPWORDS
    }


def _entity_name_matches_query(name: str, query_lower: str, query_terms: set[str]) -> bool:
    normalized_name = normalize_identifier(name).replace("_", "")
    normalized_query = normalize_identifier(query_lower).replace("_", "")
    if normalized_name and normalized_name in normalized_query:
        return True
    terms = [
        term
        for term in re.findall(r"[a-zA-Z0-9]+", name.lower())
        if (len(term) >= 3 or term.isdigit()) and term not in _ENTITY_STOPWORDS
    ]
    if not terms:
        return False
    overlap = [term for term in terms if term in query_terms]
    if not overlap:
        return False
    required = min(2, len(terms))
    return len(overlap) >= required or (len(terms) == 1 and len(overlap[0]) >= 6)


def _has_unmatched_industry_hint(query: str, graph: AttributeValueGraph) -> bool:
    normalized_industries = {
        normalize_identifier(industry).replace("_", "")
        for industry in graph.industries
    }
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9&_-]*)\s+(?:industry|sector|segment)\b",
        query,
        re.IGNORECASE,
    ):
        if normalize_identifier(match.group(1)).replace("_", "") not in normalized_industries:
            return True
    if _entity_query_terms(query) and not re.search(
        r"\b(?:total|sum|aggregate|count|how many|industry|sector|companies|firms)\b",
        query,
        re.IGNORECASE,
    ):
        return False
    for match in re.finditer(r"\b[A-Z][A-Z0-9&_-]{2,}\b", query):
        candidate = normalize_identifier(match.group(0))
        if candidate in {"usd", "eur", "cny", "rmb", "fy", "eps", "gdp"}:
            continue
        if candidate.replace("_", "") not in normalized_industries:
            return True
    return False


_ENTITY_STOPWORDS = {
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




def extract_k(query: str) -> int:
    chinese_digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    chinese_match = re.search(r"(?:前|排名前)\s*([一二两三四五六七八九十])", query)
    if chinese_match:
        return chinese_digits[chinese_match.group(1)]
    match = re.search(
        r"(?:排名前\s*|前\s*|[Tt]op\s*)(\d+)|(\d+)\s*(?:家公司|家企业|companies|firms|entities)",
        query,
    )
    if match:
        return int(next(group for group in match.groups() if group is not None))
    return 3


def extract_order(query: str) -> str:
    if re.search(r"(最低|最小|最少|倒数|bottom|lowest|smallest|least|ascending|升序)", query, re.IGNORECASE):
        return "ascending"
    return "descending"


def _composite_order(routing: RoutingResult) -> str:
    steps = (routing.trace or {}).get("steps", [])
    if isinstance(steps, list) and steps:
        order = steps[0].get("order") if isinstance(steps[0], dict) else None
        if order in {"ascending", "descending"}:
            return str(order)
    return "descending"


def _rank_field(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
) -> FieldGroundingResult:
    mentions = field_mentions(query, fields)
    if mentions:
        return FieldGroundingResult(
            field_name=mentions[0][2],
            confidence=0.95,
            scores={mentions[0][2]: 0.95},
            trace={"source": "first_field_mention_for_rank"},
        )
    return grounder.ground(query, fields)


def _margin_fields_after_rank(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
    rank_field: str,
) -> tuple[str, str, float, float]:
    candidate_fields = [
        field for field in fields
        if not re.search(r"(?:^|_)(?:margin|growth|rate|turnover)(?:_|$)", field)
    ]
    if rank_field not in candidate_fields and rank_field in fields:
        candidate_fields.append(rank_field)
    mentions = field_mentions(query, candidate_fields)
    mentioned = [field for _, _, field, _ in mentions]
    profit_mentions = [
        field
        for field in mentioned
        if field != rank_field and re.search(r"(profit|利润|收益|ebitda)", field, re.IGNORECASE)
    ]
    if profit_mentions:
        denominator = rank_field if rank_field in candidate_fields else (
            "revenue" if "revenue" in candidate_fields else profit_mentions[0]
        )
        return profit_mentions[0], denominator, 0.95, 0.9
    if "net_profit" in candidate_fields and "revenue" in candidate_fields:
        return "net_profit", "revenue", 0.75, 0.75
    if "operating_profit" in candidate_fields and "revenue" in candidate_fields:
        return "operating_profit", "revenue", 0.72, 0.75
    numerator, denominator, num_conf, den_conf = resolve_margin_fields(query, fields, grounder)
    if numerator == denominator and numerator == rank_field:
        fallback_num = next((field for field in candidate_fields if field != rank_field), numerator)
        numerator = fallback_num
        num_conf = min(num_conf, 0.5)
    return numerator, denominator, num_conf, den_conf


def _sum_field_after_topk(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
    rank_field: str,
) -> FieldGroundingResult:
    mentions = field_mentions(query, fields)
    if len(mentions) >= 2:
        return FieldGroundingResult(
            field_name=mentions[-1][2],
            confidence=0.95,
            scores={mentions[-1][2]: 0.95},
            trace={"source": "last_field_mention_for_topk_sum"},
        )
    return FieldGroundingResult(
        field_name=rank_field,
        confidence=0.7,
        scores={rank_field: 0.7},
        trace={"source": "fallback_rank_field_for_topk_sum"},
    )


def _lookup_field_after_rank(
    query: str,
    fields: Sequence[str],
    grounder: FieldGrounder,
    rank_field: str,
) -> FieldGroundingResult:
    mentions = field_mentions(query, fields)
    if len(mentions) >= 2:
        return FieldGroundingResult(
            field_name=mentions[-1][2],
            confidence=0.95,
            scores={mentions[-1][2]: 0.95},
            trace={"source": "last_field_mention_for_rank_lookup"},
        )
    grounded = grounder.ground(query, fields)
    if grounded.field_name != rank_field:
        return grounded
    return FieldGroundingResult(
        field_name=rank_field,
        confidence=0.7,
        scores={rank_field: 0.7},
        trace={"source": "fallback_rank_field_for_rank_lookup"},
    )


def _slot_grounded(slots: dict[str, Slot], key: str) -> object | None:
    value = slots.get(key)
    if isinstance(value, Slot):
        return value.grounded_value
    return value


def _asks_for_year_output(query: str) -> bool:
    normalized = query.lower()
    return bool(
        re.search(r"\bwhich\s+year\b|\bwhat\s+year\b|\byear\s+had\b", normalized)
        or "哪一年" in query
        or "哪个年份" in query
    )


def _asks_for_entity_output(query: str) -> bool:
    normalized = query.lower()
    return bool(
        re.search(r"\bwho\b", normalized)
        or re.search(r"\bwhich\s+(?:segment|company|entity|person|director|business|division)\b", normalized)
        or re.search(r"\bwhich\b.*\b(?:director|officer|employee|person|name)\b", normalized)
        or "谁" in query
    )


def _unit_after_number(query: str, start: int) -> str | None:
    tail = query[start:start + 40].strip()
    match = re.match(
        r"(?:\s*(million|billion|thousand))?\s*(usd|eur|cny|rmb|dollars?)\b",
        tail,
        re.IGNORECASE,
    )
    if not match:
        return None
    scale = (match.group(1) or "").lower()
    currency = match.group(2).upper()
    if currency == "DOLLARS":
        currency = "USD"
    return f"{scale} {currency}".strip()


def _textual_threshold(text: str) -> tuple[float, str | None] | None:
    lowered = text.lower()
    if re.search(r"\ba\s+billion\b|\bone\s+billion\b|\bbillion\s+dollars?\b", lowered):
        return 1.0, "billion USD"
    if re.search(r"\ba\s+million\b|\bone\s+million\b|\bmillion\s+dollars?\b", lowered):
        return 1.0, "million USD"
    return None


def _dimension_values(graph: AttributeValueGraph) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for token in graph.tokens:
        for key, value in (token.dimensions or {}).items():
            if value is None:
                continue
            text = str(value)
            if not text:
                continue
            bucket = values.setdefault(str(key), [])
            if text not in bucket:
                bucket.append(text)
    return values


def _entity_terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if (len(term) >= 3 or term.isdigit()) and term not in _ENTITY_STOPWORDS
    }


def _slot(surface: str, value: object, confidence: float) -> Slot:
    return Slot(surface=surface, grounded_value=value, confidence=confidence)
