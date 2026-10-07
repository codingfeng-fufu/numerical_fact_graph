from __future__ import annotations

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource
from graph_numeric.core.unit_resolver import OPERATION_COMPATIBILITY
from graph_numeric.operators.executor import execute, execute_composite
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot
from graph_numeric.operators.operator_registry import (
    EXECUTOR_OPERATOR_ALIASES,
    OPERATOR_REGISTRY,
    PUBLIC_OPERATORS,
)
from graph_numeric.operators.operator_solvers import solve_composite_operator_plan, solve_operator_plan
from graph_numeric.learning.router import RuleBasedRouter


def _slot(surface: str, value: object, conf: float = 0.9) -> Slot:
    return Slot(surface=surface, grounded_value=value, confidence=conf)


def _make_graph() -> AttributeValueGraph:
    values = {
        "Alpha": {
            2021: {"revenue": 700, "net_profit": 70, "employees": 40},
            2022: {"revenue": 900, "net_profit": 90, "employees": 45},
            2023: {"revenue": 1000, "net_profit": 100, "employees": 50},
            2024: {"revenue": 1200, "net_profit": 120, "employees": 55},
        },
        "Beta": {
            2021: {"revenue": 600, "net_profit": 50, "employees": 25},
            2022: {"revenue": 700, "net_profit": 60, "employees": 28},
            2023: {"revenue": 800, "net_profit": 80, "employees": 30},
            2024: {"revenue": 950, "net_profit": 95, "employees": 35},
        },
        "Gamma": {
            2021: {"revenue": 2400, "net_profit": 250, "employees": 180},
            2022: {"revenue": 2600, "net_profit": 260, "employees": 190},
            2023: {"revenue": 3000, "net_profit": 300, "employees": 200},
            2024: {"revenue": 3200, "net_profit": 320, "employees": 210},
        },
    }
    tokens: list[AttributeValueToken] = []
    for company, by_year in values.items():
        industry = "TECH" if company != "Gamma" else "FINANCE"
        for year, fields in by_year.items():
            entity = f"{company}:{year}"
            for field, value in fields.items():
                unit = "million USD" if field in {"revenue", "net_profit"} else "count"
                tokens.append(
                    AttributeValueToken(
                        token_id=f"{entity}:{field}",
                        entity_id=entity,
                        company_name=company,
                        field_name=field,
                        field_label=field.replace("_", " "),
                        value=float(value),
                        year=year,
                        industry=industry,
                        unit=unit,
                    )
                )
    return AttributeValueGraph(tuple(tokens), source_name="full_ops")


def _route_solve_execute(query: str):
    graph = _make_graph()
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)
    return graph, routing, plan, result


def test_public_registry_contains_full_design_operator_set() -> None:
    expected = {
        "LOOKUP",
        "SUM",
        "COUNT",
        "AVG",
        "MIN",
        "MAX",
        "ARGMIN",
        "ARGMAX",
        "RATIO",
        "SHARE",
        "MARGIN",
        "GROWTH",
        "PERCENT_CHANGE",
        "DIFFERENCE",
        "TOP_K",
        "RANK",
        "COMPARE",
        "TREND",
        "PREDICT",
        "FORECAST",
        "PRODUCT",
    }
    assert expected.issubset(set(PUBLIC_OPERATORS))
    assert OPERATOR_REGISTRY.canonical_executor_operator("SHARE") == "RATIO"
    assert OPERATOR_REGISTRY.canonical_executor_operator("PRODUCT") == "PRODUCT"
    assert EXECUTOR_OPERATOR_ALIASES["FORECAST"] == "PREDICT"
    assert OPERATION_COMPATIBILITY["PRODUCT"] >= {"money", "shares", "per_share", "count"}


def test_graph_api_exposes_filter_lookup_profiles_and_units() -> None:
    graph = _make_graph()
    tech_2024 = graph.filter_records(
        field="revenue",
        filters={"year": 2024, "industry": "TECH"},
    )
    assert {token.company_name for token in tech_2024} == {"Alpha", "Beta"}

    alpha_lookup = graph.lookup("net_profit", {"entity": "Alpha", "year": 2024})
    assert len(alpha_lookup) == 1
    assert alpha_lookup[0].unit == "million USD"

    assert "revenue" in graph.get_candidate_fields()
    profile = graph.get_field_profile("revenue")
    assert profile["count"] == 12
    assert profile["unit"] == "million USD"
    assert len(graph.get_records_by_entity("Beta")) == 12


def test_argmin_routes_solves_and_executes() -> None:
    graph, routing, plan, result = _route_solve_execute("2024 年 revenue 最低的公司是哪家")
    assert routing.operator == "ARGMIN"
    assert plan.operator == "ARGMIN"
    assert result.answer == 950.0
    assert result.selected_tokens[0].company_name == "Beta"
    assert result.checks["filter_satisfied"] is True
    assert graph is not None


def test_margin_routes_as_margin_and_executes_ratio_semantics() -> None:
    _, routing, plan, result = _route_solve_execute("Alpha 2024 年的净利润率是多少")
    assert routing.operator == "MARGIN"
    assert plan.operator == "MARGIN"
    assert plan.slots["numerator_field"].grounded_value == "net_profit"
    assert plan.slots["denominator_field"].grounded_value == "revenue"
    assert abs(result.answer - (120 / 1200)) < 1e-9
    assert result.checks["arithmetic_verified"] is True


def test_share_routes_as_share_and_executes_ratio_semantics() -> None:
    _, routing, plan, result = _route_solve_execute("2024 年 TECH 行业 net_profit 占 revenue 的占比")
    assert routing.operator == "SHARE"
    assert plan.operator == "SHARE"
    assert abs(result.answer - ((120 + 95) / (1200 + 950))) < 1e-9
    assert result.checks["unit_compatible"] is True


def test_product_executes_selected_token_multiplication_with_compound_unit() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="award:shares",
                entity_id="award",
                company_name="Alpha",
                field_name="shares_granted",
                field_label="Shares granted",
                value=3063816.0,
                unit="shares",
            ),
            AttributeValueToken(
                token_id="award:price",
                entity_id="award",
                company_name="Alpha",
                field_name="average_price",
                field_label="Average price",
                value=4.0,
                unit="USD per share",
            ),
        ),
        source_name="product_operator",
    )
    plan = OperatorPlan(
        operator="PRODUCT",
        slots={
            "factor_token_ids": _slot(
                "shares times price",
                ["award:shares", "award:price"],
                1.0,
            ),
        },
    )

    result = execute(graph, plan)

    assert result.answer == 12255264.0
    assert result.calculation == "3063816.0 * 4.0 = 12255264.0"
    assert result.metadata["executor_operator"] == "PRODUCT"
    assert result.metadata["output_unit"] == "USD"
    assert result.checks["arithmetic_verified"] is True
    assert result.checks["unit_compatible"] is True


def test_product_uses_question_constant_slot_without_polluting_graph() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="dividend:quarterly",
                entity_id="dividend",
                company_name="Alpha",
                field_name="quarterly_dividend",
                field_label="Quarterly dividend",
                value=0.25,
                source=TokenSource(document_id="doc", text_excerpt="quarterly dividend was $0.25"),
                unit="USD per share",
            ),
        ),
        source_name="product_question_constant",
    )
    routing = RuleBasedRouter().route("What was the total dividend for all four quarters?")
    plan = solve_operator_plan(
        "What was the total dividend for all four quarters?",
        graph,
        routing,
        operator="PRODUCT",
    )

    result = execute(graph, plan)

    assert plan.slots["constant_slot"].grounded_value == [
        {
            "surface": "all four quarters",
            "value": 4.0,
            "source": "question_text",
            "role": "multiplicative_factor",
        }
    ]
    assert len(graph.tokens) == 1
    assert result.answer == 1.0
    assert result.metadata["question_constant_factors"] == [
        {
            "surface": "all four quarters",
            "value": 4.0,
            "source": "question_text",
            "role": "multiplicative_factor",
        }
    ]
    assert [token.token_id for token in result.selected_tokens] == [
        "dividend:quarterly",
        "question_constant:all_four_quarters",
    ]


def test_percent_change_is_growth_alias() -> None:
    _, routing, plan, result = _route_solve_execute("Alpha revenue 从 2023 到 2024 的 percent change")
    assert routing.operator == "PERCENT_CHANGE"
    assert plan.operator == "PERCENT_CHANGE"
    assert abs(result.answer - 0.2) < 1e-9
    assert result.metadata["output_unit"] == "ratio_dimensionless"
    assert result.checks["arithmetic_verified"] is True
    assert {token.year for token in result.selected_tokens} == {2023, 2024}


def test_percent_change_of_percentage_rate_returns_ratio_not_percentage_points() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="alpha:2023:discount_rate",
                entity_id="alpha",
                company_name="Alpha",
                field_name="discount_rate",
                field_label="Discount rate",
                value=24.0,
                year=2023,
                unit="percent",
            ),
            AttributeValueToken(
                token_id="alpha:2024:discount_rate",
                entity_id="alpha",
                company_name="Alpha",
                field_name="discount_rate",
                field_label="Discount rate",
                value=22.1,
                year=2024,
                unit="percent",
            ),
        ),
        source_name="rate_like",
    )
    plan = OperatorPlan(
        operator="PERCENT_CHANGE",
        slots={
            "target_field": _slot("discount rate", "discount_rate"),
            "from_time": _slot("2023", 2023),
            "to_time": _slot("2024", 2024),
            "entity": _slot("Alpha", "Alpha"),
        },
    )

    result = execute(graph, plan)

    assert result.answer == (22.1 - 24.0) / 24.0
    assert result.metadata["output_unit"] == "ratio_dimensionless"
    assert result.checks["arithmetic_verified"] is True


def test_rank_routes_and_executes_top_k_semantics() -> None:
    _, routing, plan, result = _route_solve_execute("2024 年 revenue 排名前 2 的公司")
    assert routing.operator == "RANK"
    assert plan.operator == "RANK"
    assert result.answer == 1200.0
    assert [token.company_name for token in result.selected_tokens] == ["Gamma", "Alpha"]


def test_compare_routes_and_executes_difference_semantics() -> None:
    _, routing, plan, result = _route_solve_execute("比较 Alpha 和 Beta 2024 年 revenue 哪个更高")
    assert routing.operator == "COMPARE"
    assert plan.operator == "COMPARE"
    assert result.answer == 250.0
    assert {token.company_name for token in result.selected_tokens} == {"Alpha", "Beta"}


def test_compare_between_dimension_values_returns_gap() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="north:alpha:revenue",
                entity_id="Alpha",
                company_name="Alpha",
                field_name="revenue",
                field_label="Revenue",
                value=1200.0,
                year=2024,
                dimensions={"document_id": "north"},
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="south:alpha:revenue",
                entity_id="Alpha",
                company_name="Alpha",
                field_name="revenue",
                field_label="Revenue",
                value=1180.0,
                year=2024,
                dimensions={"document_id": "south"},
                unit="million USD",
            ),
        ),
        source_name="compare_dimension_values",
    )
    plan = OperatorPlan(
        operator="COMPARE",
        slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
            "entity": _slot("Alpha", "Alpha"),
            "left_dimension_filter": _slot("south", {"document_id": "south"}),
            "right_dimension_filter": _slot("north", {"document_id": "north"}),
            "comparison_type": _slot("difference", "difference"),
        },
    )

    result = execute(graph, plan)

    assert result.answer == 20.0
    assert result.metadata["difference_mode"] == "absolute_gap"


def test_forecast_routes_and_executes_predict_semantics() -> None:
    _, routing, plan, result = _route_solve_execute("Forecast Alpha 2025 revenue")
    assert routing.operator == "FORECAST"
    assert plan.operator == "FORECAST"
    assert plan.slots["horizon"].grounded_value == 2025
    assert abs(result.answer - 1338.0) < 1e-6
    assert result.metadata["method"] == "configured_auto"
    assert result.metadata["forecast_profile"] == "recent_growth_risk_t0.22_w0.5"
    assert result.metadata["selected_method"] == "linear_naive_rules:w=0.920"
    assert result.checks["output_type_valid"] is True


def test_predict_executor_respects_linear_method_override() -> None:
    graph = _make_graph()
    plan = OperatorPlan(
        operator="PREDICT",
        slots={
            "target_field": _slot("revenue", "revenue"),
            "entity": _slot("Alpha", "Alpha"),
            "history_range": _slot("2021-2024", "2021-2024"),
            "horizon": _slot("2025", 2025),
            "method": _slot("linear", "linear"),
        },
    )
    result = execute(graph, plan)

    assert abs(result.answer - 1350.0) < 1e-6
    assert result.metadata["method"] == "linear"
    assert result.metadata["selected_method"] == "linear"


def test_predict_executor_accepts_selector_config_slot() -> None:
    graph = _make_graph()
    plan = OperatorPlan(
        operator="PREDICT",
        slots={
            "target_field": _slot("revenue", "revenue"),
            "entity": _slot("Alpha", "Alpha"),
            "history_range": _slot("2021-2024", "2021-2024"),
            "horizon": _slot("2025", 2025),
            "method": _slot("selector", "selector"),
            "selector_config": _slot(
                "linear-naive blend",
                {
                    "selector_type": "linear_naive_rules",
                    "base_weight": 0.5,
                },
            ),
        },
    )
    result = execute(graph, plan)

    assert abs(result.answer - 1275.0) < 1e-6
    assert result.metadata["method"] == "configured_auto"
    assert result.metadata["selector_config"]["base_weight"] == 0.5


def test_composite_operator_plan_validates_step_references() -> None:
    step1 = OperatorPlan(
        operator="ARGMAX",
        slots={
            "target_entity_type": _slot("company", "company"),
            "target_field": _slot("revenue", "revenue"),
        },
        trace={"step_id": "s1"},
    )
    step2 = OperatorPlan(
        operator="MARGIN",
        slots={
            "numerator_field": _slot("net_profit", "net_profit"),
            "denominator_field": _slot("revenue", "revenue"),
        },
        depends_on={"entity": "$s1.entity"},
        trace={"step_id": "s2"},
    )
    composite = CompositeOperatorPlan(steps=(step1, step2))
    assert composite.is_valid()
    assert composite.to_dict()["steps"][0]["operator"] == "ARGMAX"


def test_execute_composite_argmax_then_margin_binds_entity() -> None:
    graph = _make_graph()
    step1 = OperatorPlan(
        operator="ARGMAX",
        slots={
            "target_entity_type": _slot("company", "company"),
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        },
        trace={"step_id": "s1"},
    )
    step2 = OperatorPlan(
        operator="MARGIN",
        slots={
            "numerator_field": _slot("net_profit", "net_profit"),
            "denominator_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        },
        depends_on={"entity": "$s1.entity"},
        trace={"step_id": "s2"},
    )

    result = execute_composite(graph, CompositeOperatorPlan(steps=(step1, step2)))

    assert abs(result.answer - 0.1) < 1e-9
    assert result.checks["steps_verified"] is True
    assert result.metadata["dependency_bindings"]["s2"]["entity"]["value"] == "Gamma"
    assert result.metadata["final_step_id"] == "s2"
    assert {token.company_name for token in result.selected_tokens} == {"Gamma"}


def test_execute_composite_top_k_then_sum_binds_entity_set() -> None:
    graph = _make_graph()
    step1 = OperatorPlan(
        operator="TOP_K",
        slots={
            "target_entity_type": _slot("company", "company"),
            "target_field": _slot("revenue", "revenue"),
            "k": _slot("2", 2),
            "year": _slot("2024", 2024),
            "order": _slot("descending", "descending"),
        },
        trace={"step_id": "s1"},
    )
    step2 = OperatorPlan(
        operator="SUM",
        slots={
            "target_field": _slot("net_profit", "net_profit"),
            "year": _slot("2024", 2024),
        },
        depends_on={"entities": "$s1.entities"},
        trace={"step_id": "s2"},
    )

    result = execute_composite(graph, CompositeOperatorPlan(steps=(step1, step2)))

    assert result.answer == 440.0
    assert result.metadata["dependency_bindings"]["s2"]["entities"]["value"] == ["Gamma", "Alpha"]
    assert {
        record["company_name"]
        for record in result.metadata["steps"][1]["records_used"]
    } == {"Gamma", "Alpha"}
    assert {token.company_name for token in result.selected_tokens} == {"Gamma", "Alpha"}


def test_unit_normalization_sum_ratio_and_ranking() -> None:
    tokens = (
        AttributeValueToken(
            token_id="alpha:2024:revenue",
            entity_id="alpha:2024",
            company_name="Alpha",
            field_name="revenue",
            field_label="revenue",
            value=1.2,
            year=2024,
            unit="billion USD",
        ),
        AttributeValueToken(
            token_id="beta:2024:revenue",
            entity_id="beta:2024",
            company_name="Beta",
            field_name="revenue",
            field_label="revenue",
            value=500.0,
            year=2024,
            unit="million USD",
        ),
        AttributeValueToken(
            token_id="alpha:2024:net_profit",
            entity_id="alpha:2024",
            company_name="Alpha",
            field_name="net_profit",
            field_label="net profit",
            value=120.0,
            year=2024,
            unit="million USD",
        ),
        AttributeValueToken(
            token_id="beta:2024:net_profit",
            entity_id="beta:2024",
            company_name="Beta",
            field_name="net_profit",
            field_label="net profit",
            value=40.0,
            year=2024,
            unit="million USD",
        ),
    )
    graph = AttributeValueGraph(tokens)

    sum_result = execute(
        graph,
        OperatorPlan(
            operator="SUM",
            slots={
                "target_field": _slot("revenue", "revenue"),
                "year": _slot("2024", 2024),
            },
        ),
    )
    assert abs(sum_result.answer - 1.7) < 1e-9
    assert sum_result.metadata["output_unit"] == "billion USD"
    assert sum_result.metadata["unit_normalized"] is True
    assert sum_result.checks["arithmetic_verified"] is True

    ratio_result = execute(
        graph,
        OperatorPlan(
            operator="MARGIN",
            slots={
                "numerator_field": _slot("net_profit", "net_profit"),
                "denominator_field": _slot("revenue", "revenue"),
                "entity": _slot("Alpha", "Alpha"),
                "year": _slot("2024", 2024),
            },
        ),
    )
    assert abs(ratio_result.answer - 0.1) < 1e-9
    assert ratio_result.checks["arithmetic_verified"] is True

    argmax_result = execute(
        graph,
        OperatorPlan(
            operator="ARGMAX",
            slots={
                "target_entity_type": _slot("company", "company"),
                "target_field": _slot("revenue", "revenue"),
                "year": _slot("2024", 2024),
            },
        ),
    )
    assert argmax_result.selected_tokens[0].company_name == "Alpha"
    assert abs(argmax_result.answer - 1.2) < 1e-9


def test_composite_router_solver_executor_for_rank_then_margin_query() -> None:
    graph = _make_graph()
    query = "2024 年 revenue 最高的公司净利润率是多少"
    routing = RuleBasedRouter().route(query)

    assert routing.operator == "COMPOSITE"
    assert routing.route_type == "composite"
    assert routing.trace["composite_case"] == "arg_rank_then_margin"

    plan = solve_composite_operator_plan(query, graph, routing)
    assert plan.is_valid()
    assert [step.operator for step in plan.steps] == ["ARGMAX", "MARGIN"]

    result = execute_composite(graph, plan)
    assert abs(result.answer - 0.1) < 1e-9
    assert result.metadata["dependency_bindings"]["s2"]["entity"]["value"] == "Gamma"


def test_composite_router_solver_executor_for_topk_then_sum_query() -> None:
    graph = _make_graph()
    query = "2024 年 revenue 前 2 名公司的 net_profit 总和是多少"
    routing = RuleBasedRouter().route(query)

    assert routing.operator == "COMPOSITE"
    assert routing.route_type == "composite"
    assert routing.trace["composite_case"] == "topk_then_sum"

    plan = solve_composite_operator_plan(query, graph, routing)
    assert plan.is_valid()
    assert [step.operator for step in plan.steps] == ["TOP_K", "SUM"]

    result = execute_composite(graph, plan)
    assert result.answer == 440.0
