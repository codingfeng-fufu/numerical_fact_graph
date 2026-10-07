from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource
from graph_numeric.operators.executor import ExecutionResult, execute
from graph_numeric.learning.field_grounder import FieldGrounder
from graph_numeric.operators.operator_plan import Slot
from graph_numeric.operators.operator_solvers import OperatorSolver, extract_condition, solve_operator_plan
from graph_numeric.learning.router import RoutingResult, RuleBasedRouter
from graph_numeric.runtime.output_normalization import normalize_execution_output


def _make_graph() -> AttributeValueGraph:
    tokens = []
    values = {
        "Alpha": {
            2023: {"revenue": 1000, "net_profit": 100, "employees": 50},
            2024: {"revenue": 1200, "net_profit": 120, "employees": 55},
        },
        "Beta": {
            2023: {"revenue": 800, "net_profit": 80, "employees": 30},
            2024: {"revenue": 950, "net_profit": 95, "employees": 35},
        },
    }
    for company, by_year in values.items():
        for year, fields in by_year.items():
            entity = f"{company}:{year}"
            for field, value in fields.items():
                tokens.append(
                    AttributeValueToken(
                        token_id=f"{entity}:{field}",
                        entity_id=entity,
                        company_name=company,
                        field_name=field,
                        field_label=field.replace("_", " "),
                        value=float(value),
                        year=year,
                        industry="TECH",
                    )
                )
    return AttributeValueGraph(tuple(tokens), source_name="solver_test")


def _make_ood_entity_graph() -> AttributeValueGraph:
    tokens = []
    values = {
        "Ai Infrastructure Co 1": {
            2022: {"revenue": 100.0, "operating_profit": 40.0},
            2024: {"revenue": 140.0, "operating_profit": 44.0},
        },
        "Ai Infrastructure Co 2": {
            2022: {"revenue": 155.0, "operating_profit": 55.0},
            2024: {"revenue": 253.0, "operating_profit": 66.0},
        },
    }
    for company, by_year in values.items():
        entity_prefix = company.lower().replace(" ", "_")
        for year, fields in by_year.items():
            entity = f"{entity_prefix}:{year}"
            for field, value in fields.items():
                tokens.append(
                    AttributeValueToken(
                        token_id=f"{entity}:{field}",
                        entity_id=entity,
                        company_name=company,
                        field_name=field,
                        field_label=field.replace("_", " "),
                        value=float(value),
                        year=year,
                        industry="AI INFRASTRUCTURE",
                    )
                )
    return AttributeValueGraph(tuple(tokens), source_name="ood_entity_solver_test")


def _value(slot: object) -> object:
    assert isinstance(slot, Slot)
    return slot.grounded_value


def test_extract_condition_supports_symbols_and_chinese() -> None:
    assert extract_condition("revenue >= 1000")[:2] == (">=", 1000.0)
    assert extract_condition("收入 不低于 1000")[:2] == (">=", 1000.0)
    assert extract_condition("net_profit < 100")[:2] == ("<", 100.0)


def test_sum_solver_binds_field_year_and_industry() -> None:
    graph = _make_graph()
    routing = RuleBasedRouter().route("2024 年 TECH 行业公司的 revenue 总和是多少")
    plan = solve_operator_plan("2024 年 TECH 行业公司的 revenue 总和是多少", graph, routing)

    assert plan.operator == "SUM"
    assert _value(plan.slots["target_field"]) == "revenue"
    assert _value(plan.slots["year"]) == 2024
    assert _value(plan.slots["industry"]) == "TECH"
    assert execute(graph, plan).answer == 2150.0


def test_sum_solver_keeps_parallel_year_mentions_on_latest_year() -> None:
    graph = _make_graph()
    query = "What is the total revenue in 2023 and 2024 respectively?"
    routing = RoutingResult(operator="SUM", confidence=1.0, intent="aggregation", route_type="oracle")
    plan = solve_operator_plan(query, graph, routing)

    assert _value(plan.slots["year"]) == 2024
    assert _value(plan.slots["target_field"]) == "revenue"
    assert execute(graph, plan).answer == 2150.0


def test_sum_solver_binds_explicit_parallel_fields() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "mro:crude",
                "mro",
                "MRO",
                "miles_of_private_crude_oil_pipelines",
                "miles of private crude oil pipelines",
                176.0,
            ),
            AttributeValueToken(
                "mro:refined",
                "mro",
                "MRO",
                "miles_of_private_refined_products_pipelines",
                "miles of private refined products pipelines",
                850.0,
            ),
            AttributeValueToken(
                "mro:leased",
                "mro",
                "MRO",
                "leased_common_carrier_refined_product_pipelines",
                "leased common carrier refined product pipelines",
                217.0,
            ),
        ),
        source_name="mro",
    )
    query = "what was total miles of private crude oil pipelines and private refined products pipelines?"
    routing = RoutingResult(operator="SUM", confidence=1.0, intent="aggregation", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert [token.field_name for token in result.selected_tokens] == [
        "miles_of_private_crude_oil_pipelines",
        "miles_of_private_refined_products_pipelines",
    ]
    assert result.answer == 1026.0


def test_share_solver_uses_token_ids_when_same_field_has_qualifiers() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "ecl:cash_total",
                "ecl",
                "ECL",
                "cash_and_cash_equivalents",
                "cash and cash equivalents",
                327.0,
                year=2016,
            ),
            AttributeValueToken(
                "ecl:cash_outside_us",
                "ecl",
                "ECL",
                "cash_and_cash_equivalents",
                "cash and cash equivalents held outside of the u.s.",
                184.0,
                year=2016,
            ),
        ),
        source_name="ecl",
    )
    query = (
        "what portion of cash and cash equivalents on hand are held outside of the u.s. "
        "as of december 31, 2016?"
    )
    routing = RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert result.answer == pytest.approx(184.0 / 327.0)
    assert [token.token_id for token in result.selected_tokens] == [
        "ecl:cash_outside_us",
        "ecl:cash_total",
    ]


def test_share_solver_binds_denominator_to_same_table_aggregate_row() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "tax:federal",
                "tax:2015:federal",
                "Federal",
                "income_taxes",
                "income taxes",
                135.0,
                year=2015,
                source=TokenSource(document_id="doc", table="taxes", row=1, column="2015"),
                dimensions={"row_label": "Federal", "table_id": "taxes"},
            ),
            AttributeValueToken(
                "tax:foreign",
                "tax:2015:foreign",
                "Foreign",
                "income_taxes",
                "income taxes",
                565.0,
                year=2015,
                source=TokenSource(document_id="doc", table="taxes", row=2, column="2015"),
                dimensions={"row_label": "Foreign", "table_id": "taxes"},
            ),
            AttributeValueToken(
                "tax:total",
                "tax:2015:total",
                "Total",
                "income_taxes",
                "income taxes",
                700.0,
                year=2015,
                source=TokenSource(document_id="doc", table="taxes", row=3, column="2015"),
                dimensions={"row_label": "Total", "table_id": "taxes", "is_total_row": "true"},
            ),
            AttributeValueToken(
                "other:total",
                "other:2015:total",
                "Total",
                "income_taxes",
                "income taxes",
                300.0,
                year=2015,
                source=TokenSource(document_id="doc", table="other", row=3, column="2015"),
                dimensions={"row_label": "Total", "table_id": "other", "is_total_row": "true"},
            ),
        ),
        source_name="same_table_total_share",
    )
    query = "What was federal income taxes as a percentage of total income taxes in 2015?"
    routing = RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert result.answer == pytest.approx(135.0 / 700.0)
    assert plan.slots["numerator_token_ids"].grounded_value == ["tax:federal"]
    assert plan.slots["denominator_token_ids"].grounded_value == ["tax:total"]
    assert [token.token_id for token in result.selected_tokens] == ["tax:federal", "tax:total"]


def test_count_solver_binds_condition_slots() -> None:
    graph = _make_graph()
    routing = RuleBasedRouter().route("2024 年 TECH 行业 revenue 不低于 1000 的公司有几家")
    plan = solve_operator_plan("2024 年 TECH 行业 revenue 不低于 1000 的公司有几家", graph, routing)

    assert plan.operator == "COUNT"
    assert _value(plan.slots["condition_field"]) == "revenue"
    assert _value(plan.slots["condition_op"]) == ">="
    assert _value(plan.slots["condition_threshold"]) == 1000.0
    assert execute(graph, plan).answer == 1.0


def test_count_solver_does_not_parse_threshold_digits_as_year() -> None:
    graph = _make_graph()
    query = "how many TECH companies had employees >= 40 in 2024"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)

    assert _value(plan.slots["year"]) == 2024
    assert _value(plan.slots["condition_threshold"]) == 40.0
    assert execute(graph, plan).answer == 1.0


def test_count_solver_does_not_parse_decimal_threshold_as_year() -> None:
    graph = _make_graph()
    query = "revenue < 1995.0 的公司有几家"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)

    assert "year" not in plan.slots
    assert _value(plan.slots["condition_op"]) == "<"
    assert _value(plan.slots["condition_threshold"]) == 1995.0
    assert execute(graph, plan).answer == 4.0


def test_count_solver_keeps_cross_year_candidate_set_unfiltered() -> None:
    graph = _make_graph()
    query = "从 2023 到 2024 revenue 大于 900 的记录数量是多少"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)

    assert plan.operator == "COUNT"
    assert "year" not in plan.slots
    assert _value(plan.slots["condition_threshold"]) == 900.0
    assert execute(graph, plan).answer == 3.0


def test_avg_solver_keeps_explicit_year_list_as_candidate_set() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken("2019:net_loss", "2019", "2019", "net_loss", "Net Loss", 15571.0, year=2019),
            AttributeValueToken("2018:net_loss", "2018", "2018", "net_loss", "Net Loss", 24122.0, year=2018),
        ),
        source_name="tatqa_avg",
    )
    query = "What is the average Net Loss for December 31, 2018 and 2019?"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)

    assert plan.operator == "AVG"
    assert "year" not in plan.slots
    assert _value(plan.slots["target_field"]) == "net_loss"
    assert execute(graph, plan).answer == 19846.5


def test_ratio_solver_binds_numerator_and_denominator_mentions() -> None:
    graph = _make_graph()
    routing = RuleBasedRouter().route("2024 年 TECH 行业 net_profit 占 revenue 的比例")
    plan = solve_operator_plan("2024 年 TECH 行业 net_profit 占 revenue 的比例", graph, routing)

    assert plan.operator == "RATIO"
    assert _value(plan.slots["numerator_field"]) == "net_profit"
    assert _value(plan.slots["denominator_field"]) == "revenue"
    expected = (120 + 95) / (1200 + 950)
    result = execute(graph, plan)
    assert abs(result.answer - expected) < 1e-9
    assert result.metadata["executor_operator"] == "RATIO"
    assert result.metadata["replay_operands"] == [215.0, 2150.0]


def test_ratio_output_normalization_updates_replay_operands() -> None:
    numerator = AttributeValueToken("num", "row", "A", "cost", "cost", 48.0)
    denominator = AttributeValueToken("den", "row", "A", "notes", "notes", 7.0)
    result = ExecutionResult(
        answer=48.0 / 7000.0,
        selected_tokens=(numerator, denominator),
        calculation="48 / 7000",
        metadata={
            "numerator_token_ids": ["num"],
            "denominator_token_ids": ["den"],
            "replay_operands": [48.0, 7000.0],
        },
    )
    plan = SimpleNamespace(operator="RATIO", trace={"proposal_note": "percentage"})

    normalized = normalize_execution_output(result, plan)

    assert normalized.answer == pytest.approx(48.0 / 7.0)
    assert normalized.metadata["replay_operands"] == [48.0, 7.0]


def test_ratio_solver_does_not_treat_industry_token_as_entity() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken("a:revenue", "a:2025", "HelioGrid Power", "revenue", "revenue", 2240.0, year=2025, industry="RENEWABLE POWER"),
            AttributeValueToken("a:net_profit", "a:2025", "HelioGrid Power", "net_profit", "net profit", 224.0, year=2025, industry="RENEWABLE POWER"),
            AttributeValueToken("b:revenue", "b:2025", "Tidal Renewables", "revenue", "revenue", 1330.0, year=2025, industry="RENEWABLE POWER"),
            AttributeValueToken("b:net_profit", "b:2025", "Tidal Renewables", "net_profit", "net profit", 117.0, year=2025, industry="RENEWABLE POWER"),
        ),
        source_name="entity_overlap",
    )
    query = "2025 年 RENEWABLE POWER 行业 net_profit 占 revenue 的占比是多少"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)

    assert "entity" not in plan.slots
    expected = (224.0 + 117.0) / (2240.0 + 1330.0)
    assert execute(graph, plan).answer == expected


def test_share_solver_splits_as_percentage_of_field_mentions() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "a:sitting_fees",
                "a",
                "A",
                "sitting_fees_for_attending_board_committee_meetings",
                "Sitting Fees for attending board/ committee meetings",
                33.6,
            ),
            AttributeValueToken(
                "a:total_amount",
                "a",
                "A",
                "total_amount",
                "Total Amount",
                1243.6,
            ),
        ),
        source_name="tatqa_share_solver",
    )
    query = "What is the amount of total Sitting Fees as a percentage of Total Managerial Remuneration?"
    routing = RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(
        query,
        graph,
        routing,
        field_grounder=FieldGrounder(strategy="learnable_ranker"),
    )

    assert _value(plan.slots["numerator_field"]) == "sitting_fees_for_attending_board_committee_meetings"
    assert _value(plan.slots["denominator_field"]) == "total_amount"


def test_ratio_solver_splits_percent_of_as_part_of_frame() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "a:num",
                "a",
                "JPM",
                "less_cib_markets_net_interest_income_c",
                "less cib markets net interest income",
                3087.0,
                year=2018,
            ),
            AttributeValueToken(
                "a:managed",
                "a",
                "JPM",
                "net_interest_income_2013_managed_basis_a_b",
                "net interest income 2013 managed basis",
                55687.0,
                year=2018,
            ),
            AttributeValueToken(
                "a:excluding",
                "a",
                "JPM",
                "net_interest_income_excluding_cib_markets_a",
                "net interest income excluding cib markets",
                52600.0,
                year=2018,
            ),
        ),
        source_name="finqa_ratio_frame",
    )
    query = "in 2018 what was the percent of the cib markets net interest income as part of the managed interest income"
    routing = RoutingResult(operator="RATIO", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(
        query,
        graph,
        routing,
        field_grounder=FieldGrounder(strategy="learnable_ranker"),
    )
    result = execute(graph, plan)

    assert _value(plan.slots["numerator_field"]) == "less_cib_markets_net_interest_income_c"
    assert _value(plan.slots["denominator_field"]) == "net_interest_income_2013_managed_basis_a_b"
    assert abs(result.answer - (3087.0 / 55687.0)) < 1e-12


def test_ratio_solver_splits_what_percent_of_denominator_were_numerator_frame() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "a:expense",
                "a",
                "C",
                "total_operating_expenses",
                "Total operating expenses",
                988.0,
                year=2008,
            ),
            AttributeValueToken(
                "a:revenue",
                "a",
                "C",
                "net_interest_revenue",
                "Net interest revenue",
                3332.0,
                year=2008,
            ),
        ),
        source_name="finqa_ratio_were_frame",
    )
    query = "what percent of net interest revenue where total operating expenses in 2008?"
    routing = RoutingResult(operator="RATIO", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(
        query,
        graph,
        routing,
        field_grounder=FieldGrounder(strategy="learnable_ranker"),
    )
    result = execute(graph, plan)

    assert _value(plan.slots["numerator_field"]) == "total_operating_expenses"
    assert _value(plan.slots["denominator_field"]) == "net_interest_revenue"
    assert abs(result.answer - (988.0 / 3332.0)) < 1e-12


def test_ratio_solver_prefers_exact_total_denominator_field_over_total_column() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "zbh:interest_payments",
                "zbh",
                "ZBH",
                "interest_payments",
                "interest payments",
                1095.6,
            ),
            AttributeValueToken(
                "zbh:total_contractual_obligations",
                "zbh",
                "ZBH",
                "total_contractual_obligations",
                "total contractual obligations",
                2719.3,
                dimensions={"is_aggregate": "true"},
            ),
            AttributeValueToken(
                "zbh:total_column",
                "zbh",
                "ZBH",
                "total",
                "total",
                0.0,
            ),
        ),
        source_name="zbh_ratio_total",
    )
    query = "What percentage of total contractual obligations were interest payments?"
    routing = RoutingResult(operator="RATIO", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert _value(plan.slots["numerator_field"]) == "interest_payments"
    assert _value(plan.slots["denominator_field"]) == "total_contractual_obligations"
    assert abs(result.answer - (1095.6 / 2719.3)) < 1e-12


def test_share_solver_prefers_numerator_after_were_and_exact_total_denominator() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "zbh:interest_payments",
                "zbh",
                "ZBH",
                "interest_payments",
                "interest payments",
                1095.6,
            ),
            AttributeValueToken(
                "zbh:total_contractual_obligations",
                "zbh",
                "ZBH",
                "total_contractual_obligations",
                "total contractual obligations",
                2719.3,
                dimensions={"is_aggregate": "true"},
            ),
            AttributeValueToken(
                "zbh:total_column",
                "zbh",
                "ZBH",
                "total",
                "total",
                0.0,
            ),
        ),
        source_name="zbh_share_total",
    )
    query = "What percentage of total contractual obligations were interest payments?"
    routing = RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert _value(plan.slots["numerator_field"]) == "interest_payments"
    assert _value(plan.slots["denominator_field"]) == "total_contractual_obligations"
    assert abs(result.answer - (1095.6 / 2719.3)) < 1e-12


def test_share_solver_prefers_table_total_tokens_over_text_and_period_columns() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "text:interest_payments",
                "zbh",
                "ZBH",
                "interest_payments",
                "interest payments",
                118.8,
                industry="company",
            ),
            AttributeValueToken(
                "text:total_contractual_obligations",
                "zbh",
                "ZBH",
                "total_contractual_obligations",
                "total contractual obligations",
                2719.3,
                industry="company",
            ),
            AttributeValueToken(
                "table:interest_payments:total",
                "zbh",
                "interest payments",
                "interest_payments",
                "interest payments",
                1095.6,
                source=TokenSource(table="table_0", row=2, column="total", text_excerpt="1095.6"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 2,
                    "col_id": 1,
                    "projection": "row_metric",
                    "provenance_channel": "table",
                    "row_label": "interest payments",
                    "column_label": "total",
                },
            ),
            AttributeValueToken(
                "table:interest_payments:2015",
                "zbh",
                "interest payments",
                "interest_payments",
                "interest payments",
                834.3,
                year=2015,
                source=TokenSource(table="table_0", row=2, column="2015 and thereafter", text_excerpt="834.3"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 2,
                    "col_id": 5,
                    "projection": "row_metric",
                    "provenance_channel": "table",
                    "row_label": "interest payments",
                    "column_label": "2015 and thereafter",
                },
            ),
            AttributeValueToken(
                "table:total_contractual_obligations:total",
                "zbh",
                "total contractual obligations",
                "total_contractual_obligations",
                "total contractual obligations",
                2719.3,
                source=TokenSource(table="table_0", row=7, column="total", text_excerpt="2719.3"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 7,
                    "col_id": 1,
                    "projection": "row_metric",
                    "provenance_channel": "table",
                    "row_label": "total contractual obligations",
                    "column_label": "total",
                    "is_aggregate": "true",
                },
            ),
            AttributeValueToken(
                "table:total_contractual_obligations:2015",
                "zbh",
                "total contractual obligations",
                "total_contractual_obligations",
                "total contractual obligations",
                2005.0,
                year=2015,
                source=TokenSource(table="table_0", row=7, column="2015 and thereafter", text_excerpt="2005.0"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 7,
                    "col_id": 5,
                    "projection": "row_metric",
                    "provenance_channel": "table",
                    "row_label": "total contractual obligations",
                    "column_label": "2015 and thereafter",
                    "is_aggregate": "true",
                },
            ),
        ),
        source_name="zbh_table_share",
    )
    query = "What percentage of total contractual obligations were interest payments?"
    routing = RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert _value(plan.slots["numerator_token_ids"]) == ["table:interest_payments:total"]
    assert _value(plan.slots["denominator_token_ids"]) == ["table:total_contractual_obligations:total"]
    assert abs(result.answer - (1095.6 / 2719.3)) < 1e-12


def test_growth_solver_binds_from_and_to_years() -> None:
    graph = _make_graph()
    routing = RuleBasedRouter().route("TECH 行业 revenue 从 2023 到 2024 的增长率")
    plan = solve_operator_plan("TECH 行业 revenue 从 2023 到 2024 的增长率", graph, routing)

    assert plan.operator == "GROWTH"
    assert _value(plan.slots["target_field"]) == "revenue"
    assert _value(plan.slots["from_time"]) == 2023
    assert _value(plan.slots["to_time"]) == 2024
    assert "entity" not in plan.slots


def test_percent_change_solver_does_not_bind_year_mentions_as_entities() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "2018:equipment_notes_payable",
                "2018",
                "2018",
                "equipment_notes_payable",
                "Equipment notes payable",
                241.0,
                year=2018,
            ),
            AttributeValueToken(
                "2019:equipment_notes_payable",
                "2019",
                "2019",
                "equipment_notes_payable",
                "Equipment notes payable",
                88.0,
                year=2019,
            ),
        ),
        source_name="tatqa_growth",
    )
    query = "What is the percentage change in the equipment notes payable from 2018 to 2019?"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert plan.operator == "PERCENT_CHANGE"
    assert "entity" not in plan.slots
    assert result.answer == pytest.approx((88.0 - 241.0) / 241.0)
    assert result.checks["arithmetic_verified"] is True


def test_argmax_solver_binds_company_ranking_slots() -> None:
    graph = _make_graph()
    routing = RuleBasedRouter().route("2024 年 TECH 行业 revenue 最高的公司是哪家")
    plan = solve_operator_plan("2024 年 TECH 行业 revenue 最高的公司是哪家", graph, routing)

    assert plan.operator == "ARGMAX"
    assert _value(plan.slots["target_entity_type"]) == "company"
    assert _value(plan.slots["target_field"]) == "revenue"
    assert execute(graph, plan).answer == 1200.0


def test_argmax_solver_uses_latest_year_when_query_mentions_multiple_years_without_range() -> None:
    graph = _make_graph()
    query = "2023 和 2024 年 TECH 行业 revenue 最高的公司是哪家"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)

    assert plan.operator == "ARGMAX"
    assert _value(plan.slots["year"]) == 2024
    assert execute(graph, plan).answer == 1200.0


def test_operator_override_runs_oracle_operator_control() -> None:
    graph = _make_graph()
    routing = RuleBasedRouter().route("2024 年 TECH 行业 revenue 最高的公司是哪家")
    solver = OperatorSolver(field_grounder=FieldGrounder.from_lexical())
    plan = solver.solve(
        "2024 年 TECH 行业 revenue 最高的公司是哪家",
        graph,
        routing,
        operator="SUM",
    )

    assert routing.operator == "ARGMAX"
    assert plan.operator == "SUM"
    assert execute(graph, plan).answer == 2150.0


def test_lookup_solver_keeps_numbered_company_entity_constraint() -> None:
    graph = _make_ood_entity_graph()
    query = "For Ai Infrastructure Co 1, FY2022 operating profit value on record"
    routing = RoutingResult(operator="LOOKUP", confidence=1.0, intent="lookup", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert _value(plan.slots["entity"]) == "Ai Infrastructure Co 1"
    assert result.answer == 40.0
    assert [token.company_name for token in result.selected_tokens] == ["Ai Infrastructure Co 1"]


def test_difference_solver_uses_left_and_right_entities_without_single_entity_filter() -> None:
    graph = _make_ood_entity_graph()
    query = "Difference between Ai Infrastructure Co 1 and Ai Infrastructure Co 2 in revenue for 2024"
    routing = RoutingResult(operator="DIFFERENCE", confidence=1.0, intent="comparison", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert _value(plan.slots["left_entity"]) == "Ai Infrastructure Co 1"
    assert _value(plan.slots["right_entity"]) == "Ai Infrastructure Co 2"
    assert "entity" not in plan.slots
    assert result.answer == -113.0
    assert [token.company_name for token in result.selected_tokens] == [
        "Ai Infrastructure Co 1",
        "Ai Infrastructure Co 2",
    ]


def test_difference_solver_binds_year_change_as_to_minus_from() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "2018:personnel_related_items",
                "2018",
                "2018",
                "personnel_related_items",
                "Personnel-related items",
                32636.0,
                year=2018,
            ),
            AttributeValueToken(
                "2019:personnel_related_items",
                "2019",
                "2019",
                "personnel_related_items",
                "Personnel-related items",
                45318.0,
                year=2019,
            ),
        ),
        source_name="tatqa_difference",
    )
    query = "What is the change in Personnel-related items from 2018 to 2019?"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert plan.operator == "DIFFERENCE"
    assert _value(plan.slots["left_time"]) == 2019
    assert _value(plan.slots["right_time"]) == 2018
    assert "entity" not in plan.slots
    assert "left_entity" not in plan.slots
    assert "right_entity" not in plan.slots
    assert result.answer == 12682.0
    assert {token.year for token in result.selected_tokens} == {2018, 2019}


def test_increase_decrease_amount_executes_as_absolute_difference() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "2018:wireless_capital_expenditure",
                "2018",
                "2018",
                "wireless_capital_expenditure",
                "Wireless capital expenditure",
                1086.0,
                year=2018,
                unit="million",
            ),
            AttributeValueToken(
                "2019:wireless_capital_expenditure",
                "2019",
                "2019",
                "wireless_capital_expenditure",
                "Wireless capital expenditure",
                1320.0,
                year=2019,
                unit="million",
            ),
        ),
        source_name="tatqa_increase_decrease",
    )
    query = "What was the increase / (decrease) in wireless capital expenditure from 2018 to 2019?"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(
        query,
        graph,
        routing,
        field_grounder=FieldGrounder(strategy="learnable_ranker"),
    )
    result = execute(graph, plan)

    assert plan.operator == "DIFFERENCE"
    assert _value(plan.slots["left_time"]) == 2019
    assert _value(plan.slots["right_time"]) == 2018
    assert result.answer == 234.0


def test_difference_solver_treats_in_year_from_year_as_to_minus_from() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "2018:total",
                "2018",
                "2018",
                "total",
                "Total",
                6.6,
                year=2018,
            ),
            AttributeValueToken(
                "2019:total",
                "2019",
                "2019",
                "total",
                "Total",
                7.8,
                year=2019,
            ),
        ),
        source_name="tatqa_difference",
    )
    query = "What was the change in the total compensation in 2019 from 2018?"
    routing = RuleBasedRouter().route(query)
    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert plan.operator == "DIFFERENCE"
    assert _value(plan.slots["left_time"]) == 2019
    assert _value(plan.slots["right_time"]) == 2018
    assert result.answer == 1.2000000000000002


def test_predict_solver_resolves_numbered_company_with_industry_terms() -> None:
    graph = _make_ood_entity_graph()
    query = "Ai Infrastructure Co 2 revenue forward view for 2024"
    routing = RoutingResult(operator="FORECAST", confidence=1.0, intent="forecast", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)

    assert _value(plan.slots["entity"]) == "Ai Infrastructure Co 2"


def test_share_solver_prefers_same_column_total_row_over_total_column() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "purchase_obligations:less_than_1_year",
                "grmn",
                "purchase obligations",
                "purchase_obligations",
                "purchase obligations",
                265409.0,
                year=2006,
                source=TokenSource(table="table_0", row=2, column="payments due by period less than 1 year", text_excerpt="265409"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 2,
                    "col_id": 2,
                    "row_path": "purchase obligations",
                    "column_path": "payments due by period / less than 1 year",
                    "column_label": "less than 1 year",
                    "provenance_channel": "table",
                    "is_aggregate": False,
                },
            ),
            AttributeValueToken(
                "total:less_than_1_year",
                "grmn",
                "total",
                "total",
                "total",
                268766.0,
                year=2006,
                source=TokenSource(table="table_0", row=3, column="payments due by period less than 1 year", text_excerpt="268766"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 3,
                    "col_id": 2,
                    "row_path": "total",
                    "column_path": "payments due by period / less than 1 year",
                    "column_label": "less than 1 year",
                    "provenance_channel": "table",
                    "is_aggregate": True,
                    "is_total_row": True,
                },
                is_aggregate=True,
            ),
            AttributeValueToken(
                "total:total_column",
                "grmn",
                "total",
                "total",
                "total",
                296554.0,
                year=2006,
                source=TokenSource(table="table_0", row=3, column="payments due by period total", text_excerpt="296554"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 3,
                    "col_id": 1,
                    "row_path": "total",
                    "column_path": "payments due by period / total",
                    "column_label": "total",
                    "provenance_channel": "table",
                    "is_aggregate": True,
                    "is_total_row": True,
                },
                is_aggregate=True,
            ),
            AttributeValueToken(
                "purchase_obligations:total_column",
                "grmn",
                "purchase obligations",
                "purchase_obligations",
                "purchase obligations",
                265409.0,
                year=2006,
                source=TokenSource(table="table_0", row=2, column="payments due by period total", text_excerpt="265409"),
                dimensions={
                    "table_id": "table_0",
                    "row_id": 2,
                    "col_id": 1,
                    "row_path": "purchase obligations",
                    "column_path": "payments due by period / total",
                    "column_label": "total",
                    "provenance_channel": "table",
                    "is_aggregate": False,
                },
            ),
        ),
        source_name="grmn_same_column_total",
    )
    query = (
        "considering the payments due to less than a year , "
        "what is the percentage of purchase obligations concerning the total expenses?"
    )
    routing = RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle")

    plan = solve_operator_plan(query, graph, routing)
    result = execute(graph, plan)

    assert _value(plan.slots["numerator_token_ids"]) == ["purchase_obligations:less_than_1_year"]
    assert _value(plan.slots["denominator_token_ids"]) == ["total:less_than_1_year"]
    assert abs(result.answer - (265409.0 / 268766.0)) < 1e-12
