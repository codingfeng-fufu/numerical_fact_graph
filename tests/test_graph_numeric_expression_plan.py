from __future__ import annotations

from graph_numeric.core.attribute_graph import AttributeValueGraph
from graph_numeric.core.attribute_graph import AttributeValueToken
from graph_numeric.core.attribute_graph import TokenSource
from graph_numeric.core.attribute_graph import graph_from_csv_text
from graph_numeric.core.attribute_graph import graph_from_markdown_table
from graph_numeric.learning.field_grounder import FieldGrounder
from graph_numeric.operators.operator_solvers import OperatorSolver
from graph_numeric.pipeline.pipeline import run_operator_pipeline
from graph_numeric.learning.router import RoutingResult, RuleBasedRouter


def test_parse_percent_change_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What was the percentage change in cash flows from operations from 2014 to 2015?"
    )

    assert plan is not None
    assert plan.kind == "percent_change"
    assert plan.route_operator == "PERCENT_CHANGE"
    assert plan.root.op == "divide"
    assert [child.op for child in plan.root.children] == ["subtract", "lookup"]
    subtract = plan.root.children[0]
    assert [child.evidence_query.time_surface for child in subtract.children] == ["2015", "2014"]
    assert subtract.children[0].evidence_query.field_surface == "cash flows from operations"


def test_parse_part_to_whole_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What portion of the total noncancelable future lease commitments are due in fiscal year of 2019?"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    assert plan.route_operator == "SHARE"
    assert plan.root.op == "divide"
    assert [child.op for child in plan.root.children] == ["lookup", "lookup"]
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.time_surface == "2019"
    assert "lease commitments" in numerator.evidence_query.field_surface
    assert denominator.evidence_query.role == "whole"


def test_parse_percentage_of_x_are_y_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "what percentage of 2005 industrial packaging sales are containerboard sales?"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "containerboard sales"
    assert numerator.evidence_query.time_surface == "2005"
    assert denominator.evidence_query.field_surface == "industrial packaging sales"
    assert denominator.evidence_query.time_surface == "2005"


def test_parse_percentage_of_whole_with_text_condition_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is the percentage of the cash and cash equivalents at December 31, 2019 held in jurisdictions outside the U.S.?"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    assert plan.route_operator == "SHARE"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == (
        "cash and cash equivalents held in jurisdictions outside the u.s"
    )
    assert numerator.evidence_query.time_surface == "2019"
    assert numerator.evidence_query.role == "part"
    assert denominator.evidence_query.field_surface == "cash and cash equivalents"
    assert denominator.evidence_query.time_surface == "2019"
    assert denominator.evidence_query.role == "whole"


def test_parse_percent_of_the_change_as_percent_change():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "based on the analysis of the change in net revenue from 2014 to 2015 what was the percent of the change"
    )

    assert plan is not None
    assert plan.kind == "percent_change"
    assert plan.route_operator == "PERCENT_CHANGE"
    assert plan.root.op == "divide"
    subtract = plan.root.children[0]
    assert [child.evidence_query.time_surface for child in subtract.children] == ["2015", "2014"]
    assert subtract.children[0].evidence_query.field_surface == "net revenue"


def test_parse_percentage_net_change_as_percent_change():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "what was the percentage net change in the accrued liability for unrecognized tax benefits from 2007 to 2008?"
    )

    assert plan is not None
    assert plan.kind == "percent_change"
    assert plan.route_operator == "PERCENT_CHANGE"


def test_parse_difference_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What was the difference in revenue between 2020 and 2021?"
    )

    assert plan is not None
    assert plan.kind == "difference"
    assert plan.route_operator == "DIFFERENCE"
    assert plan.root.op == "subtract"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2021", "2020"]
    assert all(child.evidence_query.field_surface == "revenue" for child in plan.root.children)


def test_parse_net_change_compare_to_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "what is the net change in the amount spent for research and development in 2016 compare to 2015?"
    )

    assert plan is not None
    assert plan.kind == "difference"
    assert plan.route_operator == "DIFFERENCE"
    assert plan.root.op == "subtract"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2016", "2015"]
    assert plan.root.children[0].evidence_query.field_surface == "research and development"


def test_parse_ratio_of_x_to_y_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "at december 31 2009 what was the ratio of the aggregate cost to the fair value of the loans held-for-sale"
    )

    assert plan is not None
    assert plan.kind == "ratio"
    assert plan.route_operator == "RATIO"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "aggregate cost"
    assert denominator.evidence_query.field_surface == "fair value of the loans held-for-sale"
    assert numerator.evidence_query.role == "numerator"
    assert denominator.evidence_query.role == "denominator"


def test_parse_per_unit_ratio_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "productivity in the plastics business measured by million $ sales per employee was what in 2009?"
    )

    assert plan is not None
    assert plan.kind == "ratio"
    assert plan.route_operator == "RATIO"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "million $ sales"
    assert denominator.evidence_query.field_surface == "employee"
    assert numerator.evidence_query.time_surface == "2009"
    assert plan.confidence < 0.8


def test_low_confidence_expression_does_not_override_primary_route():
    graph = graph_from_csv_text(
        "\n".join(
            [
                "company_name,industry,year,sales,employee",
                "Alpha,TECH,2009,634.9,1000",
            ]
        )
    )

    pipeline = run_operator_pipeline(
        "productivity in the plastics business measured by million $ sales per employee was what in 2009?",
        graph,
        router=RuleBasedRouter(),
        solver=OperatorSolver(field_grounder=FieldGrounder(strategy="learnable_ranker")),
        enable_fallback=True,
    )

    assert pipeline.routing.trace.get("expression_route") is None


def test_low_confidence_percentage_part_of_total_does_not_parse_on_main_path():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "what percentage of furniture and equipment in 2019 as a total of all the assets in 2019?"
    )

    assert plan is None


def test_parse_average_range_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "what was the average aggregate intrinsic value of stock options exercised from 2013 to 2015"
    )

    assert plan is not None
    assert plan.kind == "average"
    assert plan.route_operator == "AVG"
    assert plan.root.op == "average"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2013", "2014", "2015"]
    assert all(
        child.evidence_query.field_surface == "aggregate intrinsic value of stock options exercised"
        for child in plan.root.children
    )


def test_parse_total_of_two_fields_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "for restructuring expense, what is the total balance of severance and related charges and lease termination costs in millions?"
    )

    assert plan is not None
    assert plan.kind == "sum"
    assert plan.route_operator == "SUM"
    assert plan.root.op == "add"
    left, right = plan.root.children
    assert left.evidence_query.field_surface == "balance of severance and related charges"
    assert right.evidence_query.field_surface == "lease termination costs in millions"


def test_parse_sum_expression_tree_normalizes_fy_short_year():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What was the sum of total revenue and total other income for FY19?"
    )

    assert plan is not None
    assert plan.kind == "sum"
    left, right = plan.root.children
    assert [left.evidence_query.time_surface, right.evidence_query.time_surface] == [
        "2019",
        "2019",
    ]
    assert left.evidence_query.field_surface == "total revenue"
    assert right.evidence_query.field_surface == "total other income"


def test_parse_lookup_expression_tree_normalizes_fiscal_short_year():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan("What was revenue in fiscal year 19?")

    assert plan is not None
    assert plan.kind == "lookup"
    assert plan.root.evidence_query.time_surface == "2019"
    assert plan.root.evidence_query.field_surface == "revenue"


def test_parse_total_same_field_across_year_list_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "what is the total amount of stock options cancelled in millions during 2017, 2016 and 2015?"
    )

    assert plan is not None
    assert plan.kind == "sum"
    assert plan.route_operator == "SUM"
    assert plan.root.op == "add"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2017", "2016", "2015"]
    assert all(
        child.evidence_query.field_surface == "stock options cancelled in millions"
        for child in plan.root.children
    )


def test_difference_query_with_total_does_not_route_to_sum():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is the difference in the total franchise revenues between 2018 and 2019?"
    )

    assert plan is not None
    assert plan.kind == "difference"
    assert plan.route_operator == "DIFFERENCE"


def test_parse_low_confidence_simple_lookup_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan("what is the the interest expense in 2009?")

    assert plan is not None
    assert plan.kind == "lookup"
    assert plan.route_operator == "LOOKUP"
    assert plan.confidence < 0.8
    assert plan.root.evidence_query.field_surface == "interest expense"
    assert plan.root.evidence_query.time_surface == "2009"


def test_parse_low_confidence_total_lookup_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan("what were total operating expenses in 2018?")

    assert plan is not None
    assert plan.kind == "lookup"
    assert plan.route_operator == "LOOKUP"
    assert plan.confidence < 0.8
    assert plan.root.evidence_query.field_surface == "total operating expenses"
    assert plan.root.evidence_query.time_surface == "2018"


def test_parse_increase_from_to_as_percent_change_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan("what was the percentage increase in the rent expense from 2010 to 2011")

    assert plan is not None
    assert plan.kind == "percent_change"
    assert plan.route_operator == "PERCENT_CHANGE"
    subtract = plan.root.children[0]
    assert [child.evidence_query.time_surface for child in subtract.children] == ["2011", "2010"]
    assert subtract.children[0].evidence_query.field_surface == "rent expense"


def test_parse_average_for_two_years_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan("What is the average Gross profit for the period December 31, 2019 and 2018?")

    assert plan is not None
    assert plan.kind == "average"
    assert plan.route_operator == "AVG"
    assert plan.root.op == "average"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2019", "2018"]
    assert all(child.evidence_query.field_surface == "gross profit" for child in plan.root.children)


def test_parse_percent_of_as_part_of_total_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "in 2013 what was the percent of the professional fees as part of the total re-organization costs"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    assert plan.route_operator == "SHARE"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "professional fees"
    assert denominator.evidence_query.field_surface == "total re-organization costs"
    assert numerator.evidence_query.time_surface == "2013"
    assert denominator.evidence_query.time_surface == "2013"
    assert plan.confidence < 0.8


def test_parse_percent_of_part_to_non_total_whole_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "in 2006 what was the percent of the recognized a pre-tax gain to the proceeds "
        "of the sale of its global branded pharmaceuticals businesses"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    assert plan.route_operator == "SHARE"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "recognized a pre-tax gain"
    assert denominator.evidence_query.field_surface == (
        "proceeds of the sale of its global branded pharmaceuticals businesses"
    )
    assert numerator.evidence_query.time_surface == "2006"
    assert denominator.evidence_query.time_surface == "2006"
    assert plan.confidence < 0.8


def test_expression_grounding_keeps_proceeds_denominator_for_percent_of_to_query():
    query = (
        "in 2006 what was the percent of the recognized a pre-tax gain to the proceeds "
        "of the sale of its global branded pharmaceuticals businesses"
    )
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="mmm:proceeds",
                entity_id="mmm:2006",
                company_name="MMM",
                field_name="proceeds_of_sale",
                field_label="proceeds of sale",
                value=1.209,
                year=2006,
                unit="billion USD",
                source=TokenSource(
                    document_id="doc",
                    row=0,
                    column="proceeds",
                    text_excerpt="3m received proceeds of $ 1.209 billion",
                ),
            ),
            AttributeValueToken(
                token_id="mmm:gain_general",
                entity_id="mmm:2006",
                company_name="MMM",
                field_name="pre_tax_gain_on_sale",
                field_label="pre-tax gain on sale",
                value=1.074,
                year=2006,
                unit="billion USD",
                source=TokenSource(
                    document_id="doc",
                    row=0,
                    column="gain",
                    text_excerpt="recognized a pre-tax gain on sale of $ 1.074 billion in 2006",
                ),
            ),
            AttributeValueToken(
                token_id="mmm:gain_specific",
                entity_id="mmm:2006",
                company_name="MMM",
                field_name="pre_tax_gain_on_sale_of_global_branded_pharmaceuticals_businesses",
                field_label="pre-tax gain on sale of global branded pharmaceuticals businesses",
                value=1.074,
                year=2006,
                unit="billion USD",
                source=TokenSource(
                    document_id="doc",
                    row=0,
                    column="gain_specific",
                    text_excerpt="recognized a pre-tax gain on sale of $ 1.074 billion in 2006",
                ),
            ),
        ),
        source_name="mmm",
    )

    pipeline = run_operator_pipeline(
        query,
        graph,
        routing=RoutingResult(
            operator="SHARE",
            confidence=1.0,
            intent="ratio",
            route_type="oracle",
        ),
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.answer is not None
    assert abs(pipeline.answer - (1.074 / 1.209)) < 1e-9
    assert pipeline.plan.slots["denominator_token_ids"].grounded_value == ["mmm:proceeds"]


def test_parse_x_as_percentage_of_total_y_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is the revenue from Canada in 2019 as a percentage of the total revenue in 2019?"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "revenue from canada"
    assert numerator.evidence_query.time_surface == "2019"
    assert numerator.evidence_query.role == "part"
    assert denominator.evidence_query.field_surface == "total revenue"
    assert denominator.evidence_query.time_surface == "2019"
    assert denominator.evidence_query.role == "whole"


def test_parse_value_of_x_as_percentage_of_total_y_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is the value of the maintenance related revenue as a percentage of the total software-related revenues in 2019?"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "maintenance related revenue"
    assert numerator.evidence_query.time_surface == "2019"
    assert denominator.evidence_query.field_surface == "total software-related revenues"
    assert denominator.evidence_query.time_surface == "2019"


def test_parse_x_expressed_as_percentage_of_total_y_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is Purchase obligations expressed as a percentage of Total contractual obligations?"
    )

    assert plan is not None
    assert plan.kind == "part_to_whole"
    numerator, denominator = plan.root.children
    assert numerator.evidence_query.field_surface == "purchase obligations"
    assert denominator.evidence_query.field_surface == "total contractual obligations"


def test_parse_growth_rate_from_to_expression_tree():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan("what is the growth rate of net sales from 2014 to 2015?")

    assert plan is not None
    assert plan.kind == "percent_change"
    assert plan.route_operator == "PERCENT_CHANGE"
    subtract = plan.root.children[0]
    assert [child.evidence_query.time_surface for child in subtract.children] == ["2015", "2014"]
    assert subtract.children[0].evidence_query.field_surface == "net sales"


def test_parse_percent_change_with_fiscal_year_range_keeps_metric_field():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is the percentage change of total assets from fiscal year 2018 to 2019?"
    )

    assert plan is not None
    assert plan.kind == "percent_change"
    subtract = plan.root.children[0]
    assert [child.evidence_query.time_surface for child in subtract.children] == ["2019", "2018"]
    assert all(
        child.evidence_query.field_surface == "total assets"
        for child in subtract.children
    )


def test_parse_change_from_to_preserves_field_with_internal_of_phrase():
    from graph_numeric.core.expression_plan import parse_expression_plan
    from graph_numeric.core.attribute_graph import normalize_identifier

    plan = parse_expression_plan(
        "What was the change in the Weighted average number of shares outstanding incl. "
        "dilutive effect of share options from 2018 to 2019?"
    )

    assert plan is not None
    assert plan.kind == "difference"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2019", "2018"]
    assert all(
        normalize_identifier(child.evidence_query.field_surface)
        == "weighted_average_number_of_shares_outstanding_incl_dilutive_effect_of_share_options"
        for child in plan.root.children
    )


def test_parse_change_between_years_keeps_trailing_date_qualifier():
    from graph_numeric.core.expression_plan import parse_expression_plan

    plan = parse_expression_plan(
        "What is the change in balance between 2017 and 2018 at January 1?"
    )

    assert plan is not None
    assert plan.kind == "difference"
    assert [child.evidence_query.time_surface for child in plan.root.children] == ["2018", "2017"]
    assert all(
        child.evidence_query.field_surface == "balance at january 1"
        for child in plan.root.children
    )


def test_pipeline_attaches_expression_trace_for_percent_change():
    graph = graph_from_csv_text(
        "\n".join(
            [
                "company_name,industry,year,revenue",
                "Alpha,TECH,2020,100",
                "Alpha,TECH,2021,125",
            ]
        )
    )

    pipeline = run_operator_pipeline(
        "percentage change in revenue from 2020 to 2021",
        graph,
        router=RuleBasedRouter(),
        solver=OperatorSolver(field_grounder=FieldGrounder(strategy="learnable_ranker")),
        enable_fallback=True,
    )

    assert pipeline.status == "ok"
    assert pipeline.selected_operator == "PERCENT_CHANGE"
    assert pipeline.plan is not None
    expression = pipeline.plan.trace["expression_plan"]
    assert expression["kind"] == "percent_change"
    assert expression["root"]["op"] == "divide"
    assert expression["root"]["children"][0]["op"] == "subtract"
    payload = pipeline.to_dict()
    assert payload["plan"]["operator"] == "PERCENT_CHANGE"
    assert payload["plan"]["trace"]["expression_plan"]["kind"] == "percent_change"
    assert payload["hybrid_query_trace"]["selected_operator"] == "PERCENT_CHANGE"


def test_expression_route_overrides_future_share_question_before_prediction():
    graph = graph_from_csv_text(
        "\n".join(
            [
                "company_name,industry,year,future_notes,total_future_notes",
                "Alpha,TECH,2017,349,1000",
            ]
        )
    )

    pipeline = run_operator_pipeline(
        "what portion of future notes are due by 2017?",
        graph,
        router=RuleBasedRouter(),
        solver=OperatorSolver(field_grounder=FieldGrounder(strategy="learnable_ranker")),
        enable_fallback=True,
    )

    assert pipeline.routing.trace["expression_route"]["original_operator"] == "PREDICT"
    assert pipeline.routing.operator == "SHARE"
    assert pipeline.selected_operator == "SHARE"
    assert pipeline.plan is not None
    assert pipeline.plan.trace["expression_plan"]["kind"] == "part_to_whole"


def test_segment_net_sales_sentence_values_become_graph_tokens():
    graph = graph_from_markdown_table(
        """
industrial packaging in millions 2006 2005 2004 .
| in millions | 2006 | 2005 | 2004 |
| --- | --- | --- | --- |
| sales | $ 4925 | $ 4625 | $ 4545 |
u.s. containerboard net sales for 2006 were $ 955 million , compared with $ 895 million in 2005 and $ 950 million for 2004 .
"""
    )

    selected = [
        token
        for token in graph.tokens
        if token.year == 2005
        and token.value == 895.0
        and token.dimensions.get("segment") == "u.s. containerboard"
    ]

    assert selected
    assert selected[0].field_name == "revenue"
    assert selected[0].unit == "$ million"
    assert selected[0].dimensions["row_label"] == "u.s. containerboard net sales"
    table_value = next(
        token
        for token in graph.tokens
        if token.token_id == "2005:revenue" and token.value == 4625.0
    )
    assert table_value.unit == "$ million"


def test_expression_grounding_selects_part_and_whole_tokens_for_share():
    graph = graph_from_markdown_table(
        """
industrial packaging in millions 2006 2005 2004 .
| in millions | 2006 | 2005 | 2004 |
| --- | --- | --- | --- |
| sales | $ 4925 | $ 4625 | $ 4545 |
u.s. containerboard net sales for 2006 were $ 955 million , compared with $ 895 million in 2005 and $ 950 million for 2004 .
"""
    )

    pipeline = run_operator_pipeline(
        "what percentage of 2005 industrial packaging sales are containerboard sales?",
        graph,
        router=RuleBasedRouter(),
        solver=OperatorSolver(field_grounder=FieldGrounder(strategy="learnable_ranker")),
        enable_fallback=True,
    )

    assert pipeline.status == "ok"
    assert pipeline.selected_operator == "SHARE"
    assert pipeline.answer == 895.0 / 4625.0
    assert pipeline.plan is not None
    slots = pipeline.plan.to_dict()["slots"]
    assert slots["numerator_token_ids"]["grounded_value"] == ["u_s_containerboard:2005:revenue"]
    assert slots["denominator_token_ids"]["grounded_value"] == ["2005:revenue"]
    metadata = pipeline.result.metadata if pipeline.result is not None else {}
    assert metadata["numerator_token_ids"] == ["u_s_containerboard:2005:revenue"]
    assert metadata["denominator_token_ids"] == ["2005:revenue"]


def test_textual_share_grounding_separates_same_field_part_and_whole_tokens():
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="cash_total_2019",
                entity_id="company:2019:company",
                company_name="Example Co",
                field_name="cash_and_cash_equivalents",
                field_label="cash and cash equivalents",
                value=454.0,
                year=2019,
                unit="million_usd",
                source=TokenSource(
                    text_excerpt=(
                        "Of the $454 million in cash and cash equivalents at December 31, 2019, "
                        "approximately $383 million was held in jurisdictions outside the U.S."
                    )
                ),
                raw_label="cash and cash equivalents",
            ),
            AttributeValueToken(
                token_id="cash_outside_us_2019",
                entity_id="company:2019:company",
                company_name="Example Co",
                field_name="cash_and_cash_equivalents",
                field_label="cash and cash equivalents",
                value=383.0,
                year=2019,
                unit="million_usd",
                source=TokenSource(
                    text_excerpt=(
                        "approximately $383 million was held in jurisdictions outside the U.S."
                    )
                ),
                raw_label="cash and cash equivalents held in jurisdictions outside the U.S.",
            ),
        )
    )

    pipeline = run_operator_pipeline(
        "What is the percentage of the cash and cash equivalents at December 31, 2019 held in jurisdictions outside the U.S.?",
        graph,
        router=RuleBasedRouter(),
        solver=OperatorSolver(field_grounder=FieldGrounder(strategy="learnable_ranker")),
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.selected_operator == "SHARE"
    assert pipeline.answer == 383.0 / 454.0
    assert pipeline.plan is not None
    slots = pipeline.plan.to_dict()["slots"]
    assert slots["numerator_token_ids"]["grounded_value"] == ["cash_outside_us_2019"]
    assert slots["denominator_token_ids"]["grounded_value"] == ["cash_total_2019"]
    metadata = pipeline.result.metadata if pipeline.result is not None else {}
    assert metadata["numerator_token_ids"] == ["cash_outside_us_2019"]
    assert metadata["denominator_token_ids"] == ["cash_total_2019"]
