from __future__ import annotations

from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
    TokenSource,
    graph_from_csv_text,
)
from graph_numeric.learning.evidence_faithfulness import TokenFaithfulnessChecker
from graph_numeric.learning.router_second_opinion import (
    ClassifierRouterSecondOpinion,
    RouterSecondOpinionResult,
)
from graph_numeric.operators.operator_plan import OperatorPlan, Slot
from graph_numeric.operators.executor import ExecutionResult
from graph_numeric.learning.metric_matcher import MetricMatcher, NliMetricJudgment
from graph_numeric.pipeline.pipeline import execute_with_fallback
from graph_numeric.pipeline.pipeline import _candidate_score_row
from graph_numeric.pipeline.pipeline import _implausible_result_trace
from graph_numeric.pipeline.pipeline import run_operator_pipeline
from graph_numeric.learning.router import RoutingResult
from graph_numeric.learning.router_factory import RuntimeFallbackHybridRouter
from graph_numeric.core.unit_resolver import UnitResolver
from graph_numeric.operators.validation import validate_preconditions
import graph_numeric.pipeline.pipeline as pipeline_module


def test_pipeline_s5_ablation_skips_validation_without_changing_binding(monkeypatch) -> None:
    class RejectedValidation:
        ok = False
        status = "precondition_violation"
        blocking_violations = ("synthetic_s5_rejection",)

        def to_dict(self):
            return {
                "status": self.status,
                "blocking_violations": list(self.blocking_violations),
            }

    monkeypatch.setattr(
        pipeline_module,
        "validate_preconditions",
        lambda graph, plan: RejectedValidation(),
    )

    rejected = run_operator_pipeline(
        "total revenue",
        _graph(),
        force_operator="SUM",
        enable_fallback=False,
    )
    accepted = run_operator_pipeline(
        "total revenue",
        _graph(),
        force_operator="SUM",
        enable_fallback=False,
        enable_s5_validation=False,
    )

    assert rejected.status == "abstained"
    assert accepted.status == "ok"
    assert accepted.answer == 1200.0
    assert accepted.attempts[-1].validation == {
        "status": "disabled_for_ablation",
        "blocking_violations": [],
    }


def test_s5_gate_config_defaults_to_all_checks_enabled() -> None:
    gates = pipeline_module.S5GateConfig()

    assert gates.to_dict() == {
        "preconditions": True,
        "unit": True,
        "verifier": True,
        "plausibility": True,
    }


def test_s5_unit_gate_filters_only_unit_failures() -> None:
    gates = pipeline_module.S5GateConfig(unit=False)

    assert pipeline_module._enabled_precondition_violations(
        ("unit_incompatible", "records_not_found"),
        gates,
    ) == ["records_not_found"]
    assert pipeline_module._enabled_blocking_checks(
        {"unit_compatible": False, "tokens_nonempty": False},
        gates,
    ) == ["tokens_nonempty"]


def test_s5_precondition_and_verifier_gates_are_independent() -> None:
    gates = pipeline_module.S5GateConfig(preconditions=False, verifier=False)

    assert pipeline_module._enabled_precondition_violations(
        ("records_not_found", "unit_incompatible"),
        gates,
    ) == ["unit_incompatible"]
    assert pipeline_module._enabled_blocking_checks(
        {"tokens_nonempty": False, "unit_compatible": False},
        gates,
    ) == ["unit_compatible"]


def _slot(surface: str, value: object, conf: float = 0.9) -> Slot:
    return Slot(surface=surface, grounded_value=value, confidence=conf)


def _graph() -> AttributeValueGraph:
    tokens = (
        AttributeValueToken(
            token_id="alpha:revenue",
            entity_id="alpha",
            company_name="Alpha",
            field_name="revenue",
            field_label="revenue",
            value=1200.0,
            year=2024,
            source=TokenSource(document_id="doc", row=0, column="revenue", text_excerpt="1200"),
            unit="million USD",
        ),
        AttributeValueToken(
            token_id="alpha:net_profit",
            entity_id="alpha",
            company_name="Alpha",
            field_name="net_profit",
            field_label="net profit",
            value=120.0,
            year=2024,
            source=TokenSource(document_id="doc", row=0, column="net_profit", text_excerpt="120"),
            unit="million USD",
        ),
    )
    return AttributeValueGraph(tokens, source_name="doc")


def _change_graph() -> AttributeValueGraph:
    tokens = (
        AttributeValueToken(
            token_id="alpha:gain:2018",
            entity_id="alpha:2018",
            company_name="Alpha",
            field_name="net_unrealized_gains",
            field_label="net unrealized gains",
            value=0.6,
            year=2018,
            source=TokenSource(
                document_id="doc",
                text_excerpt="Net unrealized gains were $0.6 million for 2018",
            ),
            unit="million USD",
            raw_label="net unrealized gains",
        ),
        AttributeValueToken(
            token_id="alpha:gain:2017",
            entity_id="alpha:2017",
            company_name="Alpha",
            field_name="net_unrealized_gains",
            field_label="net unrealized gains",
            value=6.6,
            year=2017,
            source=TokenSource(
                document_id="doc",
                text_excerpt="net unrealized gains of $6.6 million for 2017",
            ),
            unit="million USD",
            raw_label="net unrealized gains",
        ),
    )
    return AttributeValueGraph(tokens, source_name="doc")


def _ambiguous_change_graph() -> AttributeValueGraph:
    tokens = (
        AttributeValueToken(
            token_id="alpha:gain:2018:a",
            entity_id="alpha:2018:a",
            company_name="Alpha",
            field_name="net_unrealized_gains",
            field_label="net unrealized gains",
            value=0.6,
            year=2018,
            source=TokenSource(document_id="doc", text_excerpt="net unrealized gains were $0.6 million for 2018"),
            unit="million USD",
            raw_label="net unrealized gains",
        ),
        AttributeValueToken(
            token_id="alpha:gain:2018:b",
            entity_id="alpha:2018:b",
            company_name="Alpha",
            field_name="net_unrealized_gains",
            field_label="net unrealized gains",
            value=0.7,
            year=2018,
            source=TokenSource(document_id="doc", text_excerpt="net unrealized gains were also reported as $0.7 million for 2018"),
            unit="million USD",
            raw_label="net unrealized gains",
        ),
        AttributeValueToken(
            token_id="alpha:gain:2017",
            entity_id="alpha:2017",
            company_name="Alpha",
            field_name="net_unrealized_gains",
            field_label="net unrealized gains",
            value=6.6,
            year=2017,
            source=TokenSource(document_id="doc", text_excerpt="net unrealized gains of $6.6 million for 2017"),
            unit="million USD",
            raw_label="net unrealized gains",
        ),
    )
    return AttributeValueGraph(tokens, source_name="doc")


def _two_company_graph() -> AttributeValueGraph:
    return graph_from_csv_text(
        "\n".join(
            [
                "company_name,industry,year,revenue,net_profit,employees",
                "Alpha,TECH,2024,1200 million USD,120 million USD,55",
                "Beta,TECH,2024,950 million USD,95 million USD,35",
            ]
        ),
        source_name="review.csv",
    )


def _share_expression_graph() -> AttributeValueGraph:
    return graph_from_csv_text(
        "\n".join(
            [
                "company_name,year,total_revenue,cloud_revenue",
                "Alpha,2024,1000 million USD,250 million USD",
            ]
        ),
        source_name="share_expression.csv",
    )


def _equal_value_share_expression_graph() -> AttributeValueGraph:
    return graph_from_csv_text(
        "\n".join(
            [
                "company_name,year,total_revenue,cloud_revenue",
                "Alpha,2024,1000 million USD,1000 million USD",
            ]
        ),
        source_name="equal_share_expression.csv",
    )


def _cash_outflow_share_graph() -> AttributeValueGraph:
    tokens = (
        AttributeValueToken(
            token_id="cash:due_2013",
            entity_id="cash",
            company_name="Cash",
            field_name="due_2013",
            field_label="Due in 2013",
            value=3515.0,
            year=2013,
            unit="million USD",
            source=TokenSource(document_id="cash.md", row=0, column="2013", text_excerpt="3515"),
        ),
        AttributeValueToken(
            token_id="cash:total",
            entity_id="cash",
            company_name="Cash",
            field_name="total_expected_cash_outflow",
            field_label="Total expected cash outflow",
            value=23556.0,
            year=2013,
            unit="million USD",
            source=TokenSource(document_id="cash.md", row=0, column="total", text_excerpt="23556"),
        ),
    )
    return AttributeValueGraph(tokens, source_name="cash.md")


def test_validation_reports_missing_required_slots() -> None:
    report = validate_preconditions(_graph(), OperatorPlan(operator="SUM", slots={}))

    assert report.status == "precondition_violation"
    assert report.missing_slots == ("target_field",)
    assert "missing_required_slots" in report.blocking_violations


def test_precondition_report_exposes_each_predicate() -> None:
    report = validate_preconditions(
        _graph(),
        OperatorPlan(operator="SUM", slots={"target_field": _slot("revenue", "revenue")}),
    )

    assert set(report.predicates) >= {
        "required_slots",
        "slot_values",
        "field_exists",
        "records_found",
        "sufficient_factors",
        "unit_compatible",
    }
    assert report.ok == all(report.predicates.values())


def test_verifier_report_exposes_blocking_predicates() -> None:
    from graph_numeric.operators.executor import execute
    from graph_numeric.operators.verifier import Verifier

    blocking_checks = {
        "plan_valid",
        "field_exists",
        "tokens_nonempty",
        "tokens_field_consistent",
        "filter_satisfied",
        "unit_compatible",
        "arithmetic_verified",
        "output_type_valid",
    }
    graph = _graph()
    plan = OperatorPlan(operator="SUM", slots={"target_field": _slot("revenue", "revenue")})
    result = execute(graph, plan)
    report = Verifier().verify(graph, plan, result)

    assert set(report.checks) >= blocking_checks


def test_pipeline_abstains_after_precondition_violation_without_cross_operator_fallback() -> None:
    routing = RoutingResult(
        operator="LOOKUP",
        confidence=0.4,
        intent="lookup",
        fallback_operators=["SUM"],
    )
    pipeline = run_operator_pipeline(
        "2024 revenue total",
        _graph(),
        routing=routing,
        enable_fallback=True,
    )

    assert pipeline.status == "abstained"
    assert pipeline.fallback_used is False
    assert pipeline.selected_operator is None
    assert pipeline.answer is None
    assert [attempt.operator for attempt in pipeline.attempts] == ["LOOKUP"]
    assert pipeline.attempts[0].status == "precondition_violation"
    assert pipeline.abstain_trace["reason"] == "operator_precondition_failed"


def test_s5_audit_sink_does_not_change_pipeline_result() -> None:
    from graph_numeric.audit.s5_audit import S5PredicateEvent

    routing = RoutingResult(operator="SUM", confidence=1.0, intent="sum", route_type="oracle")
    plain = run_operator_pipeline(
        "2024 revenue total",
        _graph(),
        routing=routing,
        enable_fallback=False,
    )
    events: list[S5PredicateEvent] = []
    audited = run_operator_pipeline(
        "2024 revenue total",
        _graph(),
        routing=routing,
        enable_fallback=False,
        audit_sink=events.append,
    )

    assert audited.to_dict() == plain.to_dict()
    assert events
    assert all(isinstance(event, S5PredicateEvent) for event in events)


def test_pipeline_disable_fallback_limits_preferred_operators_to_top_choice() -> None:
    routing = RoutingResult(
        operator="LOOKUP",
        confidence=0.4,
        intent="lookup",
        fallback_operators=["SUM"],
    )
    pipeline = run_operator_pipeline(
        "2024 revenue total",
        _graph(),
        routing=routing,
        preferred_operators=["LOOKUP", "SUM"],
        enable_fallback=False,
    )

    assert pipeline.status == "abstained"
    assert pipeline.fallback_used is False
    assert [attempt.operator for attempt in pipeline.attempts] == ["LOOKUP"]
    assert pipeline.abstain_trace["reason"] == "operator_precondition_failed"


def test_pipeline_rejects_unknown_entity_in_single_entity_lookup() -> None:
    pipeline = run_operator_pipeline("Delta 2024 revenue", _two_company_graph())

    assert pipeline.status == "failed"
    assert pipeline.attempts[0].status == "preflight_rejected"
    assert pipeline.attempts[0].error == "entity_not_found:Delta"


def test_pipeline_rejects_unknown_industry_instead_of_falling_back_to_sole_industry() -> None:
    pipeline = run_operator_pipeline("2024 HEALTHCARE revenue total", _two_company_graph())

    assert pipeline.status == "failed"
    assert pipeline.attempts[0].status == "preflight_rejected"
    assert pipeline.attempts[0].error == "industry_not_found:HEALTHCARE"


def test_pipeline_rejects_known_metric_alias_when_field_is_missing_from_graph() -> None:
    pipeline = run_operator_pipeline("2024 TECH cash flow total", _two_company_graph())

    assert pipeline.status == "failed"
    assert pipeline.attempts[0].status == "preflight_rejected"
    assert pipeline.attempts[0].error == "field_not_found:cash_flow"


def test_preflight_accepts_open_schema_field_matching_query_phrase() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="2018:net_earnings_from_operations",
                entity_id="2018",
                company_name="2018",
                field_name="net_earnings_from_operations",
                field_label="Net earnings from operations",
                value=157133.0,
                year=2018,
                source=TokenSource(document_id="tatqa", row=3, column="2018", text_excerpt="157,133"),
            ),
            AttributeValueToken(
                token_id="2019:net_earnings_from_operations",
                entity_id="2019",
                company_name="2019",
                field_name="net_earnings_from_operations",
                field_label="Net earnings from operations",
                value=329013.0,
                year=2019,
                source=TokenSource(document_id="tatqa", row=3, column="2019", text_excerpt="329,013"),
            ),
        ),
        source_name="tatqa",
    )

    pipeline = run_operator_pipeline("What is the Net earnings from operations in 2018?", graph)

    assert pipeline.status == "ok"
    assert pipeline.answer == 157133.0
    assert pipeline.attempts[0].status == "ok"


def test_ratio_uses_field_encoded_year_and_dimension_label_match() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                "row_0:2019_actual",
                "row_0",
                "row_0",
                "2019_actual",
                "2019 actual",
                277.3,
                dimensions={
                    "remuneration_key_performance_indicator": "Group operating profit (£m)",
                    "remuneration_measure": "Annual Incentive Plan",
                },
            ),
            AttributeValueToken(
                "row_0:2019_target",
                "row_0",
                "row_0",
                "2019_target",
                "2019 target",
                270.3,
                dimensions={
                    "remuneration_key_performance_indicator": "Group operating profit (£m)",
                    "remuneration_measure": "Annual Incentive Plan",
                },
            ),
            AttributeValueToken(
                "row_1:2019_actual",
                "row_1",
                "row_1",
                "2019_actual",
                "2019 actual",
                296.4,
                dimensions={
                    "remuneration_key_performance_indicator": "Group cash generation (£m)",
                    "remuneration_measure": "Annual Incentive Plan",
                },
            ),
            AttributeValueToken(
                "row_1:2019_target",
                "row_1",
                "row_1",
                "2019_target",
                "2019 target",
                285.0,
                dimensions={
                    "remuneration_key_performance_indicator": "Group cash generation (£m)",
                    "remuneration_measure": "Annual Incentive Plan",
                },
            ),
        ),
        source_name="tatqa_ratio",
    )

    pipeline = run_operator_pipeline(
        "What is the 2019 actual group operating profit expressed as a ratio of the 2019 target group operating profit?",
        graph,
    )

    assert pipeline.status == "ok"
    assert abs(pipeline.answer - (277.3 / 270.3)) < 1e-9


def test_pipeline_keeps_valid_single_industry_query_without_explicit_industry() -> None:
    pipeline = run_operator_pipeline("2024 revenue total", _two_company_graph())

    assert pipeline.status == "ok"
    assert pipeline.answer == 2150.0


def test_unit_resolver_converts_bps_to_percent_and_blocks_cross_currency() -> None:
    resolver = UnitResolver()
    bps = resolver.detect("bps")
    percent = resolver.detect("%")
    assert resolver.normalize(100.0, bps, percent) == 1.0

    usd = resolver.detect("million USD")
    eur = resolver.detect("million EUR")
    warnings = resolver.check_compatibility("SUM", [usd, eur])
    assert any("Cross-currency" in warning for warning in warnings)


def test_unit_category_warning_has_deterministic_order() -> None:
    resolver = UnitResolver()

    warnings = resolver.check_compatibility(
        "GROWTH",
        [resolver.detect("%"), resolver.detect("million USD")],
    )

    assert (
        "Mismatched unit categories in GROWTH: ['money', 'percentage']. "
        "Values will be used without normalization."
    ) in warnings


def test_validation_blocks_cross_currency_sum() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="a:revenue",
                entity_id="a",
                company_name="A",
                field_name="revenue",
                field_label="revenue",
                value=1.0,
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="b:revenue",
                entity_id="b",
                company_name="B",
                field_name="revenue",
                field_label="revenue",
                value=1.0,
                unit="million EUR",
            ),
        )
    )
    plan = OperatorPlan(operator="SUM", slots={"target_field": _slot("revenue", "revenue")})

    report = validate_preconditions(graph, plan)

    assert "unit_incompatible" in report.blocking_violations


def test_pipeline_abstains_on_unit_incompatible_subgraph() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="a:revenue",
                entity_id="a",
                company_name="A",
                field_name="revenue",
                field_label="revenue",
                value=1.0,
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="b:revenue",
                entity_id="b",
                company_name="B",
                field_name="revenue",
                field_label="revenue",
                value=1.0,
                unit="million EUR",
            ),
        ),
        source_name="cross_currency.csv",
    )

    pipeline = run_operator_pipeline(
        "total revenue",
        graph,
        force_operator="SUM",
        enable_fallback=False,
    )
    payload = pipeline.to_dict()

    assert pipeline.status == "abstained"
    assert pipeline.answer is None
    assert payload["execution_metadata"]["abstain"] == {
        "triggered": True,
        "reason": "unit_incompatible",
        "detail": "unit_incompatible",
    }
    assert payload["hybrid_query_trace"]["abstain"]["reason"] == "unit_incompatible"


def test_pipeline_does_not_report_cross_operator_fallback_attempts() -> None:
    routing = RoutingResult(
        operator="LOOKUP",
        confidence=0.4,
        intent="lookup",
        fallback_operators=["AVG"],
    )
    pipeline = run_operator_pipeline(
        "unknown metric",
        _graph(),
        routing=routing,
        enable_fallback=True,
    )

    assert pipeline.status == "abstained"
    assert [attempt.operator for attempt in pipeline.attempts] == ["LOOKUP"]
    assert all(attempt.status == "precondition_violation" for attempt in pipeline.attempts)
    assert pipeline.abstain_trace["reason"] == "operator_precondition_failed"


def test_execute_with_fallback_raises_with_attempt_report_when_all_fail() -> None:
    routing = RoutingResult(
        operator="LOOKUP",
        confidence=0.4,
        intent="lookup",
        fallback_operators=["AVG"],
    )

    try:
        execute_with_fallback("unknown metric", _graph(), routing=routing)
    except ValueError as exc:
        assert "All operator attempts failed" in str(exc)
        assert "precondition_violation" in str(exc)
    else:
        raise AssertionError("execute_with_fallback should fail when all attempts fail")


def test_validation_reports_bad_composite_dependency() -> None:
    from graph_numeric.operators.operator_plan import CompositeOperatorPlan

    step = OperatorPlan(
        operator="SUM",
        slots={"target_field": _slot("revenue", "revenue")},
        depends_on={"entities": "$missing.entities"},
        trace={"step_id": "s1"},
    )
    report = validate_preconditions(_graph(), CompositeOperatorPlan(steps=(step,)))

    assert report.status == "precondition_violation"
    assert any("dependency_ref_unavailable" in item for item in report.blocking_violations)


def test_runtime_fallback_hybrid_router_keeps_rule_path_when_embedding_unavailable() -> None:
    routing = RuntimeFallbackHybridRouter("torch unavailable").route("total revenue")

    assert routing.operator == "SUM"
    assert routing.route_type == "hybrid_rule_runtime_fallback"
    assert routing.trace["embedding_unavailable"] is True


def test_pipeline_to_dict_exposes_hybrid_query_trace() -> None:
    pipeline = run_operator_pipeline("2024 revenue total", _graph())

    payload = pipeline.to_dict()

    trace = payload["hybrid_query_trace"]
    assert set(trace) >= {
        "trace_version",
        "strategy",
        "routing",
        "vector_grounding",
        "scalar_constraints",
        "selected_operator",
        "operator_execution",
        "subgraph_discovery",
        "error_attribution",
        "verification",
    }
    assert trace["strategy"] == "scalar_first"
    assert trace["routing"]["trace"]["cost_optimizer"]["strategy"] == "scalar_first"
    assert "rejected_chunk_ids" in trace["routing"]["trace"]["scalar_pruning"]["steps"][0]
    assert "vector_score_gap" in trace["routing"]["trace"]["cost_optimizer"]["features"]
    assert trace["routing"]["operator"] == "SUM"
    assert trace["vector_grounding"]["route_type"]
    assert trace["scalar_constraints"]["field_slots"]["target_field"] == "revenue"
    assert trace["attempts"][0]["status"] == "ok"
    assert trace["verification"]["passed"] is True
    assert "arithmetic_verified" in trace["verification"]["checks"]


def test_pipeline_exposes_runtime_evidence_arbitration_trace() -> None:
    pipeline = run_operator_pipeline(
        "what percentage of 2024 total revenue is cloud revenue?",
        _share_expression_graph(),
        evidence_arbitration_mode="shadow",
    )

    payload = pipeline.to_dict()
    trace = payload["hybrid_query_trace"]["evidence_arbitration"]

    assert payload["evidence_arbitration"] == trace
    assert trace["enabled"] is True
    assert trace["mode"] == "shadow"
    assert trace["policy"] == "safe_v5"
    assert trace["status"] in {"applied", "unavailable"}
    assert isinstance(trace["decisions"], list)


def test_share_self_binding_abstains_as_implausible_result() -> None:
    pipeline = run_operator_pipeline(
        "what percentage of 2024 total revenue is total revenue?",
        _share_expression_graph(),
        routing=RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle"),
        enable_fallback=False,
    )

    assert pipeline.status == "abstained"
    assert pipeline.answer is None
    assert pipeline.selected_operator == "SHARE"
    assert pipeline.abstain_trace["reason"] == "implausible_result"
    assert pipeline.attempts[-1].status == "implausible_result"
    assert str(pipeline.attempts[-1].error).startswith("implausible_result:share_self_binding")


def test_plausibility_gate_alone_allows_share_self_binding() -> None:
    pipeline = run_operator_pipeline(
        "what percentage of 2024 total revenue is total revenue?",
        _share_expression_graph(),
        routing=RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle"),
        enable_fallback=False,
        s5_gates=pipeline_module.S5GateConfig(plausibility=False),
    )

    assert pipeline.status == "ok"
    assert pipeline.answer == 1.0


def test_share_distinct_tokens_equal_one_is_allowed() -> None:
    pipeline = run_operator_pipeline(
        "what percentage of 2024 total revenue is cloud revenue?",
        _equal_value_share_expression_graph(),
        routing=RoutingResult(operator="SHARE", confidence=1.0, intent="ratio", route_type="oracle"),
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.selected_operator == "SHARE"
    assert pipeline.answer == 1.0
    assert pipeline.abstain_trace["triggered"] is False
    assert pipeline.result is not None
    assert pipeline.result.metadata["numerator_token_ids"] != pipeline.result.metadata["denominator_token_ids"]


def test_per_share_ratio_above_share_upper_bound_is_allowed() -> None:
    plan = OperatorPlan(
        operator="SHARE",
        slots={
            "numerator_field": _slot("net income", "net_income"),
            "denominator_field": _slot("common shares", "common_shares"),
        },
    )
    result = ExecutionResult(answer=1.56326, selected_tokens=(), calculation="217692 / 139255")

    trace = _implausible_result_trace(
        "what is the net income per common share for 2007?",
        "SHARE",
        result,
        plan,
    )

    assert trace["triggered"] is False


def test_difference_accepts_matching_share_units() -> None:
    units = [
        UnitResolver().detect("thousand shares"),
        UnitResolver().detect("thousand shares"),
    ]

    assert UnitResolver().check_compatibility("DIFFERENCE", units) == []


def test_product_route_returns_structured_result_instead_of_unknown_operator_exception() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="purchase:shares",
                entity_id="purchase",
                company_name="Alpha",
                field_name="shares_purchased",
                field_label="Shares purchased",
                value=3063816.0,
                unit="shares",
            ),
            AttributeValueToken(
                token_id="purchase:quarters",
                entity_id="purchase",
                company_name="Alpha",
                field_name="quarters",
                field_label="Quarters",
                value=4.0,
                unit="count",
            ),
        ),
        source_name="product_route",
    )

    pipeline = run_operator_pipeline(
        "had all four quarters had the same number of total shares purchased, how many total shares were purchased?",
        graph,
        routing=RoutingResult(operator="PRODUCT", confidence=1.0, intent="product", route_type="oracle"),
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.selected_operator == "PRODUCT"
    assert pipeline.answer == 12255264.0
    assert pipeline.attempts[-1].status == "ok"


def test_ratio_wording_with_non_ratio_single_token_answer_abstains_as_implausible_result() -> None:
    pipeline = run_operator_pipeline(
        "what percentage of total expected cash outflow is due in 2013?",
        _cash_outflow_share_graph(),
        routing=RoutingResult(operator="SUM", confidence=1.0, intent="sum", route_type="oracle"),
        force_operator="SUM",
        enable_fallback=False,
    )

    assert pipeline.status == "abstained"
    assert pipeline.answer is None
    assert pipeline.selected_operator == "SUM"
    assert pipeline.abstain_trace["reason"] == "implausible_result"
    assert pipeline.attempts[-1].answer == 23556.0


def test_pipeline_to_dict_exposes_unified_execution_metadata_and_report() -> None:
    pipeline = run_operator_pipeline("2024 revenue total", _graph())

    payload = pipeline.to_dict()

    metadata = payload["execution_metadata"]
    assert metadata["status"] == "ok"
    assert metadata["answer"] == 1200.0
    assert metadata["selected_token_ids"] == ["alpha:revenue"]
    assert metadata["evidence_trace_complete"] is True
    assert metadata["pipeline"]["selected_operator"] == "SUM"

    report = payload["verification_report"]
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["grouped_checks"]["arithmetic"]["arithmetic_verified"] is True
    assert report["grouped_checks"]["evidence"]["evidence_trace_complete"] is True


def test_pipeline_execution_metadata_exposes_subgraph_discovery() -> None:
    pipeline = run_operator_pipeline(
        "What was the change in net unrealized gains from 2017 to 2018?",
        _change_graph(),
        force_operator="DIFFERENCE",
        enable_fallback=False,
    )

    metadata = pipeline.to_dict()["execution_metadata"]
    discovery = metadata["subgraph_discovery"]

    assert discovery["graph_type"] == "implicit_attribute_value_graph"
    assert discovery["question_constraints"]["operator"] == "DIFFERENCE"
    assert discovery["candidate_token_count"] == 2
    assert discovery["calculation_subgraph"]["selected_token_ids"] == [
        "alpha:gain:2018",
        "alpha:gain:2017",
    ]
    assert discovery["slot_bindings"] == [
        {
            "slot": "left_time",
            "role": "被减数",
            "token_id": "alpha:gain:2018",
            "match_reason": "年份匹配 2018",
        },
        {
            "slot": "right_time",
            "role": "减数",
            "token_id": "alpha:gain:2017",
            "match_reason": "年份匹配 2017",
        },
    ]
    candidates_by_slot = {
        item["slot"]: item
        for item in discovery["slot_candidates"]
    }
    left_candidates = candidates_by_slot["left_time"]["candidates"]
    right_candidates = candidates_by_slot["right_time"]["candidates"]
    assert left_candidates[0]["token_id"] == "alpha:gain:2018"
    assert left_candidates[0]["selected"] is True
    assert left_candidates[0]["score"] > left_candidates[1]["score"]
    assert "年份匹配 2018" in left_candidates[0]["match_reasons"]
    assert left_candidates[0]["match_trace"]["metric"] == {
        "level": "rule_exact",
        "score": 1.0,
        "rationale": "field_name exact match: net_unrealized_gains",
    }
    assert left_candidates[1]["match_trace"]["metric"]["level"] == "rule_exact"
    assert left_candidates[1]["match_trace"]["time"] == {
        "level": "rule_exact",
        "score": 0.0,
        "rationale": "year mismatch: expected 2018, got 2017",
    }
    assert right_candidates[0]["token_id"] == "alpha:gain:2017"
    assert right_candidates[0]["selected"] is True
    assert right_candidates[0]["score"] > right_candidates[1]["score"]
    assert "年份匹配 2017" in right_candidates[0]["match_reasons"]
    assert discovery["consistency_checks"]["tokens_selected"] is True
    assert discovery["consistency_checks"]["units_present"] is True
    assert discovery["consistency_checks"]["single_metric_for_difference"] is True
    assert discovery["abstain"]["triggered"] is False


def test_subgraph_candidate_metric_trace_records_alias_table_match() -> None:
    token = AttributeValueToken(
        token_id="alpha:total_net_sales",
        entity_id="alpha",
        company_name="Alpha",
        field_name="total_net_sales",
        field_label="Total net sales",
        value=100.0,
        year=2024,
        unit="million USD",
    )
    row = _candidate_score_row(
        token,
        {
            "slot": "target_field",
            "role": "目标数值",
            "constraints": {"field_name": "net_sales"},
        },
        selected=False,
    )

    assert row["match_trace"]["metric"] == {
        "level": "alias_table",
        "score": 1.0,
        "rationale": "shared alias: net sales",
    }


def test_subgraph_candidate_metric_trace_records_nli_entailment_match() -> None:
    class StubNliProvider:
        def __init__(self) -> None:
            self.calls = []

        def judge_metric_equivalence(self, *, premise, hypothesis, token, expected_field):
            self.calls.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "token_id": token.token_id,
                    "expected_field": expected_field,
                }
            )
            return NliMetricJudgment(
                label="entailment",
                confidence=0.87,
                rationale="semantic metric equivalence",
                model_name="stub-nli",
            )

    token = AttributeValueToken(
        token_id="alpha:turnover_total",
        entity_id="alpha",
        company_name="Alpha",
        field_name="turnover_total",
        field_label="Total turnover",
        value=100.0,
        year=2024,
        unit="million USD",
    )
    provider = StubNliProvider()
    row = _candidate_score_row(
        token,
        {
            "slot": "target_field",
            "role": "目标数值",
            "constraints": {"field_name": "reported_revenue"},
        },
        selected=False,
        metric_matcher=MetricMatcher(nli_provider=provider, nli_accept_threshold=0.8),
    )

    assert row["match_trace"]["metric"] == {
        "level": "nli_entailment",
        "score": 0.87,
        "rationale": "semantic metric equivalence",
        "nli": {
            "premise": "Token metric: Total turnover (turnover_total).",
            "hypothesis": "Expected metric: reported revenue (reported_revenue).",
            "label": "entailment",
            "confidence": 0.87,
            "model": "stub-nli",
            "decision": "accept",
        },
    }
    assert row["score"] == 0.435
    assert provider.calls == [
        {
            "premise": "Token metric: Total turnover (turnover_total).",
            "hypothesis": "Expected metric: reported revenue (reported_revenue).",
            "token_id": "alpha:turnover_total",
            "expected_field": "reported_revenue",
        }
    ]


def test_metric_matcher_does_not_call_nli_when_rule_or_alias_matches() -> None:
    class FailingNliProvider:
        def judge_metric_equivalence(self, **kwargs):
            raise AssertionError("NLI should not run for deterministic matches")

    matcher = MetricMatcher(nli_provider=FailingNliProvider())
    exact_token = AttributeValueToken(
        token_id="alpha:revenue",
        entity_id="alpha",
        company_name="Alpha",
        field_name="revenue",
        field_label="Revenue",
        value=100.0,
    )
    alias_token = AttributeValueToken(
        token_id="alpha:total_net_sales",
        entity_id="alpha",
        company_name="Alpha",
        field_name="total_net_sales",
        field_label="Total net sales",
        value=100.0,
    )

    assert matcher.match(exact_token, "revenue")["level"] == "rule_exact"
    assert matcher.match(alias_token, "net_sales")["level"] == "alias_table"


def test_pipeline_passes_metric_matcher_into_subgraph_candidates() -> None:
    class StubNliProvider:
        def judge_metric_equivalence(self, *, premise, hypothesis, token, expected_field):
            if token.field_name == "turnover_total" and expected_field == "reported_revenue":
                return NliMetricJudgment(
                    label="entailment",
                    confidence=0.84,
                    rationale="turnover is used as reported revenue",
                    model_name="stub-nli",
                )
            return NliMetricJudgment(label="neutral", confidence=0.1, model_name="stub-nli")

    class ExactSolver:
        field_grounder = None

        def solve(self, query, graph, routing, *, operator=None):
            return OperatorPlan(
                operator="SUM",
                slots={"target_field": _slot("reported revenue", "reported_revenue", 1.0)},
            )

    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="alpha:reported_revenue",
                entity_id="alpha",
                company_name="Alpha",
                field_name="reported_revenue",
                field_label="Reported revenue",
                value=100.0,
                year=2024,
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="alpha:turnover_total",
                entity_id="alpha",
                company_name="Alpha",
                field_name="turnover_total",
                field_label="Total turnover",
                value=100.0,
                year=2024,
                unit="million USD",
            ),
        ),
        source_name="nli.csv",
    )
    pipeline = run_operator_pipeline(
        "What was Alpha reported revenue?",
        graph,
        routing=RoutingResult(operator="SUM", confidence=1.0, intent="sum"),
        solver=ExactSolver(),
        force_operator="SUM",
        enable_fallback=False,
        metric_matcher=MetricMatcher(
            nli_provider=StubNliProvider(),
            nli_accept_threshold=0.8,
        ),
    )
    candidates = pipeline.to_dict()["execution_metadata"]["subgraph_discovery"]["slot_candidates"]
    target_candidates = {
        item["slot"]: item
        for item in candidates
    }["target_field"]["candidates"]
    semantic_candidate = next(
        item for item in target_candidates
        if item["token_id"] == "alpha:turnover_total"
    )

    assert semantic_candidate["match_trace"]["metric"]["level"] == "nli_entailment"
    assert semantic_candidate["match_trace"]["metric"]["score"] == 0.84


def test_pipeline_allows_sum_target_fields_selected_tokens() -> None:
    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="mro:crude",
                entity_id="mro",
                company_name="MRO",
                field_name="miles_of_private_crude_oil_pipelines",
                field_label="miles of private crude oil pipelines",
                value=176.0,
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="mro:refined",
                entity_id="mro",
                company_name="MRO",
                field_name="miles_of_private_refined_products_pipelines",
                field_label="miles of private refined products pipelines",
                value=850.0,
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="mro:leased",
                entity_id="mro",
                company_name="MRO",
                field_name="miles_of_common_carrier_refined_product_pipelines",
                field_label="miles of common carrier refined product pipelines",
                value=217.0,
                unit="million USD",
            ),
        ),
        source_name="mro",
    )

    pipeline = run_operator_pipeline(
        "what was total miles of private crude oil pipelines and private refined products pipelines?",
        graph,
        routing=RoutingResult(operator="SUM", confidence=1.0, intent="sum"),
        force_operator="SUM",
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.answer == 1026.0
    assert [
        token.field_name
        for token in pipeline.result.selected_tokens
    ] == [
        "miles_of_private_crude_oil_pipelines",
        "miles_of_private_refined_products_pipelines",
    ]


def test_pipeline_records_low_faithfulness_and_downweights_slot_candidate() -> None:
    class LowFaithfulnessProvider:
        def judge_entailment(self, *, premise, hypothesis, token):
            return NliMetricJudgment(
                label="neutral",
                confidence=0.76,
                model_name="stub-faithfulness",
                metadata={"label_scores": {"entailment": 0.18, "neutral": 0.76}},
            )

    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="alpha:revenue",
                entity_id="alpha",
                company_name="Alpha",
                field_name="revenue",
                field_label="Revenue",
                value=100.0,
                year=2024,
                source=TokenSource(
                    document_id="doc",
                    text_excerpt="Alpha disclosed marketing cost of $100 million in 2024.",
                ),
                unit="million USD",
            ),
        ),
        source_name="doc",
    )
    pipeline = run_operator_pipeline(
        "What was Alpha revenue?",
        graph,
        routing=RoutingResult(operator="SUM", confidence=1.0, intent="sum"),
        force_operator="SUM",
        enable_fallback=False,
        faithfulness_checker=TokenFaithfulnessChecker(
            provider=LowFaithfulnessProvider(),
            entailment_threshold=0.5,
            score_penalty=0.2,
        ),
    )
    discovery = pipeline.to_dict()["execution_metadata"]["subgraph_discovery"]
    faithfulness = discovery["evidence_faithfulness"]
    candidate = discovery["slot_candidates"][0]["candidates"][0]

    assert faithfulness["summary"] == {
        "checked": 1,
        "low_faithfulness": 1,
        "missing_evidence": 0,
    }
    assert faithfulness["tokens"][0]["status"] == "low_faithfulness"
    assert faithfulness["tokens"][0]["hypothesis"] == (
        "Revenue for Alpha in 2024 was 100.0 million USD."
    )
    assert candidate["score"] == 0.4
    assert candidate["match_trace"]["faithfulness"]["status"] == "low_faithfulness"
    assert "证据蕴含低置信" in candidate["match_reasons"]


def test_pipeline_limits_faithfulness_checks_to_selected_and_top_slot_candidates() -> None:
    class RecordingFaithfulnessProvider:
        def __init__(self) -> None:
            self.checked_token_ids: list[str] = []

        def judge_entailment(self, *, premise, hypothesis, token):
            self.checked_token_ids.append(token.token_id)
            return NliMetricJudgment(
                label="entailment",
                confidence=0.92,
                model_name="stub-faithfulness",
                metadata={"label_scores": {"entailment": 0.92}},
            )

    graph = AttributeValueGraph(
        (
            AttributeValueToken(
                token_id="alpha:revenue:2024",
                entity_id="alpha:2024",
                company_name="Alpha",
                field_name="revenue",
                field_label="Revenue",
                value=100.0,
                year=2024,
                source=TokenSource(document_id="doc", text_excerpt="Revenue was $100 million in 2024."),
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="alpha:revenue:2023",
                entity_id="alpha:2023",
                company_name="Alpha",
                field_name="revenue",
                field_label="Revenue",
                value=90.0,
                year=2023,
                source=TokenSource(document_id="doc", text_excerpt="Revenue was $90 million in 2023."),
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="alpha:revenue:2022",
                entity_id="alpha:2022",
                company_name="Alpha",
                field_name="revenue",
                field_label="Revenue",
                value=80.0,
                year=2022,
                source=TokenSource(document_id="doc", text_excerpt="Revenue was $80 million in 2022."),
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="alpha:marketing:2024",
                entity_id="alpha:marketing:2024",
                company_name="Alpha",
                field_name="marketing_cost",
                field_label="Marketing cost",
                value=20.0,
                year=2024,
                source=TokenSource(document_id="doc", text_excerpt="Marketing cost was $20 million in 2024."),
                unit="million USD",
            ),
            AttributeValueToken(
                token_id="alpha:employees:2024",
                entity_id="alpha:employees:2024",
                company_name="Alpha",
                field_name="employees",
                field_label="Employees",
                value=12.0,
                year=2024,
                source=TokenSource(document_id="doc", text_excerpt="Alpha had 12 employees in 2024."),
            ),
        ),
        source_name="doc",
    )
    provider = RecordingFaithfulnessProvider()

    pipeline = run_operator_pipeline(
        "What was Alpha revenue in 2024?",
        graph,
        routing=RoutingResult(operator="SUM", confidence=1.0, intent="sum"),
        force_operator="SUM",
        enable_fallback=False,
        faithfulness_checker=TokenFaithfulnessChecker(provider=provider),
    )

    discovery = pipeline.to_dict()["execution_metadata"]["subgraph_discovery"]

    assert discovery["evidence_faithfulness"]["summary"]["checked"] == 3
    assert set(provider.checked_token_ids) == {
        "alpha:revenue:2024",
        "alpha:revenue:2023",
        "alpha:revenue:2022",
    }
    assert "alpha:marketing:2024" not in provider.checked_token_ids
    assert "alpha:employees:2024" not in provider.checked_token_ids


def test_pipeline_abstains_when_router_second_opinion_disagrees() -> None:
    class ConflictingSecondOpinion:
        def predict_operator(self, query):
            return RouterSecondOpinionResult(
                operator="COUNT",
                confidence=0.91,
                model_name="stub-router",
            )

    pipeline = run_operator_pipeline(
        "What was Alpha revenue?",
        _graph(),
        routing=RoutingResult(operator="SUM", confidence=1.0, intent="sum"),
        force_operator=None,
        enable_fallback=False,
        router_second_opinion=ConflictingSecondOpinion(),
    )
    payload = pipeline.to_dict()

    assert pipeline.status == "abstained"
    assert payload["abstain"]["reason"] == "router_disagreement"
    assert payload["abstain"]["detail"] == {
        "primary_operator": "SUM",
        "second_opinion_operator": "COUNT",
        "second_opinion_confidence": 0.91,
        "model": "stub-router",
    }
    assert payload["attempts"][0]["status"] == "router_disagreement"


def test_classifier_router_second_opinion_uses_predict_proba() -> None:
    class FakeClassifier:
        classes_ = ["SUM", "COUNT"]

        def predict_proba(self, rows):
            assert rows == ["How many companies?"]
            return [[0.1, 0.9]]

    result = ClassifierRouterSecondOpinion(
        classifier=FakeClassifier(),
        model_name="stub-classifier",
    ).predict_operator("How many companies?")

    assert result == RouterSecondOpinionResult(
        operator="COUNT",
        confidence=0.9,
        model_name="stub-classifier",
    )


def test_pipeline_abstains_when_slot_binding_is_ambiguous() -> None:
    pipeline = run_operator_pipeline(
        "What was the change in net unrealized gains from 2017 to 2018?",
        _ambiguous_change_graph(),
        force_operator="DIFFERENCE",
        enable_fallback=False,
    )
    payload = pipeline.to_dict()

    assert pipeline.status == "abstained"
    assert pipeline.answer is None
    abstain = payload["execution_metadata"]["abstain"]
    assert abstain["triggered"] is True
    assert abstain["reason"] == "ambiguous_binding"
    discovery = payload["execution_metadata"]["subgraph_discovery"]
    assert discovery["abstain"]["reason"] == "ambiguous_binding"
    left_candidates = {
        item["slot"]: item
        for item in discovery["slot_candidates"]
    }["left_time"]["candidates"]
    assert left_candidates[0]["score"] == left_candidates[1]["score"]


def test_difference_compare_to_binds_target_year_as_left_operand() -> None:
    pipeline = run_operator_pipeline(
        "What was the change in net unrealized gains in 2018 compared to 2017?",
        _change_graph(),
        routing=RoutingResult(operator="DIFFERENCE", confidence=1.0, intent="difference"),
        force_operator="DIFFERENCE",
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.answer == -6.0
    discovery = pipeline.to_dict()["execution_metadata"]["subgraph_discovery"]
    assert discovery["question_constraints"]["direction"] == {
        "start_year": 2017,
        "end_year": 2018,
        "expected_left_time": 2018,
        "expected_right_time": 2017,
    }


def test_difference_between_years_records_direction_constraint() -> None:
    pipeline = run_operator_pipeline(
        "What was the change in net unrealized gains between 2017 and 2018?",
        _change_graph(),
        routing=RoutingResult(operator="DIFFERENCE", confidence=1.0, intent="difference"),
        force_operator="DIFFERENCE",
        enable_fallback=False,
    )

    assert pipeline.status == "ok"
    assert pipeline.answer == -6.0
    discovery = pipeline.to_dict()["execution_metadata"]["subgraph_discovery"]
    assert discovery["question_constraints"]["direction"] == {
        "start_year": 2017,
        "end_year": 2018,
        "expected_left_time": 2018,
        "expected_right_time": 2017,
    }


def test_pipeline_abstains_when_difference_direction_conflicts_with_question() -> None:
    class ReversedDifferenceSolver:
        field_grounder = None

        def solve(self, query, graph, routing, *, operator=None):
            return OperatorPlan(
                operator="DIFFERENCE",
                slots={
                    "target_field": _slot("net unrealized gains", "net_unrealized_gains", 1.0),
                    "left_time": _slot("2017", 2017, 1.0),
                    "right_time": _slot("2018", 2018, 1.0),
                },
            )

    pipeline = run_operator_pipeline(
        "What was the change in net unrealized gains from 2017 to 2018?",
        _change_graph(),
        routing=RoutingResult(operator="DIFFERENCE", confidence=1.0, intent="difference"),
        solver=ReversedDifferenceSolver(),
        force_operator="DIFFERENCE",
        enable_fallback=False,
    )
    payload = pipeline.to_dict()

    assert pipeline.status == "abstained"
    assert pipeline.answer is None
    abstain = payload["execution_metadata"]["abstain"]
    assert abstain["triggered"] is True
    assert abstain["reason"] == "direction_violation"
    discovery = payload["execution_metadata"]["subgraph_discovery"]
    assert discovery["consistency_checks"]["direction_matches_question"] is False
    assert discovery["question_constraints"]["direction"] == {
        "start_year": 2017,
        "end_year": 2018,
        "expected_left_time": 2018,
        "expected_right_time": 2017,
    }


def test_hybrid_trace_reports_phase2_error_attribution_fields() -> None:
    pipeline = run_operator_pipeline("2024 revenue total", _graph())

    trace = pipeline.to_dict()["hybrid_query_trace"]

    assert trace["trace_version"] == "hybrid_trace_v2"
    assert trace["operator_execution"]["selected_operator"] == "SUM"
    assert trace["operator_execution"]["answer"] == 1200.0
    assert trace["error_attribution"] == {
        "extraction": False,
        "grounding": False,
        "planning": False,
        "unit": False,
        "filter": False,
        "arithmetic": False,
        "evidence": False,
    }
