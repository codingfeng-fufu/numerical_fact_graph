"""Tests for the Verifier module.

Covers:
  - Happy path: all 5 checks pass for every operator
  - Failure cases: each check triggered individually
  - Integration: execute() auto-populates result.checks
  - Source tracking: evidence_trace_complete check
"""
from __future__ import annotations

import dataclasses

import pytest

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource
from graph_numeric.operators.executor import ExecutionResult, execute
from graph_numeric.operators.operator_plan import OperatorPlan, Slot
from graph_numeric.operators.verifier import VerificationReport, Verifier


# ---- shared fixtures ----

def _token(
    entity: str,
    field: str,
    value: float,
    year: int | None = 2024,
    industry: str = "TECH",
) -> AttributeValueToken:
    return AttributeValueToken(
        token_id=f"{entity}:{field}",
        entity_id=entity,
        company_name=entity,
        field_name=field,
        field_label=field,
        value=value,
        year=year,
        industry=industry,
    )


def _slot(surface: str, value: object, conf: float = 0.9) -> Slot:
    return Slot(surface=surface, grounded_value=value, confidence=conf)


@pytest.fixture
def graph() -> AttributeValueGraph:
    return AttributeValueGraph(
        (
            _token("Alpha", "revenue", 1200.0, 2024),
            _token("Beta",  "revenue",  950.0, 2024),
            _token("Gamma", "revenue", 3200.0, 2024),
            _token("Alpha", "net_profit", 120.0, 2024),
            _token("Beta",  "net_profit",  95.0, 2024),
            _token("Gamma", "net_profit", 320.0, 2024),
            _token("Alpha", "revenue", 1000.0, 2023),
            _token("Beta",  "revenue",  800.0, 2023),
        ),
        source_name="verifier_test",
    )


@pytest.fixture
def verifier() -> Verifier:
    return Verifier()


ALL_CHECKS = {"plan_valid", "field_exists", "tokens_nonempty",
              "tokens_field_consistent", "arithmetic_verified",
              "evidence_trace_complete"}


def _assert_all_pass(report: VerificationReport) -> None:
    assert report.passed, f"expected all computational checks to pass, warnings={report.warnings}"
    assert ALL_CHECKS.issubset(report.checks.keys())
    for k in ("plan_valid", "field_exists", "tokens_nonempty",
              "tokens_field_consistent", "arithmetic_verified"):
        assert report.checks[k] is True, f"check '{k}' failed; warnings={report.warnings}"


def test_verification_report_groups_failure_categories() -> None:
    report = VerificationReport(
        checks={
            "plan_valid": False,
            "field_exists": False,
            "unit_compatible": False,
            "arithmetic_verified": True,
            "evidence_trace_complete": True,
        },
        warnings=["plan_valid: missing required slots {'target_field'}"],
        passed=False,
    )

    payload = report.to_dict()

    assert payload["passed"] is False
    assert payload["failed_checks"] == [
        "plan_valid",
        "field_exists",
        "unit_compatible",
    ]
    assert payload["failure_categories"] == ["grounding", "planning", "unit"]
    assert payload["grouped_checks"]["planning"]["plan_valid"] is False
    assert payload["grouped_checks"]["unit"]["unit_compatible"] is False


# ---- Happy path: one test per operator ----

class TestHappyPath:

    def test_sum(self, graph, verifier):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert result.answer == 1200 + 950 + 3200

    def test_count(self, graph, verifier):
        plan = OperatorPlan("COUNT", slots={
            "count_target": _slot("company", "company"),
            "condition_field": _slot("revenue", "revenue"),
            "condition_op": _slot(">=", ">="),
            "condition_threshold": _slot("1000", 1000.0),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert result.answer == 2.0  # Alpha + Gamma >= 1000

    def test_avg(self, graph, verifier):
        plan = OperatorPlan("AVG", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert abs(result.answer - (1200 + 950 + 3200) / 3) < 1e-6

    def test_max(self, graph, verifier):
        plan = OperatorPlan("MAX", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert result.answer == 3200.0

    def test_min(self, graph, verifier):
        plan = OperatorPlan("MIN", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert result.answer == 950.0

    def test_argmax(self, graph, verifier):
        plan = OperatorPlan("ARGMAX", slots={
            "target_entity_type": _slot("company", "company"),
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert result.answer == 3200.0

    def test_ratio(self, graph, verifier):
        plan = OperatorPlan("RATIO", slots={
            "numerator_field": _slot("net_profit", "net_profit"),
            "denominator_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        expected = (120 + 95 + 320) / (1200 + 950 + 3200)
        assert abs(result.answer - expected) < 1e-6

    def test_growth(self, graph, verifier):
        plan = OperatorPlan("GROWTH", slots={
            "target_field": _slot("revenue", "revenue"),
            "entity": _slot("Alpha", "Alpha"),
            "from_time": _slot("2023", 2023),
            "to_time": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert abs(result.answer - 0.2) < 1e-6  # (1200-1000)/1000

    def test_difference(self, graph, verifier):
        plan = OperatorPlan("DIFFERENCE", slots={
            "target_field": _slot("revenue", "revenue"),
            "left_entity": _slot("Gamma", "Gamma"),
            "right_entity": _slot("Alpha", "Alpha"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert abs(result.answer - (3200 - 1200)) < 1e-6

    def test_lookup(self, graph, verifier):
        plan = OperatorPlan("LOOKUP", slots={
            "entity": _slot("Beta", "Beta"),
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        assert result.answer == 950.0

    def test_top_k(self, graph, verifier):
        plan = OperatorPlan("TOP_K", slots={
            "target_entity_type": _slot("company", "company"),
            "target_field": _slot("revenue", "revenue"),
            "k": _slot("2", 2),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        _assert_all_pass(report)
        # top-2 are Gamma (3200) and Alpha (1200); k-th value = 1200
        assert result.answer == 1200.0
        assert len(result.selected_tokens) == 2

    def test_trend_increasing(self, verifier):
        # Build a graph with 3 years of strictly increasing revenue for one company
        tokens = tuple(
            _token("Alpha", "revenue", float(v), yr)
            for yr, v in [(2022, 1000), (2023, 1200), (2024, 1500)]
        )
        g = AttributeValueGraph(tokens, source_name="trend_test")
        plan = OperatorPlan("TREND", slots={
            "target_field": _slot("revenue", "revenue"),
            "from_time": _slot("2022", 2022),
            "to_time": _slot("2024", 2024),
        })
        result = execute(g, plan)
        report = verifier.verify(g, plan, result)
        _assert_all_pass(report)
        assert result.answer == 1.0  # increasing

    def test_trend_decreasing(self, verifier):
        tokens = tuple(
            _token("Alpha", "revenue", float(v), yr)
            for yr, v in [(2022, 1500), (2023, 1200), (2024, 900)]
        )
        g = AttributeValueGraph(tokens, source_name="trend_test")
        plan = OperatorPlan("TREND", slots={
            "target_field": _slot("revenue", "revenue"),
            "from_time": _slot("2022", 2022),
            "to_time": _slot("2024", 2024),
        })
        result = execute(g, plan)
        assert result.answer == -1.0  # decreasing

    def test_trend_fluctuating(self, verifier):
        tokens = tuple(
            _token("Alpha", "revenue", float(v), yr)
            for yr, v in [(2022, 1000), (2023, 1500), (2024, 1200)]
        )
        g = AttributeValueGraph(tokens, source_name="trend_test")
        plan = OperatorPlan("TREND", slots={
            "target_field": _slot("revenue", "revenue"),
            "from_time": _slot("2022", 2022),
            "to_time": _slot("2024", 2024),
        })
        result = execute(g, plan)
        assert result.answer == 0.0  # fluctuating


# ---- Failure cases ----

class TestFailureCases:

    def test_plan_valid_fails_on_missing_required_slot(self, graph, verifier):
        # SUM without target_field → plan_valid=False, but execute() raises before verifier
        # so we test verifier directly with a manually crafted partial result
        plan = OperatorPlan("SUM", slots={})
        fake_result = ExecutionResult(answer=0.0, selected_tokens=(), calculation="")
        report = verifier.verify(graph, plan, fake_result)
        assert report.checks["plan_valid"] is False
        assert report.passed is False

    def test_field_exists_fails_for_unknown_field(self, graph, verifier):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("headcount", "headcount"),  # not in graph
        })
        fake_result = ExecutionResult(answer=0.0, selected_tokens=(), calculation="")
        report = verifier.verify(graph, plan, fake_result)
        assert report.checks["field_exists"] is False
        assert report.passed is False

    def test_tokens_nonempty_fails_when_answer_nonzero_but_no_tokens(self, graph, verifier):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
        })
        fake_result = ExecutionResult(answer=999.0, selected_tokens=(), calculation="")
        report = verifier.verify(graph, plan, fake_result)
        assert report.checks["tokens_nonempty"] is False

    def test_tokens_nonempty_passes_when_answer_zero_and_no_tokens(self, graph, verifier):
        # COUNT with 0 results is legitimate
        plan = OperatorPlan("COUNT", slots={
            "count_target": _slot("company", "company"),
            "condition_field": _slot("revenue", "revenue"),
            "condition_op": _slot(">", ">"),
            "condition_threshold": _slot("99999", 99999.0),
        })
        result = execute(graph, plan)
        assert result.answer == 0.0
        report = verifier.verify(graph, plan, result)
        assert report.checks["tokens_nonempty"] is True

    def test_tokens_field_consistent_fails_for_wrong_field_in_tokens(self, graph, verifier):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
        })
        # Mix in a net_profit token — inconsistent
        wrong_token = _token("Alpha", "net_profit", 120.0)
        fake_result = ExecutionResult(
            answer=120.0,
            selected_tokens=(wrong_token,),
            calculation="120.0 = 120.0",
        )
        report = verifier.verify(graph, plan, fake_result)
        assert report.checks["tokens_field_consistent"] is False

    def test_arithmetic_fails_when_answer_tampered(self, graph, verifier):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        real_result = execute(graph, plan)
        # Tamper the answer
        tampered = dataclasses.replace(real_result, answer=real_result.answer + 1.0)
        report = verifier.verify(graph, plan, tampered)
        assert report.checks["arithmetic_verified"] is False

    def test_arithmetic_ratio_fails_on_wrong_answer(self, graph, verifier):
        plan = OperatorPlan("RATIO", slots={
            "numerator_field": _slot("net_profit", "net_profit"),
            "denominator_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        real_result = execute(graph, plan)
        tampered = dataclasses.replace(real_result, answer=0.5)
        report = verifier.verify(graph, plan, tampered)
        assert report.checks["arithmetic_verified"] is False

    def test_arithmetic_growth_fails_on_wrong_answer(self, graph, verifier):
        plan = OperatorPlan("GROWTH", slots={
            "target_field": _slot("revenue", "revenue"),
            "entity": _slot("Alpha", "Alpha"),
            "from_time": _slot("2023", 2023),
            "to_time": _slot("2024", 2024),
        })
        real_result = execute(graph, plan)
        tampered = dataclasses.replace(real_result, answer=0.5)
        report = verifier.verify(graph, plan, tampered)
        assert report.checks["arithmetic_verified"] is False


# ---- Integration: execute() auto-populates checks ----

_COMPUTATIONAL = {"plan_valid", "field_exists", "tokens_nonempty",
                  "tokens_field_consistent", "arithmetic_verified"}


def _assert_computational_pass(checks: dict) -> None:
    failed = {k: v for k, v in checks.items() if k in _COMPUTATIONAL and not v}
    assert not failed, f"computational checks failed: {failed}"


class TestExecuteIntegration:

    def test_execute_populates_checks_for_sum(self, graph):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        assert ALL_CHECKS.issubset(result.checks.keys())
        _assert_computational_pass(result.checks)

    def test_execute_populates_checks_for_count(self, graph):
        plan = OperatorPlan("COUNT", slots={
            "count_target": _slot("company", "company"),
            "condition_field": _slot("revenue", "revenue"),
            "condition_op": _slot(">=", ">="),
            "condition_threshold": _slot("1000", 1000.0),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        _assert_computational_pass(result.checks)

    def test_execute_populates_checks_for_ratio(self, graph):
        plan = OperatorPlan("RATIO", slots={
            "numerator_field": _slot("net_profit", "net_profit"),
            "denominator_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        _assert_computational_pass(result.checks)

    def test_execute_populates_checks_for_growth(self, graph):
        plan = OperatorPlan("GROWTH", slots={
            "target_field": _slot("revenue", "revenue"),
            "entity": _slot("Alpha", "Alpha"),
            "from_time": _slot("2023", 2023),
            "to_time": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        _assert_computational_pass(result.checks)

    def test_execute_populates_checks_for_top_k(self, graph):
        plan = OperatorPlan("TOP_K", slots={
            "target_entity_type": _slot("company", "company"),
            "target_field": _slot("revenue", "revenue"),
            "k": _slot("2", 2),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        _assert_computational_pass(result.checks)
        assert result.answer == 1200.0  # 2nd largest

    def test_verification_report_passed_flag(self, graph, verifier):
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        assert report.passed is True
        # evidence_trace_complete may be False for synthetic tokens (no source), that's OK
        assert report.checks["plan_valid"] is True
        assert report.checks["arithmetic_verified"] is True


# ---- Source tracking: evidence_trace_complete ----

class TestEvidenceTrace:

    def test_evidence_trace_false_for_tokens_without_source(self, graph, verifier):
        # The graph fixture uses synthetic tokens (source=None)
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        # Synthetic tokens have no source — advisory check should be False
        assert report.checks["evidence_trace_complete"] is False
        # But passed must still be True (evidence_trace is not a computational check)
        assert report.passed is True

    def test_evidence_trace_true_for_tokens_with_source(self, verifier):
        src = TokenSource(document_id="test_doc", table="t", row=0, column="revenue")
        token = AttributeValueToken(
            token_id="Alpha:revenue",
            entity_id="Alpha",
            company_name="Alpha",
            field_name="revenue",
            field_label="revenue",
            value=1200.0,
            year=2024,
            source=src,
        )
        graph = AttributeValueGraph((token,), source_name="test_doc")
        plan = OperatorPlan("SUM", slots={
            "target_field": _slot("revenue", "revenue"),
            "year": _slot("2024", 2024),
        })
        result = execute(graph, plan)
        report = verifier.verify(graph, plan, result)
        assert report.checks["evidence_trace_complete"] is True
        assert report.passed is True

    def test_evidence_trace_true_for_empty_tokens(self, graph, verifier):
        # COUNT with 0 results → no tokens → trace trivially complete
        plan = OperatorPlan("COUNT", slots={
            "count_target": _slot("company", "company"),
            "condition_field": _slot("revenue", "revenue"),
            "condition_op": _slot(">", ">"),
            "condition_threshold": _slot("99999", 99999.0),
        })
        result = execute(graph, plan)
        assert result.answer == 0.0
        report = verifier.verify(graph, plan, result)
        assert report.checks["evidence_trace_complete"] is True
