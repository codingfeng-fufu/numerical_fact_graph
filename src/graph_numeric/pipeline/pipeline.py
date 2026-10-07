from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

from graph_numeric.audit.s5_audit import S5PredicateEvent
from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
    FIELD_ALIASES,
    field_aliases,
    normalize_identifier,
    token_matches_year,
)
from graph_numeric.binding.s4prime import (
    S4PrimeAbstainedError,
    S4PrimeBinder,
    S4PrimeInvalidProposalError,
)
from graph_numeric.operators.executor import ExecutionResult, execute, execute_composite
from graph_numeric.learning.evidence_arbitration_runtime import (
    DEFAULT_SAFEV5_RANKER_MODEL,
    EvidenceArbitrationConfig,
    apply_runtime_evidence_arbitration,
)
from graph_numeric.learning.evidence_faithfulness import (
    TokenFaithfulnessChecker,
    evaluate_token_faithfulness,
)
from graph_numeric.learning.metric_matcher import DEFAULT_METRIC_MATCHER, MetricMatcher
from graph_numeric.learning.router_second_opinion import (
    RouterSecondOpinion,
    router_second_opinion_abstain_trace,
)
from graph_numeric.core.expression_plan import parse_expression_plan
from graph_numeric.learning.field_grounder import FieldGrounder
from graph_numeric.pipeline.hybrid_query_engine import analyze_hybrid_query
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot
from graph_numeric.operators.operator_solvers import (
    OperatorSolver,
    resolve_review_hardening_field,
    solve_composite_operator_plan,
)
from graph_numeric.operators.operator_registry import OPERATOR_REGISTRY
from graph_numeric.learning.router import RoutingResult, RuleBasedRouter
from graph_numeric.core.unit_resolver import UnitResolver
from graph_numeric.operators.validation import validate_preconditions

_BLOCKING_CHECKS = frozenset({
    "plan_valid",
    "field_exists",
    "tokens_nonempty",
    "tokens_field_consistent",
    "filter_satisfied",
    "unit_compatible",
    "arithmetic_verified",
    "output_type_valid",
})
_AMBIGUOUS_BINDING_MARGIN = 0.05


@dataclass(frozen=True)
class S5GateConfig:
    preconditions: bool = True
    unit: bool = True
    verifier: bool = True
    plausibility: bool = True

    @classmethod
    def disabled(cls) -> "S5GateConfig":
        return cls(False, False, False, False)

    def to_dict(self) -> dict[str, bool]:
        return dataclasses.asdict(self)

    def any_enabled(self) -> bool:
        return any(self.to_dict().values())


def _unit_precondition_violation(violation: str) -> bool:
    return str(violation).split(":")[-1] == "unit_incompatible"


def _enabled_precondition_violations(
    violations: tuple[str, ...] | list[str],
    gates: S5GateConfig,
) -> list[str]:
    return [
        violation
        for violation in violations
        if (gates.unit if _unit_precondition_violation(violation) else gates.preconditions)
    ]


def _enabled_blocking_checks(
    checks: dict[str, bool],
    gates: S5GateConfig,
) -> list[str]:
    return [
        name
        for name, ok in checks.items()
        if not ok
        and name in _BLOCKING_CHECKS
        and (gates.unit if name == "unit_compatible" else gates.verifier)
    ]


@dataclass(frozen=True)
class PipelineAttempt:
    operator: str
    status: str
    route_type: str
    validation: dict[str, Any] | None = None
    error: str | None = None
    answer: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "status": self.status,
            "route_type": self.route_type,
            "validation": self.validation,
            "error": self.error,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class PipelineResult:
    result: ExecutionResult | None
    routing: RoutingResult
    plan: OperatorPlan | CompositeOperatorPlan | None
    attempts: tuple[PipelineAttempt, ...]
    fallback_used: bool
    original_operator: str
    selected_operator: str | None
    status: str
    evidence_arbitration_trace: dict[str, Any] = dataclasses.field(default_factory=dict)
    abstain_trace: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def answer(self) -> Any | None:
        if self.status == "abstained":
            return None
        return self.result.answer if self.result is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "fallback_used": self.fallback_used,
            "original_operator": self.original_operator,
            "selected_operator": self.selected_operator,
            "routing": self.routing.to_dict(),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "answer": self.answer,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "execution_metadata": self.execution_metadata(),
            "verification_report": self.verification_report(),
            "hybrid_query_trace": self.hybrid_query_trace(),
            "evidence_arbitration": dict(self.evidence_arbitration_trace or {}),
            "abstain": dict(self.abstain_trace or _empty_abstain_trace()),
        }

    def execution_metadata(self) -> dict[str, Any]:
        result = self.result
        if result is None:
            return {
                "status": self.status,
                "answer": None,
                "calculation": None,
                "checks": {},
                "records_used": [],
                "selected_token_ids": [],
                "evidence_trace_complete": False,
                "output_unit": None,
                "abstain": dict(self.abstain_trace or _empty_abstain_trace()),
                "pipeline": {
                    "status": self.status,
                    "fallback_used": self.fallback_used,
                    "original_operator": self.original_operator,
                    "selected_operator": self.selected_operator,
                    "routing": self.routing.to_dict(),
                    "attempts": [attempt.to_dict() for attempt in self.attempts],
                },
            }
        metadata = dict(result.metadata or {})
        return {
            "status": self.status,
            "answer": self.answer,
            "calculation": result.calculation,
            "checks": dict(result.checks or {}),
            "records_used": list(metadata.get("records_used", [])),
            "selected_token_ids": list(
                metadata.get(
                    "selected_token_ids",
                    [token.token_id for token in result.selected_tokens],
                )
            ),
            "evidence_trace_complete": bool(
                metadata.get(
                    "evidence_trace_complete",
                    all(token.source is not None for token in result.selected_tokens),
                )
            ),
            "output_unit": metadata.get("output_unit"),
            "executor_operator": metadata.get("executor_operator"),
            "replay_operands": list(metadata.get("replay_operands", [])),
            "subgraph_discovery": dict(metadata.get("subgraph_discovery") or {}),
            "abstain": dict(self.abstain_trace or _empty_abstain_trace()),
            "pipeline": dict(
                metadata.get(
                    "pipeline",
                    {
                        "status": self.status,
                        "fallback_used": self.fallback_used,
                        "original_operator": self.original_operator,
                        "selected_operator": self.selected_operator,
                        "routing": self.routing.to_dict(),
                        "attempts": [attempt.to_dict() for attempt in self.attempts],
                    },
                )
            ),
        }

    def verification_report(self) -> dict[str, Any]:
        if self.result is None:
            return _verification_report_from_checks(
                {},
                warnings=["pipeline_result_missing: no execution result was produced"],
                passed=False,
            )
        checks = dict(self.result.checks or {})
        return _verification_report_from_checks(
            checks,
            warnings=[],
            passed=bool(checks) and all(checks.values()),
        )

    def hybrid_query_trace(self) -> dict[str, Any]:
        checks = dict(self.result.checks) if self.result is not None else {}
        metadata = dict(self.result.metadata or {}) if self.result is not None else {}
        verification_report = self.verification_report()
        return {
            "trace_version": "hybrid_trace_v2",
            "strategy": _hybrid_strategy(self.routing),
            "routing": self.routing.to_dict(),
            "vector_grounding": _vector_grounding_trace(self.routing, self.plan),
            "scalar_constraints": _scalar_constraint_trace(self.plan),
            "selected_operator": self.selected_operator,
            "fallback_used": self.fallback_used,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "operator_execution": {
                "selected_operator": self.selected_operator,
                "answer": self.answer,
                "status": self.status,
                "calculation": self.result.calculation if self.result is not None else None,
            },
            "subgraph_discovery": dict(metadata.get("subgraph_discovery") or {}),
            "abstain": dict(self.abstain_trace or _empty_abstain_trace()),
            "evidence_arbitration": dict(self.evidence_arbitration_trace or {}),
            "error_attribution": _error_attribution_from_report(
                verification_report,
                self.status,
            ),
            "verification": {
                "passed": bool(checks) and all(checks.values()),
                "checks": checks,
            },
        }


def _empty_abstain_trace() -> dict[str, Any]:
    return {
        "triggered": False,
        "reason": None,
        "detail": None,
    }


def _abstain_trace_from_attempts(attempts: list[PipelineAttempt]) -> dict[str, Any]:
    s4prime_abstain_attempts = [
        attempt for attempt in attempts if attempt.status == "s4prime_abstained"
    ]
    if s4prime_abstain_attempts:
        detail = ";".join(
            str(attempt.error or "s4prime_abstained")
            for attempt in s4prime_abstain_attempts
        )
        return {
            "triggered": True,
            "reason": "s4prime_abstained",
            "detail": detail,
        }
    invalid_proposal_attempts = [
        attempt for attempt in attempts if attempt.status == "invalid_proposal"
    ]
    if invalid_proposal_attempts:
        detail = ";".join(
            str(attempt.error or "invalid_proposal")
            for attempt in invalid_proposal_attempts
        )
        return {
            "triggered": True,
            "reason": "invalid_proposal",
            "detail": detail,
        }
    for attempt in attempts:
        validation = attempt.validation or {}
        blocking = [str(item) for item in validation.get("blocking_violations", [])]
        error = str(attempt.error or "")
        if "unit_incompatible" in blocking or "unit_compatible" in error:
            detail_items = blocking or ([error] if error else [])
            return {
                "triggered": True,
                "reason": "unit_incompatible",
                "detail": ";".join(detail_items) if detail_items else "unit_incompatible",
            }
    failed_operator_attempts = [
        attempt
        for attempt in attempts
        if attempt.status
        in {"solve_error", "precondition_violation", "verification_failed", "execution_error"}
    ]
    if failed_operator_attempts:
        details = []
        for attempt in failed_operator_attempts:
            detail = str(attempt.error or attempt.status or "")
            details.append(f"{attempt.operator}:{attempt.status}:{detail}")
        return {
            "triggered": True,
            "reason": "operator_precondition_failed",
            "detail": ";".join(details),
        }
    return _empty_abstain_trace()


def _abstain_trace_from_subgraph_discovery(discovery: Any) -> dict[str, Any]:
    if isinstance(discovery, dict):
        abstain = discovery.get("abstain")
        if isinstance(abstain, dict) and abstain.get("triggered"):
            return {
                "triggered": True,
                "reason": abstain.get("reason"),
                "detail": abstain.get("detail"),
            }
    return _empty_abstain_trace()


def _abstain_trace_from_slot_candidates(
    operator: str,
    slot_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if operator.upper() not in {
        "DIFFERENCE",
        "COMPARE",
        "GROWTH",
        "PERCENT_CHANGE",
    }:
        return _empty_abstain_trace()
    for slot_row in slot_candidates:
        candidates = slot_row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            continue
        first, second = candidates[0], candidates[1]
        first_score = _float_for_trace(first.get("score") if isinstance(first, dict) else None)
        second_score = _float_for_trace(second.get("score") if isinstance(second, dict) else None)
        if first_score is None or second_score is None or first_score <= 0:
            continue
        first_selected = bool(first.get("selected")) if isinstance(first, dict) else False
        second_selected = bool(second.get("selected")) if isinstance(second, dict) else False
        if not (first_selected or second_selected):
            continue
        first_value = _float_for_trace(first.get("value") if isinstance(first, dict) else None)
        second_value = _float_for_trace(second.get("value") if isinstance(second, dict) else None)
        if first_value is not None and second_value is not None and first_value == second_value:
            continue
        if first_score - second_score < _AMBIGUOUS_BINDING_MARGIN:
            first_id = first.get("token_id") if isinstance(first, dict) else None
            second_id = second.get("token_id") if isinstance(second, dict) else None
            slot = str(slot_row.get("slot") or "unknown")
            return {
                "triggered": True,
                "reason": "ambiguous_binding",
                "detail": (
                    f"slot {slot} top-1 {first_id} score={first_score} vs "
                    f"top-2 {second_id} score={second_score}, margin < {_AMBIGUOUS_BINDING_MARGIN}"
                ),
            }
    return _empty_abstain_trace()


def _abstain_trace_from_direction_consistency(
    checks: dict[str, bool],
    direction_constraint: dict[str, Any] | None,
) -> dict[str, Any]:
    if direction_constraint is None:
        return _empty_abstain_trace()
    if checks.get("direction_matches_question", True):
        return _empty_abstain_trace()
    return {
        "triggered": True,
        "reason": "direction_violation",
        "detail": (
            f"expected left_time={direction_constraint.get('expected_left_time')} "
            f"and right_time={direction_constraint.get('expected_right_time')} "
            "from question direction"
        ),
    }


def _float_for_trace(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _direction_constraint_from_query(query: str, operator: str) -> dict[str, Any] | None:
    compare_to_match = re.search(
        r"\b(?:in|for|during)\s+((?:19|20)\d{2})\b[^.?!]{0,80}\bcompar(?:e|ed|ing)?\s+(?:to|with)\s+((?:19|20)\d{2})\b",
        query,
        re.IGNORECASE,
    )
    if compare_to_match is not None:
        start_year = int(compare_to_match.group(2))
        end_year = int(compare_to_match.group(1))
    else:
        between_match = re.search(
            r"\bbetween\s+((?:19|20)\d{2})\b\s+and\s+((?:19|20)\d{2})\b",
            query,
            re.IGNORECASE,
        )
        if between_match is not None:
            start_year = int(between_match.group(1))
            end_year = int(between_match.group(2))
        else:
            start_year = end_year = None
    match = re.search(
        r"\bfrom\b[^0-9]{0,80}\b((?:19|20)\d{2})\b[^0-9]{0,80}\bto\b[^0-9]{0,80}\b((?:19|20)\d{2})\b",
        query,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"从[^0-9]{0,80}\b((?:19|20)\d{2})\b[^0-9]{0,80}到[^0-9]{0,80}\b((?:19|20)\d{2})\b",
            query,
        )
    if start_year is None or end_year is None:
        if match is None:
            return None
        start_year = int(match.group(1))
        end_year = int(match.group(2))
    canonical = OPERATOR_REGISTRY.canonical_executor_operator(operator)
    if canonical in {"DIFFERENCE", "COMPARE"}:
        lowered = query.casefold()
        if any(term in lowered for term in ("decrease", "decline", "reduction")):
            return {
                "start_year": start_year,
                "end_year": end_year,
                "expected_left_time": start_year,
                "expected_right_time": end_year,
            }
        return {
            "start_year": start_year,
            "end_year": end_year,
            "expected_left_time": end_year,
            "expected_right_time": start_year,
        }
    if canonical == "GROWTH":
        return {
            "start_year": start_year,
            "end_year": end_year,
            "expected_from_time": start_year,
            "expected_to_time": end_year,
        }
    return None


def run_operator_pipeline(
    query: str,
    graph: AttributeValueGraph,
    *,
    router: Any | None = None,
    routing: RoutingResult | None = None,
    solver: OperatorSolver | None = None,
    field_grounder: FieldGrounder | None = None,
    force_operator: str | None = None,
    preferred_operators: list[str] | tuple[str, ...] | None = None,
    enable_fallback: bool = True,
    evidence_arbitration_mode: str = "off",
    evidence_ranker_model_path: str | None = None,
    metric_matcher: MetricMatcher | None = None,
    faithfulness_checker: TokenFaithfulnessChecker | None = None,
    router_second_opinion: RouterSecondOpinion | None = None,
    router_second_opinion_min_confidence: float = 0.0,
    binder_mode: str = "baseline",
    s4prime_binder: Any | None = None,
    enable_s5_validation: bool = True,
    s5_gates: S5GateConfig | None = None,
    audit_sink: Callable[[S5PredicateEvent], None] | None = None,
    disabled_s5_predicates: frozenset[str] = frozenset(),
) -> PipelineResult:
    router = router or RuleBasedRouter()
    evidence_model_path = (
        Path(evidence_ranker_model_path)
        if evidence_ranker_model_path is not None
        else DEFAULT_SAFEV5_RANKER_MODEL
    )
    evidence_arbitration_trace = _pipeline_evidence_arbitration_trace(
        evidence_arbitration_mode,
        evidence_model_path,
        status="disabled" if evidence_arbitration_mode == "off" else "not_applicable",
    )
    preflight_reason = _preflight_rejection(query, graph)
    if preflight_reason is not None and binder_mode != "s4prime":
        reject_routing = RoutingResult(
            operator="REJECT",
            confidence=1.0,
            intent="no_answer",
            route_type="preflight",
            fallback_operators=[],
            trace={"rule": "review_hardening_preflight", "reason": preflight_reason},
        )
        return PipelineResult(
            result=None,
            routing=reject_routing,
            plan=None,
            attempts=(
                PipelineAttempt(
                    operator="REJECT",
                    status="preflight_rejected",
                    route_type="preflight",
                    error=preflight_reason,
                ),
            ),
            fallback_used=False,
            original_operator="REJECT",
            selected_operator=None,
            status="failed",
            evidence_arbitration_trace=_pipeline_evidence_arbitration_trace(
                evidence_arbitration_mode,
                evidence_model_path,
                status="disabled" if evidence_arbitration_mode == "off" else "not_applicable",
                warnings=["pipeline_rejected_before_arbitration"],
            ),
        )
    routing = routing or router.route(query)
    routing = _routing_with_expression_route(query, routing)
    routing = _routing_with_strategy_selection(query, graph, routing)
    if preflight_reason is not None:
        routing = dataclasses.replace(
            routing,
            trace={
                **dict(routing.trace or {}),
                "preflight_hint": {
                    "blocking": False,
                    "reason": preflight_reason,
                    "delegated_to": "s4prime",
                },
            },
        )
    if force_operator is None:
        router_disagreement = router_second_opinion_abstain_trace(
            query=query,
            primary_operator=routing.operator,
            second_opinion=router_second_opinion,
            min_confidence=router_second_opinion_min_confidence,
        )
        if router_disagreement.get("triggered"):
            return PipelineResult(
                result=None,
                routing=routing,
                plan=None,
                attempts=(
                    PipelineAttempt(
                        operator=routing.operator,
                        status="router_disagreement",
                        route_type=routing.route_type,
                        error="router_disagreement",
                    ),
                ),
                fallback_used=False,
                original_operator=routing.operator,
                selected_operator=None,
                status="abstained",
                evidence_arbitration_trace=evidence_arbitration_trace,
                abstain_trace=dict(router_disagreement),
            )
    solver = solver or OperatorSolver(field_grounder=field_grounder)
    effective_s5_gates = (
        s5_gates if enable_s5_validation and s5_gates is not None
        else S5GateConfig() if enable_s5_validation
        else S5GateConfig.disabled()
    )
    original_operator = force_operator or routing.operator
    operators = _candidate_operators(
        routing,
        force_operator,
        enable_fallback=enable_fallback,
        preferred_operators=preferred_operators,
    )
    attempts: list[PipelineAttempt] = []

    for index, operator in enumerate(operators):
        effective_routing = _routing_for_attempt(routing, operator, index)
        try:
            plan = _solve_plan(
                query,
                graph,
                effective_routing,
                solver,
                operator,
                binder_mode=binder_mode,
                s4prime_binder=s4prime_binder,
            )
        except S4PrimeInvalidProposalError as exc:
            attempts.append(
                PipelineAttempt(
                    operator=operator,
                    status="invalid_proposal",
                    route_type=effective_routing.route_type,
                    validation=dict(exc.validation or {}),
                    error=str(exc),
                )
            )
            continue
        except S4PrimeAbstainedError as exc:
            attempts.append(
                PipelineAttempt(
                    operator=operator,
                    status="s4prime_abstained",
                    route_type=effective_routing.route_type,
                    validation=dict(exc.validation or {}),
                    error=str(exc),
                )
            )
            continue
        except Exception as exc:
            attempts.append(
                PipelineAttempt(
                    operator=operator,
                    status="solve_error",
                    route_type=effective_routing.route_type,
                    error=str(exc),
                )
            )
            continue

        try:
            plan, evidence_arbitration_trace = apply_runtime_evidence_arbitration(
                query=query,
                graph=graph,
                plan=plan,
                config=EvidenceArbitrationConfig(
                    mode=evidence_arbitration_mode,
                    model_path=evidence_model_path,
                ),
                field_grounder=solver.field_grounder,
            )
        except Exception as exc:
            evidence_arbitration_trace = _pipeline_evidence_arbitration_trace(
                evidence_arbitration_mode,
                evidence_model_path,
                status="unavailable",
                warnings=[f"runtime_arbitration_failed:{exc}"],
            )

        validation = (
            validate_preconditions(graph, plan)
            if effective_s5_gates.preconditions or effective_s5_gates.unit or audit_sink is not None
            else None
        )
        validation_predicates = (
            dict(getattr(validation, "predicates", {}) or {})
            if validation is not None
            else {}
        )
        predicate_ok_for_audit = (
            _s5_predicates_passed(
                validation_predicates,
                disabled_s5_predicates=disabled_s5_predicates,
            )
            if validation is not None
            else True
        )
        enabled_precondition_violations = (
            _enabled_precondition_violations(
                list(
                    _active_precondition_violations(
                        validation,
                        disabled_s5_predicates=disabled_s5_predicates,
                    )
                ),
                effective_s5_gates,
            )
            if validation is not None
            else []
        )
        validation_payload = (
            {
                **validation.to_dict(),
                "status": "ok" if not enabled_precondition_violations else validation.status,
                "blocking_violations": enabled_precondition_violations,
            }
            if validation is not None
            else {"status": "disabled_for_ablation", "blocking_violations": []}
        )
        if audit_sink is not None and validation is not None:
            unchecked = None if predicate_ok_for_audit else _execute_for_s5_audit(graph, plan)
            _emit_s5_predicate_events(
                audit_sink,
                attempt_index=index,
                operator=operator,
                predicates=validation_predicates,
                unchecked_status=unchecked["status"] if unchecked else "not_run",
                unchecked_answer=unchecked["answer"] if unchecked else None,
                selected_token_ids=tuple(unchecked["selected_token_ids"]) if unchecked else (),
            )
        if enabled_precondition_violations:
            attempts.append(
                PipelineAttempt(
                    operator=operator,
                    status=validation.status,
                    route_type=effective_routing.route_type,
                    validation=validation_payload,
                    error=";".join(enabled_precondition_violations),
                )
            )
            continue

        try:
            result = execute_composite(graph, plan) if isinstance(plan, CompositeOperatorPlan) else execute(graph, plan)
            if audit_sink is not None:
                _emit_s5_predicate_events(
                    audit_sink,
                    attempt_index=index,
                    operator=operator,
                    predicates={
                        name: bool(ok)
                        for name, ok in (result.checks or {}).items()
                        if name in _BLOCKING_CHECKS
                    },
                    unchecked_status="ok",
                    unchecked_answer=result.answer,
                    selected_token_ids=tuple(token.token_id for token in result.selected_tokens),
                )
            failed_checks = [
                name
                for name in _enabled_blocking_checks(
                    dict(result.checks or {}),
                    effective_s5_gates,
                )
                if name not in disabled_s5_predicates
            ]
            if failed_checks:
                attempts.append(
                    PipelineAttempt(
                        operator=operator,
                        status="verification_failed",
                        route_type=effective_routing.route_type,
                        validation=validation_payload,
                        error=";".join(failed_checks),
                    )
                )
                continue
            success_attempts = attempts + [
                PipelineAttempt(
                    operator=operator,
                    status="ok",
                    route_type=effective_routing.route_type,
                    validation=validation_payload,
                    answer=result.answer,
                )
            ]
            implausible_trace = (
                _implausible_result_trace(
                    query,
                    operator,
                    result,
                    plan,
                    disabled_s5_predicates=disabled_s5_predicates,
                )
                if effective_s5_gates.plausibility
                else _empty_abstain_trace()
            )
            if audit_sink is not None:
                _emit_s5_predicate_events(
                    audit_sink,
                    attempt_index=index,
                    operator=operator,
                    predicates=_implausible_predicates(query, operator, result, plan),
                    unchecked_status="ok",
                    unchecked_answer=result.answer,
                    selected_token_ids=tuple(token.token_id for token in result.selected_tokens),
                )
            if implausible_trace.get("triggered"):
                implausible_attempts = success_attempts + [
                    PipelineAttempt(
                        operator=operator,
                        status="implausible_result",
                        route_type=effective_routing.route_type,
                        validation=validation_payload,
                        error=f"implausible_result:{implausible_trace.get('detail')}",
                        answer=result.answer,
                    )
                ]
                result = _attach_pipeline_metadata(
                    result,
                    implausible_attempts,
                    query=query,
                    graph=graph,
                    plan=plan,
                    original_operator=original_operator,
                    selected_operator=operator,
                    fallback_used=index > 0,
                    routing=routing,
                    metric_matcher=metric_matcher,
                    faithfulness_checker=faithfulness_checker,
                )
                result = _with_pipeline_abstain_metadata(result, implausible_trace)
                return PipelineResult(
                    result=result,
                    routing=routing,
                    plan=plan,
                    attempts=tuple(implausible_attempts),
                    fallback_used=index > 0,
                    original_operator=original_operator,
                    selected_operator=operator,
                    status="abstained",
                    evidence_arbitration_trace=evidence_arbitration_trace,
                    abstain_trace=implausible_trace,
                )
            result = _attach_pipeline_metadata(
                result,
                success_attempts,
                query=query,
                graph=graph,
                plan=plan,
                original_operator=original_operator,
                selected_operator=operator,
                fallback_used=index > 0,
                routing=routing,
                metric_matcher=metric_matcher,
                faithfulness_checker=faithfulness_checker,
            )
            abstain_trace = (
                _empty_abstain_trace()
                if _plan_is_s4prime(plan)
                else _abstain_trace_from_subgraph_discovery(
                    (result.metadata or {}).get("subgraph_discovery"),
                )
            )
            return PipelineResult(
                result=result,
                routing=routing,
                plan=plan,
                attempts=tuple(success_attempts),
                fallback_used=index > 0,
                original_operator=original_operator,
                selected_operator=operator,
                status="abstained" if abstain_trace.get("triggered") else "ok",
                evidence_arbitration_trace=evidence_arbitration_trace,
                abstain_trace=abstain_trace,
            )
        except Exception as exc:
            attempts.append(
                PipelineAttempt(
                    operator=operator,
                    status="execution_error",
                    route_type=effective_routing.route_type,
                    validation=validation_payload,
                    error=str(exc),
                )
            )

    abstain_trace = _abstain_trace_from_attempts(attempts)
    return PipelineResult(
        result=None,
        routing=routing,
        plan=None,
        attempts=tuple(attempts),
        fallback_used=False,
        original_operator=original_operator,
        selected_operator=None,
        status="abstained" if abstain_trace.get("triggered") else "failed",
        evidence_arbitration_trace=evidence_arbitration_trace,
        abstain_trace=abstain_trace,
    )


def _s5_predicates_passed(
    predicates: dict[str, bool],
    *,
    disabled_s5_predicates: frozenset[str],
) -> bool:
    return all(
        passed
        for predicate, passed in predicates.items()
        if predicate not in disabled_s5_predicates
    )


def _active_precondition_violations(
    validation: Any,
    *,
    disabled_s5_predicates: frozenset[str],
) -> tuple[str, ...]:
    predicate_to_violation = {
        "required_slots": "missing_required_slots",
        "slot_values": "invalid_slots",
        "field_exists": "field_not_found",
        "records_found": "records_not_found",
        "sufficient_factors": "insufficient_factors",
        "unit_compatible": "unit_incompatible",
        "dependencies_available": "dependency_ref_unavailable",
    }
    disabled_violations = {
        violation
        for predicate, violation in predicate_to_violation.items()
        if predicate in disabled_s5_predicates
    }
    return tuple(
        violation
        for violation in validation.blocking_violations
        if not any(disabled in violation for disabled in disabled_violations)
    )


def _execute_for_s5_audit(
    graph: AttributeValueGraph,
    plan: OperatorPlan | CompositeOperatorPlan,
) -> dict[str, Any]:
    try:
        result = execute_composite(graph, plan) if isinstance(plan, CompositeOperatorPlan) else execute(graph, plan)
    except Exception:
        return {"status": "execution_error", "answer": None, "selected_token_ids": ()}
    return {
        "status": "ok",
        "answer": result.answer,
        "selected_token_ids": tuple(token.token_id for token in result.selected_tokens),
    }


def _emit_s5_predicate_events(
    audit_sink: Callable[[S5PredicateEvent], None],
    *,
    attempt_index: int,
    operator: str,
    predicates: dict[str, bool],
    unchecked_status: str,
    unchecked_answer: float | str | None,
    selected_token_ids: tuple[str, ...],
) -> None:
    for predicate, passed in predicates.items():
        audit_sink(
            S5PredicateEvent(
                attempt_index=attempt_index,
                operator=operator,
                predicate=predicate,
                evaluated=True,
                passed=bool(passed),
                unchecked_status=unchecked_status,
                unchecked_answer=unchecked_answer,
                selected_token_ids=selected_token_ids,
            )
        )


def _implausible_result_trace(
    query: str,
    operator: str,
    result: ExecutionResult,
    plan: OperatorPlan | CompositeOperatorPlan,
    *,
    disabled_s5_predicates: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    predicates = _implausible_predicates(query, operator, result, plan)
    active_failed = {
        predicate
        for predicate, passed in predicates.items()
        if not passed and predicate not in disabled_s5_predicates
    }
    answer = _finite_float(getattr(result, "answer", None))
    if answer is None:
        return _empty_abstain_trace()
    op = str(operator or "").upper()
    if "share_self_binding" in active_failed:
        return _implausible_trace("share_self_binding")
    if "share_expected_range" in active_failed:
        return _implausible_trace(f"{op.lower()}_outside_expected_range:{answer}")
    if "ratio_answer_equals_numerator" in active_failed:
        return _implausible_trace("ratio_answer_equals_numerator")
    if "ratio_requires_two_values" in active_failed:
        return _implausible_trace("ratio_question_answered_by_single_value")
    return _empty_abstain_trace()


def _implausible_predicates(
    query: str,
    operator: str,
    result: ExecutionResult,
    plan: OperatorPlan | CompositeOperatorPlan,
) -> dict[str, bool]:
    del plan
    answer = _finite_float(getattr(result, "answer", None))
    if answer is None:
        return {
            "share_self_binding": True,
            "share_expected_range": True,
            "ratio_answer_equals_numerator": True,
            "ratio_requires_two_values": True,
        }
    op = str(operator or "").upper()
    ratio_family = {"RATIO", "SHARE", "MARGIN"}
    share_range_ok = True
    if op in {"SHARE", "MARGIN"} and not _query_asks_per_x(query):
        share_range_ok = 0.0 < answer <= 1.5
    ratio_requires_two_values_ok = True
    if _query_asks_ratio_or_share(query) and op not in ratio_family:
        ratio_requires_two_values_ok = not _answer_equals_single_selected_value(answer, result)
    return {
        "share_self_binding": not (op == "SHARE" and _share_has_self_binding(result)),
        "share_expected_range": share_range_ok,
        "ratio_answer_equals_numerator": not (
            op in ratio_family and _answer_equals_numerator_value(answer, result)
        ),
        "ratio_requires_two_values": ratio_requires_two_values_ok,
    }


def _share_has_self_binding(result: ExecutionResult) -> bool:
    metadata = dict(result.metadata or {})
    numerator_ids = {str(token_id) for token_id in metadata.get("numerator_token_ids") or []}
    denominator_ids = {str(token_id) for token_id in metadata.get("denominator_token_ids") or []}
    if not numerator_ids or not denominator_ids:
        return False
    if numerator_ids & denominator_ids:
        return True
    tokens_by_id = {token.token_id: token for token in result.selected_tokens or ()}
    numerator_spans = {
        span
        for token_id in numerator_ids
        if (span := _token_source_span_key(tokens_by_id.get(token_id))) is not None
    }
    denominator_spans = {
        span
        for token_id in denominator_ids
        if (span := _token_source_span_key(tokens_by_id.get(token_id))) is not None
    }
    return bool(numerator_spans & denominator_spans)


def _token_source_span_key(token: Any) -> tuple[Any, ...] | None:
    if token is None:
        return None
    source = getattr(token, "source", None)
    if source is None:
        return None
    char_start = getattr(source, "char_start", None)
    char_end = getattr(source, "char_end", None)
    if char_start is not None and char_end is not None:
        return (
            getattr(source, "document_id", None),
            getattr(source, "page", None),
            getattr(source, "table", None),
            getattr(source, "row", None),
            getattr(source, "column", None),
            char_start,
            char_end,
        )
    row = getattr(source, "row", None)
    column = getattr(source, "column", None)
    text_excerpt = getattr(source, "text_excerpt", None)
    if row is None and column is None and not text_excerpt:
        return None
    return (
        getattr(source, "document_id", None),
        getattr(source, "page", None),
        getattr(source, "table", None),
        row,
        column,
        text_excerpt,
    )


def _implausible_trace(detail: str) -> dict[str, Any]:
    return {
        "triggered": True,
        "reason": "implausible_result",
        "detail": detail,
    }


def _query_asks_ratio_or_share(query: str) -> bool:
    normalized = normalize_identifier(query)
    terms = {term for term in normalized.split("_") if term}
    return bool(
        any(
            term in normalized
            for term in (
                "what_percentage",
                "what_percent",
                "percentage_of",
                "percent_of",
                "portion",
                "share_of",
                "of_total",
                "as_a_percent",
                "as_a_percentage",
            )
        )
        or (("percentage" in normalized or "percent" in normalized) and "total" in normalized)
        or bool({"ratio", "portion"} & terms)
    )


def _query_asks_per_x(query: str) -> bool:
    normalized = normalize_identifier(query)
    return any(
        term in normalized
        for term in (
            "per_share",
            "per_common_share",
            "per_unit",
            "per_employee",
            "per_customer",
        )
    )


def _plan_is_s4prime(plan: OperatorPlan | CompositeOperatorPlan) -> bool:
    if isinstance(plan, CompositeOperatorPlan):
        if str((plan.trace or {}).get("solver") or "").startswith("s4prime"):
            return True
        return any(_plan_is_s4prime(step) for step in plan.steps)
    return str((plan.trace or {}).get("solver") or "").startswith("s4prime")


def _answer_equals_numerator_value(answer: float, result: ExecutionResult) -> bool:
    metadata = dict(result.metadata or {})
    numerator_ids = {str(token_id) for token_id in metadata.get("numerator_token_ids") or []}
    if not numerator_ids:
        return False
    rows = metadata.get("unit_normalization")
    for index, token in enumerate(result.selected_tokens or ()):
        if token.token_id not in numerator_ids:
            continue
        if _close_float(answer, float(token.value)):
            return True
        if isinstance(rows, list) and index < len(rows) and isinstance(rows[index], dict):
            normalized = _finite_float(rows[index].get("normalized_value"))
            if normalized is not None and _close_float(answer, normalized):
                return True
    return False


def _answer_equals_single_selected_value(answer: float, result: ExecutionResult) -> bool:
    tokens = tuple(result.selected_tokens or ())
    if len(tokens) != 1:
        return False
    token = tokens[0]
    if _close_float(answer, float(token.value)):
        return True
    rows = (result.metadata or {}).get("unit_normalization")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        normalized = _finite_float(rows[0].get("normalized_value"))
        if normalized is not None and _close_float(answer, normalized):
            return True
    return False


def _with_pipeline_abstain_metadata(
    result: ExecutionResult,
    abstain_trace: dict[str, Any],
) -> ExecutionResult:
    metadata = dict(result.metadata or {})
    pipeline = dict(metadata.get("pipeline") or {})
    pipeline["status"] = "abstained"
    metadata["pipeline"] = pipeline
    discovery = dict(metadata.get("subgraph_discovery") or {})
    discovery["abstain"] = dict(abstain_trace)
    metadata["subgraph_discovery"] = discovery
    return dataclasses.replace(result, metadata=metadata)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    if number in (float("inf"), float("-inf")):
        return None
    return number


def _close_float(left: float, right: float, *, abs_tol: float = 1e-9) -> bool:
    return abs(left - right) <= abs_tol


def execute_with_fallback(
    query: str,
    graph: AttributeValueGraph,
    *,
    router: Any | None = None,
    routing: RoutingResult | None = None,
    solver: OperatorSolver | None = None,
    force_operator: str | None = None,
) -> ExecutionResult:
    pipeline = run_operator_pipeline(
        query,
        graph,
        router=router,
        routing=routing,
        solver=solver,
        force_operator=force_operator,
        enable_fallback=True,
    )
    if pipeline.result is None:
        errors = [attempt.to_dict() for attempt in pipeline.attempts]
        raise ValueError(f"All operator attempts failed: {errors}")
    return pipeline.result


def _candidate_operators(
    routing: RoutingResult,
    force_operator: str | None,
    *,
    enable_fallback: bool,
    preferred_operators: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if force_operator:
        return [force_operator]
    if preferred_operators:
        first_preferred = next((operator for operator in preferred_operators if operator), None)
        return [first_preferred] if first_preferred else [routing.operator]
    return [routing.operator] if routing.operator else []


def _pipeline_evidence_arbitration_trace(
    mode: str,
    model_path: Path,
    *,
    status: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": mode != "off",
        "mode": mode,
        "policy": "safe_v5",
        "status": status,
        "model_path": str(model_path),
        "decisions": [],
        "warnings": list(warnings or []),
    }


def _routing_for_attempt(routing: RoutingResult, operator: str, index: int) -> RoutingResult:
    if operator == routing.operator and index == 0:
        return routing
    return dataclasses.replace(
        routing,
        operator=operator,
        confidence=min(routing.confidence, 0.75),
        route_type="fallback" if index > 0 else routing.route_type,
        trace={
            **(routing.trace or {}),
            "fallback_attempt_index": index,
            "original_operator": routing.operator,
        },
    )


def _routing_with_strategy_selection(
    query: str,
    graph: AttributeValueGraph,
    routing: RoutingResult,
) -> RoutingResult:
    trace = dict(routing.trace or {})
    if "strategy_selector" in trace:
        return routing
    analysis = analyze_hybrid_query(query, graph)
    trace["strategy_selector"] = analysis.optimizer.to_dict()
    trace["cost_optimizer"] = analysis.optimizer.to_dict()
    trace["vector_candidates"] = [candidate.to_dict() for candidate in analysis.vector_candidates]
    trace["scalar_pruning"] = analysis.scalar_trace.to_dict()
    return dataclasses.replace(routing, trace=trace)


def _routing_with_expression_route(query: str, routing: RoutingResult) -> RoutingResult:
    expression = parse_expression_plan(query)
    if expression is None or expression.route_operator == routing.operator:
        return routing
    if expression.confidence < 0.8:
        return routing
    fallback_operators = [
        routing.operator,
        *routing.fallback_operators,
    ]
    deduped_fallbacks: list[str] = []
    for operator in fallback_operators:
        if operator and operator != expression.route_operator and operator not in deduped_fallbacks:
            deduped_fallbacks.append(operator)
    trace = {
        **(routing.trace or {}),
        "expression_route": {
            "original_operator": routing.operator,
            "expression_operator": expression.route_operator,
            "kind": expression.kind,
            "confidence": expression.confidence,
        },
    }
    return dataclasses.replace(
        routing,
        operator=expression.route_operator,
        confidence=max(routing.confidence, expression.confidence),
        intent="expression",
        route_type="expression_override",
        fallback_operators=deduped_fallbacks,
        trace=trace,
    )


def _solve_plan(
    query: str,
    graph: AttributeValueGraph,
    routing: RoutingResult,
    solver: OperatorSolver,
    operator: str,
    *,
    binder_mode: str = "baseline",
    s4prime_binder: Any | None = None,
) -> OperatorPlan | CompositeOperatorPlan:
    if binder_mode == "s4prime":
        binder = s4prime_binder or S4PrimeBinder()
        binding = binder.bind(query, graph, routing)
        if not binding.validation.ok:
            raise S4PrimeInvalidProposalError(
                "invalid_proposal",
                validation=binding.validation.to_dict(),
            )
        if binding.plan is None:
            raise S4PrimeInvalidProposalError(
                "invalid_proposal:no_plan",
                validation=binding.validation.to_dict(),
            )
        return binding.plan
    if binder_mode != "baseline":
        raise ValueError(f"Unsupported binder_mode: {binder_mode}")
    if operator == "COMPOSITE":
        plan = solve_composite_operator_plan(
            query,
            graph,
            routing,
            field_grounder=solver.field_grounder,
            unit_resolver=solver.unit_resolver,
        )
        return _attach_expression_plan_trace(
            query,
            operator,
            plan,
            graph=graph,
            field_grounder=solver.field_grounder,
        )
    plan = solver.solve(query, graph, routing, operator=operator)
    return _attach_expression_plan_trace(
        query,
        operator,
        plan,
        graph=graph,
        field_grounder=solver.field_grounder,
    )


def _attach_expression_plan_trace(
    query: str,
    operator: str,
    plan: OperatorPlan | CompositeOperatorPlan,
    *,
    graph: AttributeValueGraph | None = None,
    field_grounder: FieldGrounder | None = None,
) -> OperatorPlan | CompositeOperatorPlan:
    expression = parse_expression_plan(query)
    if expression is None or expression.route_operator != operator:
        return plan
    if (
        isinstance(plan, OperatorPlan)
        and graph is not None
        and field_grounder is not None
    ):
        plan = _attach_expression_leaf_grounding(plan, expression, graph, field_grounder)
    trace = {
        **(plan.trace or {}),
        "expression_plan": expression.to_dict(),
    }
    return dataclasses.replace(plan, trace=trace)


def _attach_expression_leaf_grounding(
    plan: OperatorPlan,
    expression: Any,
    graph: AttributeValueGraph,
    field_grounder: FieldGrounder,
) -> OperatorPlan:
    if plan.operator != "SHARE" or expression.kind != "part_to_whole":
        return plan
    if _plan_has_structured_share_selection(plan):
        return plan
    root = expression.root
    if root.op != "divide" or len(root.children) != 2:
        return plan
    numerator_leaf, denominator_leaf = root.children
    numerator_query = getattr(numerator_leaf, "evidence_query", None)
    denominator_query = getattr(denominator_leaf, "evidence_query", None)
    if numerator_query is None or denominator_query is None:
        return plan

    numerator = _ground_expression_evidence_query(
        numerator_query,
        graph,
        field_grounder,
        prefer_whole=False,
    )
    denominator = _ground_expression_evidence_query(
        denominator_query,
        graph,
        field_grounder,
        prefer_whole=True,
    )
    if numerator is None or denominator is None:
        return plan
    numerator_tokens, numerator_trace = numerator
    denominator_tokens, denominator_trace = denominator
    if not numerator_tokens or not denominator_tokens:
        return plan

    slots = dict(plan.slots or {})
    slots["numerator_field"] = Slot(
        surface=numerator_query.field_surface,
        grounded_value=numerator_tokens[0].field_name,
        confidence=0.94,
    )
    slots["denominator_field"] = Slot(
        surface=denominator_query.field_surface,
        grounded_value=denominator_tokens[0].field_name,
        confidence=0.94,
    )
    slots["numerator_token_ids"] = Slot(
        surface=numerator_query.field_surface,
        grounded_value=[token.token_id for token in numerator_tokens],
        confidence=numerator_trace["confidence"],
    )
    slots["denominator_token_ids"] = Slot(
        surface=denominator_query.field_surface,
        grounded_value=[token.token_id for token in denominator_tokens],
        confidence=denominator_trace["confidence"],
    )
    for global_filter in (
        "dimension_filters",
        "numerator_dimension_filters",
        "denominator_dimension_filters",
        "entity",
        "entities",
    ):
        slots.pop(global_filter, None)

    trace = {
        **(plan.trace or {}),
        "expression_grounding": {
            "strategy": "expression_leaf_token_ranker_v1",
            "numerator": numerator_trace,
            "denominator": denominator_trace,
        },
    }
    return dataclasses.replace(plan, slots=slots, trace=trace)


def _plan_has_structured_share_selection(plan: OperatorPlan) -> bool:
    slots = plan.slots or {}
    return any(
        key in slots
        for key in (
            "numerator_token_ids",
            "denominator_token_ids",
            "numerator_dimension_filters",
            "denominator_dimension_filters",
            "denominator_row_selector",
        )
    )


def _ground_expression_evidence_query(
    evidence_query: Any,
    graph: AttributeValueGraph,
    field_grounder: FieldGrounder,
    *,
    prefer_whole: bool,
) -> tuple[tuple[AttributeValueToken, ...], dict[str, Any]] | None:
    if not graph.tokens:
        return None
    candidate_field = _ground_expression_field(
        evidence_query.field_surface,
        graph,
        field_grounder,
    )
    scored = [
        (
            _expression_token_score(
                token,
                evidence_query,
                candidate_field,
                prefer_whole=prefer_whole,
            ),
            token,
        )
        for token in graph.tokens
    ]
    scored = [(score, token) for score, token in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -_token_source_row_index(item[1])), reverse=True)
    best_score, best_token = scored[0]
    if best_score < 5.0:
        return None
    return (
        (best_token,),
        {
            "field_surface": evidence_query.field_surface,
            "time_surface": evidence_query.time_surface,
            "role": evidence_query.role,
            "grounded_field": candidate_field,
            "selected_token_ids": [best_token.token_id],
            "selected_values": [best_token.value],
            "confidence": min(0.99, max(0.5, best_score / 12.0)),
            "top_candidates": [
                {
                    "token_id": token.token_id,
                    "field_name": token.field_name,
                    "year": token.year,
                    "value": token.value,
                    "score": round(score, 4),
                }
                for score, token in scored[:3]
            ],
        },
    )


def _ground_expression_field(
    field_surface: str,
    graph: AttributeValueGraph,
    field_grounder: FieldGrounder,
) -> str | None:
    if not graph.fields:
        return None
    try:
        return field_grounder.ground(field_surface, graph.fields).field_name
    except Exception:
        return graph.fields[0] if graph.fields else None


def _expression_token_score(
    token: AttributeValueToken,
    evidence_query: Any,
    grounded_field: str | None,
    *,
    prefer_whole: bool,
) -> float:
    score = 0.0
    if evidence_query.time_surface:
        try:
            expected_year = int(evidence_query.time_surface)
        except (TypeError, ValueError):
            expected_year = None
        if expected_year is not None:
            score += 4.0 if token_matches_year(token, expected_year) else -5.0
    if grounded_field and token.field_name == grounded_field:
        score += 3.0
    elif grounded_field:
        score -= 1.0

    leaf_terms = _meaningful_expression_terms(evidence_query.field_surface)
    token_terms = _token_grounding_terms(token)
    overlap = leaf_terms & token_terms
    score += 2.2 * len(overlap)
    if leaf_terms and leaf_terms.issubset(token_terms):
        score += 1.5
    local_terms = _token_value_window_terms(token)
    local_overlap = leaf_terms & local_terms
    score += 0.9 * len(local_overlap)
    if leaf_terms and leaf_terms.issubset(local_terms):
        score += 1.0
    anchor_terms = _expression_anchor_terms(evidence_query.field_surface)
    if anchor_terms:
        token_anchor_overlap = anchor_terms & (token_terms | local_terms)
        if token_anchor_overlap:
            score += 6.0 * len(token_anchor_overlap)
        else:
            score -= 12.0
    if _field_word_overlap(evidence_query.field_surface, token):
        score += 0.7

    condition_terms = _expression_condition_terms(evidence_query, grounded_field)
    if condition_terms:
        label_terms = _token_label_terms(token)
        label_overlap = condition_terms & label_terms
        local_condition_overlap = condition_terms & local_terms
        if prefer_whole:
            if label_overlap:
                score -= 1.8 + 0.5 * len(label_overlap)
        else:
            score += 1.4 * len(label_overlap)
            score += 1.1 * len(local_condition_overlap)
            if condition_terms.issubset(label_terms | local_terms):
                score += 1.5
    elif prefer_whole and _token_label_has_text_condition(token):
        score -= 2.0

    dimensions = token.dimensions or {}
    has_segment = bool(dimensions.get("segment"))
    if prefer_whole:
        score += 1.4 if not has_segment else -1.0
        if _token_is_total_like(token):
            score += 1.2
    else:
        score += 1.8 if has_segment else 0.0
    return score


def _meaningful_expression_terms(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "from",
        "in",
        "is",
        "net",
        "of",
        "sales",
        "sale",
        "revenue",
        "revenues",
        "the",
        "total",
        "was",
        "were",
    }
    return {
        term
        for term in normalize_identifier(value).split("_")
        if term and not term.isdigit() and term not in stopwords
    }


def _expression_anchor_terms(value: str) -> set[str]:
    anchors = {
        "asset",
        "assets",
        "cash",
        "cost",
        "costs",
        "debt",
        "equity",
        "expense",
        "expenses",
        "fee",
        "fees",
        "gain",
        "gains",
        "income",
        "liabilities",
        "liability",
        "loss",
        "losses",
        "proceed",
        "proceeds",
        "profit",
        "profits",
        "revenue",
        "revenues",
        "share",
        "shares",
        "tax",
    }
    return {
        term
        for term in normalize_identifier(value).split("_")
        if term in anchors
    }


def _token_grounding_terms(token: AttributeValueToken) -> set[str]:
    values: list[object] = [
        token.entity_id,
        token.company_name,
        token.field_name,
        token.field_label,
        token.raw_label,
    ]
    if token.source is not None:
        values.extend([token.source.column, token.source.text_excerpt])
    values.extend((token.dimensions or {}).values())
    terms: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set, frozenset)):
            nested = " ".join(str(item) for item in value)
        elif isinstance(value, dict):
            nested = " ".join(str(item) for item in value.values())
        else:
            nested = "" if value is None else str(value)
        terms.update(term for term in normalize_identifier(nested).split("_") if term)
    return terms


def _token_label_terms(token: AttributeValueToken) -> set[str]:
    values: list[object] = [
        token.field_label,
        token.raw_label,
        token.source.column if token.source is not None else None,
    ]
    values.extend((token.dimensions or {}).values())
    terms: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set, frozenset)):
            nested = " ".join(str(item) for item in value)
        elif isinstance(value, dict):
            nested = " ".join(str(item) for item in value.values())
        else:
            nested = "" if value is None else str(value)
        terms.update(term for term in normalize_identifier(nested).split("_") if term)
    return terms


def _token_value_window_terms(token: AttributeValueToken, *, radius: int = 64) -> set[str]:
    if token.source is None or not token.source.text_excerpt:
        return set()
    text = token.source.text_excerpt
    match = _find_value_surface(text, token.value)
    if match is None:
        return set()
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return {
        term
        for term in normalize_identifier(text[start:end]).split("_")
        if term
    }


def _find_value_surface(text: str, value: float) -> re.Match[str] | None:
    for surface in _value_surface_forms(value):
        match = re.search(
            rf"(?<![\d.]){re.escape(surface)}(?![\d.])",
            text,
            re.IGNORECASE,
        )
        if match is not None:
            return match
    return None


def _value_surface_forms(value: float) -> tuple[str, ...]:
    forms: list[str] = []
    if float(value).is_integer():
        integer = int(value)
        forms.extend([f"{integer:,}", str(integer)])
    forms.append(f"{value:g}")
    forms.append(str(value))
    return tuple(dict.fromkeys(forms))


def _expression_condition_terms(evidence_query: Any, grounded_field: str | None) -> set[str]:
    leaf_terms = _meaningful_expression_terms(evidence_query.field_surface)
    if not grounded_field:
        return leaf_terms
    field_terms = _meaningful_expression_terms(grounded_field)
    for alias in field_aliases(grounded_field):
        field_terms.update(_meaningful_expression_terms(alias))
    return leaf_terms - field_terms


def _token_label_has_text_condition(token: AttributeValueToken) -> bool:
    terms = _token_label_terms(token)
    return bool(
        terms
        & {
            "held",
            "located",
            "stored",
            "kept",
            "generated",
            "derived",
            "attributable",
            "outside",
            "inside",
            "jurisdictions",
        }
    )


def _field_word_overlap(field_surface: str, token: AttributeValueToken) -> bool:
    field_words = {"sale", "sales", "revenue", "revenues", "net_sales"}
    surface_terms = set(normalize_identifier(field_surface).split("_"))
    if not surface_terms & field_words:
        return False
    token_terms = _token_grounding_terms(token)
    return bool(token_terms & field_words)


def _token_is_total_like(token: AttributeValueToken) -> bool:
    text = " ".join(
        str(value)
        for value in (
            token.raw_label,
            token.field_label,
            (token.dimensions or {}).get("row_label"),
            token.company_name,
        )
        if value is not None
    )
    normalized = normalize_identifier(text)
    return (
        "total" in normalized
        or "aggregate" in normalized
        or not (token.dimensions or {}).get("segment")
    )


def _token_source_row_index(token: AttributeValueToken) -> int:
    if token.source is None or token.source.row is None:
        return 10**6
    return int(token.source.row)


def _attach_pipeline_metadata(
    result: ExecutionResult,
    attempts: list[PipelineAttempt],
    *,
    query: str,
    graph: AttributeValueGraph,
    plan: OperatorPlan | CompositeOperatorPlan,
    original_operator: str,
    selected_operator: str,
    fallback_used: bool,
    routing: RoutingResult,
    metric_matcher: MetricMatcher | None = None,
    faithfulness_checker: TokenFaithfulnessChecker | None = None,
) -> ExecutionResult:
    metadata = dict(result.metadata or {})
    metadata["pipeline"] = {
        "status": "ok",
        "fallback_used": fallback_used,
        "original_operator": original_operator,
        "selected_operator": selected_operator,
        "routing": routing.to_dict(),
        "attempts": [attempt.to_dict() for attempt in attempts],
    }
    metadata["subgraph_discovery"] = _subgraph_discovery_trace(
        query=query,
        graph=graph,
        plan=plan,
        result=result,
        selected_operator=selected_operator,
        routing=routing,
        metric_matcher=metric_matcher,
        faithfulness_checker=faithfulness_checker,
    )
    return dataclasses.replace(result, metadata=metadata)


def _subgraph_discovery_trace(
    *,
    query: str,
    graph: AttributeValueGraph,
    plan: OperatorPlan | CompositeOperatorPlan,
    result: ExecutionResult,
    selected_operator: str,
    routing: RoutingResult,
    metric_matcher: MetricMatcher | None = None,
    faithfulness_checker: TokenFaithfulnessChecker | None = None,
) -> dict[str, Any]:
    selected_tokens = tuple(result.selected_tokens or ())
    slot_bindings = _subgraph_slot_bindings(plan, selected_tokens)
    bindings_by_token = {binding["token_id"]: binding for binding in slot_bindings}
    selected_token_ids = [token.token_id for token in selected_tokens]
    selected_token_id_set = set(selected_token_ids)
    direction_constraint = _direction_constraint_from_query(query, selected_operator)
    consistency_checks = _subgraph_consistency_checks(
        selected_operator,
        selected_tokens,
        plan=plan,
        direction_constraint=direction_constraint,
    )
    preliminary_slot_candidates: list[dict[str, Any]] | None = None
    faithfulness_tokens: tuple[AttributeValueToken, ...] = ()
    if faithfulness_checker is not None:
        preliminary_slot_candidates = _subgraph_slot_candidate_rankings(
            plan,
            graph,
            selected_tokens,
            top_k=3,
            metric_matcher=metric_matcher,
        )
        faithfulness_tokens = _subgraph_faithfulness_check_tokens(
            graph,
            selected_tokens,
            preliminary_slot_candidates,
        )
    evidence_faithfulness = evaluate_token_faithfulness(
        faithfulness_tokens,
        faithfulness_checker,
    )
    slot_candidates = _subgraph_slot_candidate_rankings(
        plan,
        graph,
        selected_tokens,
        top_k=3,
        metric_matcher=metric_matcher,
        faithfulness_by_token={
            str(row["token_id"]): row
            for row in evidence_faithfulness.get("tokens", [])
        },
    )
    direction_abstain = _abstain_trace_from_direction_consistency(consistency_checks, direction_constraint)
    slot_abstain = _abstain_trace_from_slot_candidates(selected_operator, slot_candidates)
    abstain_trace = direction_abstain if direction_abstain.get("triggered") else slot_abstain
    return {
        "graph_type": "implicit_attribute_value_graph",
        "question": query,
        "question_constraints": _question_constraints_trace(
            plan,
            selected_operator,
            routing,
            direction_constraint=direction_constraint,
        ),
        "candidate_token_count": len(graph.tokens),
        "candidate_tokens": [
            _subgraph_token_row(
                token,
                selected=token.token_id in selected_token_id_set,
                role=str(bindings_by_token.get(token.token_id, {}).get("role") or ""),
            )
            for token in graph.tokens
        ],
        "slot_bindings": slot_bindings,
        "slot_candidates": slot_candidates,
        "calculation_subgraph": {
            "operator": selected_operator,
            "selected_token_ids": selected_token_ids,
            "facts": [
                _subgraph_token_row(
                    token,
                    selected=True,
                    role=str(bindings_by_token.get(token.token_id, {}).get("role") or ""),
                )
                for token in selected_tokens
            ],
            "calculation": result.calculation,
            "answer": result.answer,
            "evidence": [
                {
                    "token_id": token.token_id,
                    "text_excerpt": token.source.text_excerpt if token.source is not None else None,
                }
                for token in selected_tokens
            ],
        },
        "evidence_faithfulness": evidence_faithfulness,
        "consistency_checks": consistency_checks,
        "abstain": abstain_trace,
    }


def _question_constraints_trace(
    plan: OperatorPlan | CompositeOperatorPlan,
    selected_operator: str,
    routing: RoutingResult,
    *,
    direction_constraint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = plan.to_dict()
    if isinstance(plan, CompositeOperatorPlan):
        return {
            "operator": selected_operator,
            "routing_operator": routing.operator,
            "plan_type": "composite",
            "steps": payload.get("steps", []),
        }
    trace = {
        "operator": selected_operator,
        "routing_operator": routing.operator,
        "plan_type": "single",
        "slots": payload.get("slots", {}),
        "filters": payload.get("filters", []),
        "trace": payload.get("trace"),
    }
    if direction_constraint is not None:
        trace["direction"] = dict(direction_constraint)
    return trace


def _subgraph_slot_bindings(
    plan: OperatorPlan | CompositeOperatorPlan,
    selected_tokens: tuple[AttributeValueToken, ...],
) -> list[dict[str, Any]]:
    if isinstance(plan, CompositeOperatorPlan):
        rows: list[dict[str, Any]] = []
        for index, step in enumerate(plan.steps):
            rows.extend(
                {
                    **row,
                    "step_id": f"s{index + 1}",
                }
                for row in _subgraph_slot_bindings(step, selected_tokens)
            )
        return rows

    values = _plan_slot_values(plan)
    rows: list[dict[str, Any]] = []
    for token in selected_tokens:
        slot, role, reason = _slot_role_for_token(plan.operator, values, token)
        rows.append(
            {
                "slot": slot,
                "role": role,
                "token_id": token.token_id,
                "match_reason": reason,
            }
        )
    return rows


def _subgraph_slot_candidate_rankings(
    plan: OperatorPlan | CompositeOperatorPlan,
    graph: AttributeValueGraph,
    selected_tokens: tuple[AttributeValueToken, ...],
    *,
    top_k: int,
    metric_matcher: MetricMatcher | None = None,
    faithfulness_by_token: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(plan, CompositeOperatorPlan):
        rows: list[dict[str, Any]] = []
        for index, step in enumerate(plan.steps):
            rows.extend(
                {
                    **row,
                    "step_id": f"s{index + 1}",
                }
                for row in _subgraph_slot_candidate_rankings(
                    step,
                    graph,
                    selected_tokens,
                    top_k=top_k,
                    metric_matcher=metric_matcher,
                    faithfulness_by_token=faithfulness_by_token,
                )
            )
        return rows

    selected_ids = {token.token_id for token in selected_tokens}
    values = _plan_slot_values(plan)
    rows: list[dict[str, Any]] = []
    for spec in _candidate_slot_specs(plan.operator, values):
        candidates = [
            _candidate_score_row(
                token,
                spec,
                selected=token.token_id in selected_ids,
                metric_matcher=metric_matcher,
                faithfulness_trace=(faithfulness_by_token or {}).get(token.token_id),
            )
            for token in graph.tokens
        ]
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["token_id"])))
        rows.append(
            {
                "slot": spec["slot"],
                "role": spec["role"],
                "constraints": dict(spec.get("constraints") or {}),
                "candidates": candidates[:top_k],
            }
        )
    return rows


def _subgraph_faithfulness_check_tokens(
    graph: AttributeValueGraph,
    selected_tokens: tuple[AttributeValueToken, ...],
    slot_candidates: list[dict[str, Any]],
) -> tuple[AttributeValueToken, ...]:
    token_ids = {token.token_id for token in selected_tokens}
    for slot_row in slot_candidates:
        for candidate in slot_row.get("candidates", []):
            token_id = candidate.get("token_id")
            if token_id is not None:
                token_ids.add(str(token_id))
    return tuple(token for token in graph.tokens if token.token_id in token_ids)


def _candidate_slot_specs(operator: str, values: dict[str, Any]) -> list[dict[str, Any]]:
    operator = operator.upper()
    target_field = values.get("target_field")
    if operator in {"DIFFERENCE", "COMPARE"}:
        specs: list[dict[str, Any]] = []
        if values.get("left_time") is not None:
            specs.append(
                _slot_spec(
                    "left_time",
                    "被减数",
                    field=target_field,
                    year=_optional_int_for_trace(values.get("left_time")),
                )
            )
        if values.get("right_time") is not None:
            specs.append(
                _slot_spec(
                    "right_time",
                    "减数",
                    field=target_field,
                    year=_optional_int_for_trace(values.get("right_time")),
                )
            )
        if values.get("left_field") is not None:
            specs.append(_slot_spec("left_field", "被减数", field=values.get("left_field")))
        if values.get("right_field") is not None:
            specs.append(_slot_spec("right_field", "减数", field=values.get("right_field")))
        return specs or [_slot_spec("target_field", "参与差值计算", field=target_field)]
    if operator in {"GROWTH", "PERCENT_CHANGE"}:
        return [
            _slot_spec(
                "from_time",
                "基期",
                field=target_field,
                year=_optional_int_for_trace(values.get("from_time")),
            ),
            _slot_spec(
                "to_time",
                "目标期",
                field=target_field,
                year=_optional_int_for_trace(values.get("to_time")),
            ),
        ]
    if operator in {"RATIO", "SHARE", "MARGIN"}:
        return [
            _slot_spec("numerator_field", "分子", field=values.get("numerator_field")),
            _slot_spec("denominator_field", "分母", field=values.get("denominator_field")),
        ]
    if target_field is not None:
        return [_slot_spec("target_field", "目标数值", field=target_field)]
    return []


def _slot_spec(
    slot: str,
    role: str,
    *,
    field: object | None = None,
    year: int | None = None,
    entity: object | None = None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if field is not None:
        constraints["field_name"] = str(field)
    if year is not None:
        constraints["year"] = year
    if entity is not None:
        constraints["entity"] = str(entity)
    return {
        "slot": slot,
        "role": role,
        "constraints": constraints,
    }


def _candidate_score_row(
    token: AttributeValueToken,
    spec: dict[str, Any],
    *,
    selected: bool,
    metric_matcher: MetricMatcher | None = None,
    faithfulness_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    constraints = dict(spec.get("constraints") or {})
    score = 0.0
    reasons: list[str] = []
    match_trace: dict[str, dict[str, Any]] = {}
    expected_field = constraints.get("field_name")
    if expected_field is not None:
        matcher = metric_matcher or DEFAULT_METRIC_MATCHER
        metric_trace = matcher.match(token, str(expected_field))
        match_trace["metric"] = metric_trace
        if metric_trace["score"] > 0:
            score += 0.5 * float(metric_trace["score"])
            reasons.append(f"指标匹配 {token.field_name}")
        else:
            reasons.append(f"指标不匹配 {token.field_name}")
    expected_year = constraints.get("year")
    if expected_year is not None:
        time_trace = _time_match_trace(token, int(expected_year))
        match_trace["time"] = time_trace
        if time_trace["score"] > 0:
            score += 0.4 * float(time_trace["score"])
            reasons.append(f"年份匹配 {token.year}")
        elif token.year is not None:
            reasons.append(f"年份不匹配 {token.year}")
        else:
            reasons.append("年份缺失")
    expected_entity = constraints.get("entity")
    if expected_entity is not None:
        if _matches_entity(token, str(expected_entity)):
            score += 0.2
            reasons.append(f"实体匹配 {expected_entity}")
        else:
            reasons.append(f"实体不匹配 {token.company_name}")
    if selected:
        score += 0.1
        reasons.append("执行器选中")
    if faithfulness_trace:
        match_trace["faithfulness"] = dict(faithfulness_trace)
        penalty = float(faithfulness_trace.get("score_penalty") or 0.0)
        if penalty > 0:
            score = max(0.0, score - penalty)
            reasons.append("证据蕴含低置信")
    return {
        "token_id": token.token_id,
        "score": round(score, 4),
        "selected": selected,
        "field_name": token.field_name,
        "year": token.year,
        "value": token.value,
        "unit": token.unit,
        "match_reasons": reasons,
        "match_trace": match_trace,
    }


def _metric_match_trace(token: AttributeValueToken, expected_field: str) -> dict[str, Any]:
    return DEFAULT_METRIC_MATCHER.match(token, expected_field)


def _time_match_trace(token: AttributeValueToken, expected_year: int) -> dict[str, Any]:
    if token.year == expected_year:
        return {
            "level": "rule_exact",
            "score": 1.0,
            "rationale": f"year exact match: {expected_year}",
        }
    return {
        "level": "rule_exact",
        "score": 0.0,
        "rationale": f"year mismatch: expected {expected_year}, got {token.year}",
    }


def _plan_slot_values(plan: OperatorPlan) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in (plan.slots or {}).items():
        if isinstance(value, Slot):
            values[key] = value.grounded_value
        else:
            values[key] = value
    return values


def _slot_role_for_token(
    operator: str,
    values: dict[str, Any],
    token: AttributeValueToken,
) -> tuple[str, str, str]:
    operator = operator.upper()
    if operator in {"DIFFERENCE", "COMPARE"}:
        if "left_time" in values and token.year == _optional_int_for_trace(values.get("left_time")):
            return "left_time", "被减数", f"年份匹配 {token.year}"
        if "right_time" in values and token.year == _optional_int_for_trace(values.get("right_time")):
            return "right_time", "减数", f"年份匹配 {token.year}"
        if "left_field" in values and token.field_name == str(values.get("left_field")):
            return "left_field", "被减数", f"指标匹配 {token.field_name}"
        if "right_field" in values and token.field_name == str(values.get("right_field")):
            return "right_field", "减数", f"指标匹配 {token.field_name}"
        return "selected_token_ids", "参与差值计算", "执行器选中"
    if operator in {"GROWTH", "PERCENT_CHANGE"}:
        if "from_time" in values and token.year == _optional_int_for_trace(values.get("from_time")):
            return "from_time", "基期", f"年份匹配 {token.year}"
        if "to_time" in values and token.year == _optional_int_for_trace(values.get("to_time")):
            return "to_time", "目标期", f"年份匹配 {token.year}"
        return "selected_token_ids", "参与增长率计算", "执行器选中"
    if operator in {"RATIO", "SHARE", "MARGIN"}:
        if "numerator_field" in values and token.field_name == str(values.get("numerator_field")):
            return "numerator_field", "分子", f"指标匹配 {token.field_name}"
        if "denominator_field" in values and token.field_name == str(values.get("denominator_field")):
            return "denominator_field", "分母", f"指标匹配 {token.field_name}"
        return "selected_token_ids", "参与比例计算", "执行器选中"
    if "target_field" in values and token.field_name == str(values.get("target_field")):
        return "target_field", "目标数值", f"指标匹配 {token.field_name}"
    return "selected_token_ids", "执行器选中数值", "执行器选中"


def _optional_int_for_trace(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _subgraph_consistency_checks(
    operator: str,
    selected_tokens: tuple[AttributeValueToken, ...],
    *,
    plan: OperatorPlan | CompositeOperatorPlan,
    direction_constraint: dict[str, Any] | None,
) -> dict[str, bool]:
    fields = {token.field_name for token in selected_tokens}
    units = {token.unit for token in selected_tokens if token.unit}
    checks = {
        "tokens_selected": bool(selected_tokens),
        "units_present": all(bool(token.unit) for token in selected_tokens),
        "evidence_present": all(token.source is not None for token in selected_tokens),
    }
    if operator.upper() in {"DIFFERENCE", "COMPARE", "GROWTH", "PERCENT_CHANGE"}:
        checks["single_metric_for_difference"] = len(fields) <= 1
    if direction_constraint is not None and isinstance(plan, OperatorPlan):
        checks["direction_matches_question"] = _plan_matches_direction_constraint(
            plan,
            operator,
            direction_constraint,
        )
    if len(selected_tokens) >= 2:
        checks["units_uniform"] = len(units) <= 1
    return checks


def _plan_matches_direction_constraint(
    plan: OperatorPlan,
    operator: str,
    direction_constraint: dict[str, Any],
) -> bool:
    values = _plan_slot_values(plan)
    canonical = OPERATOR_REGISTRY.canonical_executor_operator(operator)
    if canonical in {"DIFFERENCE", "COMPARE"}:
        expected_left = direction_constraint.get("expected_left_time")
        expected_right = direction_constraint.get("expected_right_time")
        actual_left = _optional_int_for_trace(values.get("left_time"))
        actual_right = _optional_int_for_trace(values.get("right_time"))
        if actual_left is None or actual_right is None:
            return True
        return actual_left == expected_left and actual_right == expected_right
    if canonical == "GROWTH":
        expected_from = direction_constraint.get("expected_from_time")
        expected_to = direction_constraint.get("expected_to_time")
        actual_from = _optional_int_for_trace(values.get("from_time"))
        actual_to = _optional_int_for_trace(values.get("to_time"))
        if actual_from is None or actual_to is None:
            return True
        return actual_from == expected_from and actual_to == expected_to
    return True


def _subgraph_token_row(
    token: AttributeValueToken,
    *,
    selected: bool,
    role: str,
) -> dict[str, Any]:
    source = token.source
    return {
        "token_id": token.token_id,
        "selected": selected,
        "role": role or None,
        "entity_id": token.entity_id,
        "company_name": token.company_name,
        "field_name": token.field_name,
        "field_label": token.field_label,
        "raw_label": token.raw_label,
        "canonical_concept_id": token.canonical_concept_id,
        "year": token.year,
        "value": token.value,
        "unit": token.unit,
        "dimensions": dict(token.dimensions or {}),
        "source": (
            {
                "document_id": source.document_id,
                "page": source.page,
                "table": source.table,
                "row": source.row,
                "column": source.column,
                "char_start": source.char_start,
                "char_end": source.char_end,
                "text_excerpt": source.text_excerpt,
            }
            if source is not None
            else None
        ),
    }


def _preflight_rejection(query: str, graph: AttributeValueGraph) -> str | None:
    lowered = query.lower()
    normalized = normalize_identifier(query)
    if re.search(r"\bif\b.*\bdoubled\b|without\s+using\s+table\s+evidence", lowered):
        return "unsupported_counterfactual_or_evidence_constraint"
    if "nonexistent" in lowered:
        return "nonexistent_entity_requested"

    dimension_reason = _unknown_dimension_reason(query, graph)
    if dimension_reason is not None:
        return dimension_reason

    industry_reason = _unknown_industry_reason(query, graph)
    if industry_reason is not None:
        return industry_reason

    field_reason = _unknown_field_reason(normalized, graph)
    if field_reason is not None:
        return field_reason

    entity_reason = _entity_ambiguity_or_missing_reason(query, graph)
    if entity_reason is not None:
        return entity_reason

    unit_reason = _unit_mismatch_reason(query, graph)
    if unit_reason is not None:
        return unit_reason

    return None


def _unknown_dimension_reason(query: str, graph: AttributeValueGraph) -> str | None:
    values = _dimension_values(graph)
    if not values or "report" not in query.lower():
        return None
    normalized_query = normalize_identifier(query).replace("_", "")
    if "bothreports" in normalized_query or "acrossbothreports" in normalized_query:
        return None
    known = {
        normalize_identifier(value).replace("_", "")
        for bucket in values.values()
        for value in bucket
    }
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]*)\s+reports?\b", query, re.IGNORECASE):
        candidate = normalize_identifier(match.group(1)).replace("_", "")
        if candidate not in known and candidate not in {"both", "which", "what"}:
            return f"dimension_not_found:{match.group(1)}"
    return None


def _unknown_industry_reason(query: str, graph: AttributeValueGraph) -> str | None:
    known = {_compact_identifier(industry) for industry in graph.industries}
    if not known:
        return None
    normalized_query = normalize_identifier(query).replace("_", "")
    if any(industry and industry in normalized_query for industry in known):
        return None

    for candidate in _explicit_industry_candidates(query):
        if _compact_identifier(candidate) not in known:
            return f"industry_not_found:{candidate}"
    return None


def _unknown_field_reason(normalized_query: str, graph: AttributeValueGraph) -> str | None:
    fields = set(graph.fields)
    if _query_mentions_open_schema_field(normalized_query, fields):
        return None
    covered_aliases = _covered_field_aliases(fields)
    candidates: list[tuple[int, str, str]] = []
    for field, aliases in FIELD_ALIASES.items():
        if field in fields or field in {"company_name", "industry", "year", "document_id"}:
            continue
        aliases = (field, *field_aliases(field))
        for alias in aliases:
            alias_normalized = normalize_identifier(alias)
            semantic_field = resolve_review_hardening_field(alias_normalized.replace("_", " "), graph.fields)
            if (
                not alias_normalized
                or alias_normalized in covered_aliases
                or semantic_field in fields
                or _alias_is_covered_by_open_schema_field(alias_normalized, fields)
                or _alias_is_too_broad_for_unknown_field_rejection(field, alias_normalized)
            ):
                continue
            candidates.append((len(alias_normalized.split("_")), field, alias_normalized))
    for _, field, alias_normalized in sorted(candidates, reverse=True):
        if _normalized_phrase_in_query(alias_normalized, normalized_query):
            return f"field_not_found:{field}"
    return None


def _query_mentions_open_schema_field(normalized_query: str, fields: set[str]) -> bool:
    query_parts = normalized_query.split("_")
    for field in fields:
        normalized_field = normalize_identifier(field)
        if normalized_field and _normalized_phrase_in_query(normalized_field, normalized_query):
            return True
        field_parts = [
            part
            for part in normalized_field.split("_")
            if part and part not in _FIELD_MATCH_STOPWORDS
        ]
        optional_tail_removed = [
            part for part in field_parts if part not in _FIELD_OPTIONAL_SUFFIXES
        ]
        if len(optional_tail_removed) >= 2 and _parts_appear_in_order(optional_tail_removed, query_parts):
            return True
        if len(field_parts) == 1:
            if _single_open_schema_field_part_matches(field_parts[0], normalized_field, query_parts):
                return True
            continue
        width = min(len(field_parts), 3)
        if any(
            _normalized_phrase_in_query("_".join(field_parts[index:index + width]), normalized_query)
            for index in range(len(field_parts) - width + 1)
        ):
            return True
        if _parts_appear_in_order(field_parts, query_parts):
            return True
    return False


def _single_open_schema_field_part_matches(
    part: str,
    normalized_field: str,
    query_parts: list[str],
) -> bool:
    if part not in query_parts:
        return False
    if part in _FIELD_BROAD_SINGLE_PARTS:
        prefixes = normalized_field.split("_")[:-1]
        return bool(prefixes and any(prefix in _FIELD_TOTAL_PREFIXES for prefix in prefixes))
    return True


def _parts_appear_in_order(parts: list[str], query_parts: list[str]) -> bool:
    position = 0
    for part in parts:
        try:
            position = query_parts.index(part, position) + 1
        except ValueError:
            return False
    return True


def _entity_ambiguity_or_missing_reason(query: str, graph: AttributeValueGraph) -> str | None:
    if parse_expression_plan(query) is not None:
        return None
    names = sorted({token.company_name for token in graph.tokens})
    lowered = query.lower()

    industry = _query_industry(query, graph)
    candidate_names = [
        name
        for name in names
        if industry is None or any(
            token.company_name == name and token.industry == industry
            for token in graph.tokens
        )
    ]
    exact_hits = [name for name in candidate_names if name.lower() in lowered]
    if exact_hits:
        return None

    query_terms = {
        term
        for term in re.findall(r"[A-Za-z0-9]+", lowered)
        if len(term) >= 3 and term not in _PREFLIGHT_ENTITY_STOPWORDS
    }
    for term in query_terms:
        matches = [
            name
            for name in candidate_names
            if term in {part.lower() for part in re.findall(r"[A-Za-z0-9]+", name)}
        ]
        if len(matches) >= 2 and _query_looks_like_single_entity_lookup(query):
            return f"ambiguous_entity:{term}"
    if _query_looks_like_single_entity_lookup(query):
        known_entity_terms = {
            term
            for name in candidate_names
            for term in re.findall(r"[A-Za-z0-9]+", name.lower())
        }
        for original, term in _ordered_query_terms(query):
            if (
                term in query_terms
                and term not in known_entity_terms
                and not _term_overlaps_known_field_or_industry(term, graph)
            ):
                return f"entity_not_found:{original}"
    return None


def _unit_mismatch_reason(query: str, graph: AttributeValueGraph) -> str | None:
    requested = _requested_unit_text(query)
    if requested is None:
        return None
    field = _field_hint(query, graph)
    if field is None:
        return None
    resolver = UnitResolver()
    requested_info = resolver.detect(requested)
    field_units = [
        resolver.detect(str(token.unit))
        for token in graph.select(field_name=field)
        if token.unit
    ]
    if field in {"revenue", "net_profit", "operating_income"} and requested_info.unit_category in {"count", "percentage"}:
        return f"unit_incompatible:{field}:{requested}"
    if field == "employees" and requested_info.unit_category in {"money", "percentage"}:
        return f"unit_incompatible:{field}:{requested}"
    if "margin" in normalize_identifier(query) and requested_info.unit_category == "money":
        return f"unit_incompatible:margin:{requested}"
    if field_units and all(info.unit_category != requested_info.unit_category for info in field_units):
        return f"unit_incompatible:{field}:{requested}"
    return None


def _dimension_values(graph: AttributeValueGraph) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for token in graph.tokens:
        for key, value in (token.dimensions or {}).items():
            text = str(value)
            if not text:
                continue
            bucket = values.setdefault(str(key), [])
            if text not in bucket:
                bucket.append(text)
    return values


def _query_industry(query: str, graph: AttributeValueGraph) -> str | None:
    normalized_query = normalize_identifier(query).replace("_", "")
    for industry in graph.industries:
        if normalize_identifier(industry).replace("_", "") in normalized_query:
            return industry
    return None


def _compact_identifier(value: object) -> str:
    return normalize_identifier(str(value)).replace("_", "")


def _normalized_phrase_in_query(alias_normalized: str, query_normalized: str) -> bool:
    alias_parts = alias_normalized.split("_")
    query_parts = query_normalized.split("_")
    if not alias_parts:
        return False
    if len(alias_parts) == 1:
        return alias_parts[0] in query_parts
    width = len(alias_parts)
    return any(query_parts[index:index + width] == alias_parts for index in range(len(query_parts) - width + 1))


def _explicit_industry_candidates(query: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9&_-]*)\s+(?:industry|sector|segment)\b",
        query,
        re.IGNORECASE,
    ):
        candidate = match.group(1)
        if candidate.lower() not in {"which", "what", "the", "this", "that"}:
            candidates.append(candidate.upper() if candidate.isupper() else candidate)
    if _query_looks_like_single_entity_lookup(query):
        return candidates
    field_terms = _known_field_terms()
    for match in re.finditer(r"\b[A-Z][A-Z0-9&_-]{2,}\b", query):
        candidate = match.group(0)
        term = normalize_identifier(candidate)
        if (
            term
            and term not in field_terms
            and term.lower() not in {"which", "what", "the", "this", "that"}
            and term not in {"usd", "eur", "cny", "rmb", "fy", "eps", "gdp"}
        ):
            candidates.append(candidate)
    return list(dict.fromkeys(candidates))


def _known_field_terms() -> set[str]:
    terms: set[str] = set()
    for field, aliases in FIELD_ALIASES.items():
        if field in {"company_name", "industry", "year", "document_id"}:
            continue
        for alias in (field, *aliases):
            terms.update(normalize_identifier(alias).split("_"))
    return {term for term in terms if term}


def _covered_field_aliases(fields: set[str]) -> set[str]:
    aliases: set[str] = set()
    for field in fields:
        for alias in (field, *field_aliases(field)):
            alias_normalized = normalize_identifier(alias)
            if alias_normalized:
                aliases.add(alias_normalized)
    return aliases


def _alias_is_covered_by_open_schema_field(alias_normalized: str, fields: set[str]) -> bool:
    alias_parts = [
        part
        for part in alias_normalized.split("_")
        if part and part not in _FIELD_MATCH_STOPWORDS
    ]
    if not alias_parts:
        return False
    for field in fields:
        field_parts = [
            part
            for part in normalize_identifier(field).split("_")
            if part and part not in _FIELD_MATCH_STOPWORDS
        ]
        if not field_parts:
            continue
        if set(alias_parts).issubset(set(field_parts)):
            return True
    return False


def _alias_is_too_broad_for_unknown_field_rejection(
    field: str,
    alias_normalized: str,
) -> bool:
    return (
        field in {"total_assets", "total_liabilities", "employees", "store_count"}
        and alias_normalized
        in {
            "asset",
            "assets",
            "liability",
            "liabilities",
            "employee",
            "employees",
            "store",
            "stores",
        }
    )


def _ordered_query_terms(query: str) -> list[tuple[str, str]]:
    return [
        (match.group(0), match.group(0).lower())
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9]*", query)
    ]


def _term_overlaps_known_field_or_industry(term: str, graph: AttributeValueGraph) -> bool:
    if term in _known_field_terms():
        return True
    return any(term in normalize_identifier(industry).split("_") for industry in graph.industries)


def _query_looks_like_single_entity_lookup(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(r"\b(?:revenue|sales|net earnings|net profit|top-line|profit margin)\b", lowered)
        and not re.search(
            r"\b(?:total|sum|aggregate|count|how many|which|identify|name|highest|largest|strongest|lowest)\b",
            lowered,
        )
    )


def _requested_unit_text(query: str) -> str | None:
    lowered = query.lower()
    if re.search(r"\bin\s+employees\b|\bin\s+headcount\b|\bin\s+workforce\b", lowered):
        return "employees"
    if re.search(r"\bin\s+percent\b|\bin\s+percentage\b|\bin\s+%\b", lowered):
        return "percent"
    match = re.search(
        r"\b(?:in|above|over|greater\s+than|cleared)\s+(?:a\s+|one\s+|\d+(?:\.\d+)?\s+)?((?:million|billion|thousand)\s+)?(usd|eur|cny|rmb|dollars?)\b",
        lowered,
    )
    if match:
        scale = (match.group(1) or "").strip()
        currency = match.group(2).upper()
        if currency == "DOLLARS":
            currency = "USD"
        return f"{scale} {currency}".strip()
    return None


def _field_hint(query: str, graph: AttributeValueGraph) -> str | None:
    normalized = normalize_identifier(query)
    tokens = set(normalized.split("_"))
    fields = set(graph.fields)
    if (
        "top_line" in normalized
        or "topline" in tokens
        or "revenue" in tokens
        or "sales" in tokens
        or "turnover" in tokens
    ):
        return "revenue" if "revenue" in fields else None
    if (
        "bottom_line" in normalized
        or "bottomline" in tokens
        or "net_profit" in normalized
        or "net_earnings" in normalized
        or "profit_after_tax" in normalized
    ):
        return "net_profit" if "net_profit" in fields else None
    if "employees" in tokens or "workforce" in tokens or "headcount" in tokens:
        return "employees" if "employees" in fields else None
    if "margin" in normalized:
        return "net_profit" if "net_profit" in fields else None
    return None


_PREFLIGHT_ENTITY_STOPWORDS = {
    "and",
    "the",
    "for",
    "from",
    "with",
    "year",
    "fiscal",
    "report",
    "reports",
    "total",
    "sum",
    "aggregate",
    "count",
    "revenue",
    "sales",
    "profit",
    "margin",
    "tech",
    "finance",
    "retail",
    "energy",
    "company",
    "companies",
    "firm",
    "firms",
    "issuer",
    "issuers",
    "cloud",
}

_FIELD_MATCH_STOPWORDS = {
    "and",
    "the",
    "of",
    "for",
    "from",
    "to",
    "in",
    "at",
    "as",
    "a",
    "an",
    "total",
    "net",
}

_FIELD_OPTIONAL_SUFFIXES = {
    "amount",
    "amounts",
    "expense",
    "expenses",
    "value",
    "values",
    "million",
    "millions",
    "usd",
}

_FIELD_BROAD_SINGLE_PARTS = {
    "asset",
    "assets",
    "liability",
    "liabilities",
    "revenue",
    "sales",
    "employee",
    "employees",
    "store",
    "stores",
}

_FIELD_TOTAL_PREFIXES = {"total", "net", "gross"}


def _hybrid_strategy(routing: RoutingResult) -> str:
    trace = routing.trace or {}
    for trace_key in ("cost_optimizer", "strategy_selector"):
        payload = trace.get(trace_key)
        if isinstance(payload, dict) and payload.get("strategy"):
            return str(payload["strategy"])
    strategy = trace.get("hybrid_strategy")
    if strategy:
        return str(strategy)
    route_type = routing.route_type or ""
    if "embedding" in route_type or "hybrid" in route_type:
        return "hybrid"
    return "hybrid"


def _vector_grounding_trace(
    routing: RoutingResult,
    plan: OperatorPlan | CompositeOperatorPlan | None,
) -> dict[str, Any]:
    field_scores: dict[str, float] = {}
    if isinstance(plan, OperatorPlan):
        field_scores = dict(plan.field_scores)
    return {
        "route_type": routing.route_type,
        "operator": routing.operator,
        "confidence": routing.confidence,
        "intent": routing.intent,
        "fallback_operators": list(routing.fallback_operators),
        "routing_trace": dict(routing.trace or {}),
        "field_scores": field_scores,
    }


def _scalar_constraint_trace(
    plan: OperatorPlan | CompositeOperatorPlan | None,
) -> dict[str, Any]:
    if isinstance(plan, CompositeOperatorPlan):
        return {
            "steps": [_scalar_constraint_trace(step) for step in plan.steps],
        }
    if not isinstance(plan, OperatorPlan):
        return {"field_slots": {}, "filters": [], "other_slots": {}}

    field_slots: dict[str, Any] = {}
    other_slots: dict[str, Any] = {}
    for key, value in (plan.slots or {}).items():
        grounded = getattr(value, "grounded_value", value)
        if key.endswith("_field") or key in {"target_field", "condition_field"}:
            field_slots[key] = grounded
        else:
            other_slots[key] = grounded
    return {
        "field_slots": field_slots,
        "filters": [
            {
                "type": item.type,
                "grounded_key": item.grounded_key,
                "grounded_value": item.grounded_value,
                "confidence": item.confidence,
            }
            for item in (plan.filters or [])
        ],
        "other_slots": other_slots,
    }


def _verification_report_from_checks(
    checks: dict[str, bool],
    *,
    warnings: list[str],
    passed: bool,
) -> dict[str, Any]:
    from graph_numeric.operators.verifier import VerificationReport

    return VerificationReport(
        checks=checks,
        warnings=warnings,
        passed=passed,
    ).to_dict()


def _error_attribution_from_report(report: dict[str, Any], status: str) -> dict[str, bool]:
    categories = set(report.get("failure_categories", []))
    return {
        "extraction": status == "extraction_failed" or "extraction" in categories,
        "grounding": "grounding" in categories,
        "planning": (status != "ok") or "planning" in categories,
        "unit": "unit" in categories,
        "filter": "filter" in categories,
        "arithmetic": "arithmetic" in categories,
        "evidence": "evidence" in categories,
    }
