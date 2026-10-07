from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from graph_numeric.runtime.answer_override_registry import (
    ANSWER_OVERRIDE_CONDITION_REGISTRY,
    MAX_OVERRIDE_ERROR_RATE,
)


DISPATCHABLE_ABSTAIN_REASONS = frozenset(
    {
        "unsupported_composition",
        "missing_slot",
        "unit_incompatible",
        "implausible_result",
        "operator_precondition_failed",
        "answer_type_mismatch",
    }
)

NON_DISPATCHABLE_ABSTAIN_REASONS = frozenset(
    {
        "ambiguous_binding",
        "conflicting_evidence",
        "direction_violation",
    }
)


@dataclass(frozen=True)
class SelectionResult:
    selected_source: str
    selected_answer: Any
    selected_operator: str | None
    pipeline_decision: str
    abstain_reason: str | None
    fallback_gate: dict[str, Any] | None
    shadow_candidates: tuple[dict[str, Any], ...]
    recovered_from: str | None


def select_answer_candidate(
    *,
    pipeline: Any,
    verification: dict[str, Any],
    candidates: list[dict[str, Any]],
    fallback: dict[str, Any] | None,
    text_fact: dict[str, Any] | None,
    error_category: str | None,
    question: str | None = None,
    question_answer_type: str | None = None,
    ranker_intent: str | None = None,
) -> SelectionResult:
    if text_fact is not None:
        candidate = _candidate_by_source(candidates, "text_fact")
        return SelectionResult(
            selected_source="text_fact",
            selected_answer=_candidate_answer(candidate, text_fact),
            selected_operator=_candidate_operator(candidate, text_fact),
            pipeline_decision="text_fact",
            abstain_reason=None,
            fallback_gate=None,
            shadow_candidates=(),
            recovered_from=None,
        )

    fallback_candidate = _candidate_by_source(candidates, "financial_fallback")
    if pipeline_is_answered(pipeline, verification):
        candidate = _candidate_by_source(candidates, "pipeline")
        type_mismatch_reason = pipeline_answer_type_mismatch_reason(
            pipeline,
            candidate,
            question_answer_type=question_answer_type,
            ranker_intent=ranker_intent,
        )
        if type_mismatch_reason is not None:
            gate = fallback_dispatch_gate(
                type_mismatch_reason,
                fallback,
                question=question,
                question_answer_type=question_answer_type,
                ranker_intent=ranker_intent,
            )
            if fallback_candidate is not None and gate["allowed"]:
                return SelectionResult(
                    selected_source="financial_fallback",
                    selected_answer=_candidate_answer(fallback_candidate, fallback),
                    selected_operator=_candidate_operator(fallback_candidate, fallback),
                    pipeline_decision="abstain",
                    abstain_reason=type_mismatch_reason,
                    fallback_gate=gate,
                    shadow_candidates=(),
                    recovered_from=type_mismatch_reason,
                )
            shadow = (dict(fallback_candidate),) if fallback_candidate is not None else ()
            return SelectionResult(
                selected_source="abstain",
                selected_answer=None,
                selected_operator=_pipeline_operator(pipeline),
                pipeline_decision="abstain",
                abstain_reason=type_mismatch_reason,
                fallback_gate=gate,
                shadow_candidates=shadow,
                recovered_from=None,
            )
        override_gate = pipeline_answer_override_gate(
            pipeline,
            fallback,
            question=question,
            question_answer_type=question_answer_type,
            ranker_intent=ranker_intent,
        )
        if fallback_candidate is not None and override_gate["allowed"]:
            return SelectionResult(
                selected_source="financial_fallback",
                selected_answer=_candidate_answer(fallback_candidate, fallback),
                selected_operator=_candidate_operator(fallback_candidate, fallback),
                pipeline_decision="answered_overridden",
                abstain_reason=None,
                fallback_gate=override_gate,
                shadow_candidates=(),
                recovered_from="pipeline_answered",
            )
        shadow = (dict(fallback_candidate),) if fallback_candidate is not None else ()
        return SelectionResult(
            selected_source="pipeline",
            selected_answer=_candidate_answer(candidate, None, default=getattr(pipeline, "answer", None)),
            selected_operator=_candidate_operator(candidate, None, default=_pipeline_operator(pipeline)),
            pipeline_decision="answered",
            abstain_reason=None,
            fallback_gate=(
                override_gate
                if fallback_candidate is not None
                else None
            ),
            shadow_candidates=shadow,
            recovered_from=None,
        )

    abstain_reason = normalize_abstain_reason(pipeline, error_category)
    gate = fallback_dispatch_gate(
        abstain_reason,
        fallback,
        question=question,
        question_answer_type=question_answer_type,
        ranker_intent=ranker_intent,
    )
    if fallback_candidate is not None and gate["allowed"]:
        return SelectionResult(
            selected_source="financial_fallback",
            selected_answer=_candidate_answer(fallback_candidate, fallback),
            selected_operator=_candidate_operator(fallback_candidate, fallback),
            pipeline_decision="abstain",
            abstain_reason=abstain_reason,
            fallback_gate=gate,
            shadow_candidates=(),
            recovered_from=abstain_reason,
        )

    shadow = (dict(fallback_candidate),) if fallback_candidate is not None else ()
    return SelectionResult(
        selected_source="abstain",
        selected_answer=None,
        selected_operator=_pipeline_operator(pipeline),
        pipeline_decision="abstain",
        abstain_reason=abstain_reason,
        fallback_gate=gate,
        shadow_candidates=shadow,
        recovered_from=None,
    )


def pipeline_is_answered(pipeline: Any, verification: dict[str, Any]) -> bool:
    if getattr(pipeline, "status", None) != "ok":
        return False
    if not bool(verification.get("passed")):
        return False
    return getattr(pipeline, "answer", None) is not None


def pipeline_answer_type_mismatch_reason(
    pipeline: Any,
    candidate: dict[str, Any] | None,
    *,
    question_answer_type: str | None = None,
    ranker_intent: str | None = None,
) -> str | None:
    if not _boolean_intent(question_answer_type, ranker_intent):
        return None
    operator = str(_candidate_operator(candidate, None, default=_pipeline_operator(pipeline)) or "").upper()
    answer = _candidate_answer(candidate, None, default=getattr(pipeline, "answer", None))
    if operator == "BOOLEAN" and _answer_looks_boolean(answer):
        return None
    return "answer_type_mismatch"


def fallback_dispatch_gate(
    reason: str | None,
    fallback: dict[str, Any] | None,
    *,
    question: str | None = None,
    question_answer_type: str | None = None,
    ranker_intent: str | None = None,
) -> dict[str, Any]:
    if fallback is None:
        return {"allowed": False, "reason": "no_fallback", "abstain_reason": reason}
    if reason in NON_DISPATCHABLE_ABSTAIN_REASONS:
        return {"allowed": False, "reason": "non_dispatchable_abstain_reason", "abstain_reason": reason}
    if _boolean_intent(question_answer_type, ranker_intent) and str(fallback.get("operator") or "").upper() != "BOOLEAN":
        return {
            "allowed": False,
            "reason": "boolean_intent_blocks_numeric_fallback",
            "abstain_reason": reason,
        }
    if _indexed_return_family_fallback(fallback) and not _explicit_indexed_return_question(question):
        return {
            "allowed": False,
            "reason": "indexed_return_wording_required",
            "abstain_reason": reason,
            "strategy": str(fallback.get("strategy") or ""),
        }
    if _percent_change_question(question, ranker_intent) and _fallback_output_kind(fallback) == "absolute_diff":
        return {
            "allowed": False,
            "reason": "percent_change_blocks_absolute_diff_fallback",
            "abstain_reason": reason,
            "output_kind": "absolute_diff",
            "strategy": str(fallback.get("strategy") or ""),
        }
    confidence = float(fallback.get("confidence") or 0.0)
    if confidence < 0.85:
        return {
            "allowed": False,
            "reason": "fallback_confidence_below_threshold",
            "abstain_reason": reason,
            "confidence": confidence,
        }
    if reason in DISPATCHABLE_ABSTAIN_REASONS:
        return {
            "allowed": True,
            "reason": "dispatchable_abstain_reason",
            "abstain_reason": reason,
            "confidence": confidence,
        }
    return {"allowed": False, "reason": "unknown_abstain_reason", "abstain_reason": reason}


def pipeline_answer_override_gate(
    pipeline: Any,
    fallback: dict[str, Any] | None,
    *,
    question: str | None = None,
    question_answer_type: str | None = None,
    ranker_intent: str | None = None,
) -> dict[str, Any]:
    if fallback is None:
        return {"allowed": False, "reason": "no_fallback"}
    confidence = float(fallback.get("confidence") or 0.0)
    if confidence < 0.88:
        return {
            "allowed": False,
            "reason": "fallback_confidence_below_override_threshold",
            "confidence": confidence,
        }

    fallback_operator = str(fallback.get("operator") or "").upper()
    fallback_strategy = str(fallback.get("strategy") or "").strip()
    selected_operator = str(_pipeline_operator(pipeline) or "").upper()
    answer = getattr(pipeline, "answer", None)
    text = str(question or "").lower()
    normalized = _compact_question_text(text)

    if (
        fallback_operator == "AVG"
        and selected_operator != "AVG"
        and ("average" in normalized or "avg" in normalized)
    ):
        return _override_gate("average_fallback_overrides_non_average_pipeline", confidence)

    if (
        fallback_operator == "PERCENT_CHANGE"
        and fallback_strategy == "sentence_year_value_percent_change"
        and selected_operator == "PERCENT_CHANGE"
        and _percent_change_question(question, ranker_intent)
        and _numeric_answers_disagree(answer, fallback.get("answer"))
    ):
        return _override_gate("sentence_percent_change_overrides_pipeline_binding", confidence)

    if (
        fallback_strategy == "tax_position_activity_net_change"
        and fallback_operator == "SUM"
        and selected_operator == "DIFFERENCE"
        and abs(_coerce_float(answer) or 0.0) <= 1e-12
        and "tax_position" in normalized
        and ("net_change" in normalized or "change" in normalized)
    ):
        return _override_gate("tax_position_activity_overrides_zero_difference", confidence)

    return {"allowed": False, "reason": "pipeline_answered"}


def _override_gate(reason: str, confidence: float) -> dict[str, Any]:
    registry_gate = _override_condition_gate(reason)
    if not registry_gate["enabled"]:
        return {
            "allowed": False,
            "reason": "override_condition_disabled",
            "condition_id": reason,
            "confidence": confidence,
            "pipeline_answered": True,
            **registry_gate,
        }
    return {
        "allowed": True,
        "reason": reason,
        "condition_id": reason,
        "confidence": confidence,
        "pipeline_answered": True,
        "registry": registry_gate,
    }


def _override_condition_gate(condition_id: str) -> dict[str, Any]:
    condition = ANSWER_OVERRIDE_CONDITION_REGISTRY.get(condition_id)
    if condition is None:
        return {
            "condition_id": condition_id,
            "enabled": False,
            "hit_count": 0,
            "error_count": 0,
            "error_rate": 1.0,
            "max_error_rate": MAX_OVERRIDE_ERROR_RATE,
            "justification": "unregistered override condition",
            "absorb_debt": True,
            "absorb_path": "register this override before enabling it",
        }
    return condition.to_gate_payload()


def _compact_question_text(text: str) -> str:
    return "_".join(part for part in re.split(r"[^a-z0-9]+", text.lower()) if part)


def _mentions_month(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _numeric_answers_disagree(left: Any, right: Any) -> bool:
    left_number = _coerce_float(left)
    right_number = _coerce_float(right)
    if left_number is None or right_number is None:
        return False
    return abs(left_number - right_number) > max(1e-6, abs(right_number) * 0.05)


def _coerce_float(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").replace("%", "").split(";", 1)[0].strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent_change_question(question: str | None, ranker_intent: str | None) -> bool:
    if str(ranker_intent or "").upper() in {"PERCENT_CHANGE", "GROWTH"}:
        return True
    text = str(question or "").lower()
    if not text:
        return False
    return bool(
        "percent of the change" in text
        or "percentage of the change" in text
        or "percentage change" in text
        or "percent change" in text
        or "percentage net change" in text
        or "percent net change" in text
        or "growth rate" in text
        or ("rate of change" in text and ("percent" in text or "percentage" in text))
        or (
            ("percent" in text or "percentage" in text)
            and any(term in text for term in ("change", "increase", "decrease", "decline"))
        )
    )


def _fallback_output_kind(fallback: dict[str, Any] | None) -> str:
    if not isinstance(fallback, dict):
        return "unknown"
    kind = str(fallback.get("output_kind") or "").strip().lower()
    if kind:
        return kind
    strategy = str(fallback.get("strategy") or "").strip().lower()
    operator = str(fallback.get("operator") or "").strip().upper()
    absolute_diff_strategies = {
        "rate_point_change",
        "signed_same_row_year_change",
        "simple_year_difference",
        "net_change_balance",
        "markdown_rollforward_net_change",
        "beginning_end_balance_change",
        "wide_year_change",
        "period_begin_end_field_difference",
        "same_row_year_difference",
        "difference_expression",
        "raw_delta_for_percent_wording",
    }
    if strategy in absolute_diff_strategies or operator == "DIFFERENCE":
        return "absolute_diff"
    if operator in {"RATIO", "SHARE", "PERCENT_CHANGE", "GROWTH", "MARGIN"}:
        return "ratio"
    if operator in {"LOOKUP", "SUM", "AVG", "PRODUCT", "COUNT", "MAX", "MIN"}:
        return "value"
    return "unknown"


def _indexed_return_family_fallback(fallback: dict[str, Any] | None) -> bool:
    if not isinstance(fallback, dict):
        return False
    strategy = str(fallback.get("strategy") or "").strip()
    return strategy in {
        "cumulative_total_return",
        "indexed_stock_return_text",
        "indexed_return_outperform_text",
        "indexed_stock_outperformance_percent",
    }


def _explicit_indexed_return_question(question: str | None) -> bool:
    text = str(question or "").lower()
    if not text:
        return False
    normalized = _compact_question_text(text)
    if "cumulative" in normalized and "return" in normalized:
        return True
    if ("index" in normalized or "indexed" in normalized) and "return" in normalized:
        return True
    if any(term in normalized for term in ("s_p_500", "sp_500", "nasdaq", "dow_jones", "djia")):
        return True
    if "outperform" in normalized and any(term in normalized for term in ("peer_group", "benchmark", "market", "index")):
        return True
    return False


def _boolean_intent(question_answer_type: str | None, ranker_intent: str | None) -> bool:
    return str(question_answer_type or "").strip().lower() == "boolean" or str(ranker_intent or "").upper() == "BOOLEAN"


def _answer_looks_boolean(answer: Any) -> bool:
    if isinstance(answer, bool):
        return True
    return str(answer).strip().lower() in {"yes", "no", "true", "false"}


def normalize_abstain_reason(pipeline: Any, error_category: str | None) -> str:
    texts = [str(error_category or "")]
    abstain_trace = getattr(pipeline, "abstain_trace", None)
    if isinstance(abstain_trace, dict):
        reason = str(abstain_trace.get("reason") or "")
        if reason == "operator_precondition_failed":
            texts.append(reason)
            detail = abstain_trace.get("detail")
            if detail is not None:
                texts.append(str(detail))
    for attempt in getattr(pipeline, "attempts", ()) or ():
        if hasattr(attempt, "to_dict"):
            attempt_dict = attempt.to_dict()
        elif isinstance(attempt, dict):
            attempt_dict = attempt
        else:
            attempt_dict = {}
        for key in ("error", "status", "operator"):
            value = attempt_dict.get(key)
            if value is not None:
                texts.append(str(value))
    normalized = " ".join(texts).lower()
    if "direction_violation" in normalized or "direction violation" in normalized:
        return "direction_violation"
    if "conflicting_evidence" in normalized or "conflicting evidence" in normalized:
        return "conflicting_evidence"
    if "ambiguous_binding" in normalized or "multiple_candidates" in normalized or "ambiguous binding" in normalized:
        return "ambiguous_binding"
    if "unit_incompatible" in normalized or "unit mismatch" in normalized or "incompatible unit" in normalized:
        return "unit_incompatible"
    if "implausible_result" in normalized or "implausible result" in normalized:
        return "implausible_result"
    if "operator_precondition_failed" in normalized or "operator precondition failed" in normalized:
        return "operator_precondition_failed"
    if "answer_type_mismatch" in normalized or "answer type mismatch" in normalized:
        return "answer_type_mismatch"
    if "unsupported_composition" in normalized or "unsupported composition" in normalized:
        return "unsupported_composition"
    if (
        "records_not_found" in normalized
        or "field_not_found" in normalized
        or "entity_not_found" in normalized
        or "missing_required_slots" in normalized
        or "invalid_slots" in normalized
        or "precondition_violation" in normalized
        or "requires at least" in normalized
        or "from-value" in normalized
    ):
        return "missing_slot"
    return "missing_slot"


def _candidate_by_source(candidates: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    return next((candidate for candidate in candidates if candidate.get("source") == source), None)


def _candidate_answer(
    candidate: dict[str, Any] | None,
    fallback_like: dict[str, Any] | None,
    *,
    default: Any = None,
) -> Any:
    if candidate is not None and "answer" in candidate:
        return candidate.get("answer")
    if fallback_like is not None:
        return fallback_like.get("answer")
    return default


def _candidate_operator(
    candidate: dict[str, Any] | None,
    fallback_like: dict[str, Any] | None,
    *,
    default: str | None = None,
) -> str | None:
    if candidate is not None and candidate.get("operator") is not None:
        return str(candidate.get("operator"))
    if fallback_like is not None and fallback_like.get("operator") is not None:
        return str(fallback_like.get("operator"))
    return default


def _pipeline_operator(pipeline: Any) -> str | None:
    selected = getattr(pipeline, "selected_operator", None)
    if selected:
        return str(selected)
    routing = getattr(pipeline, "routing", None)
    routed = getattr(routing, "operator", None)
    return str(routed) if routed else None
