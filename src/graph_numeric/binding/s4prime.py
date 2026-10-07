from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping
from urllib import error, request

from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
    normalize_identifier,
)
from graph_numeric.extraction.llm_extraction import LLMGraphExtractorConfig
from graph_numeric.learning.router import RoutingResult
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, OperatorPlan, Slot
from graph_numeric.operators.operator_registry import OPERATOR_SPECS
from graph_numeric.operators.operator_solvers import resolve_question_constant_slots
from graph_numeric.runtime.llm_cache import chat_completion_with_cache, default_prompt_cache_path


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
STEP_REFERENCE_PATTERN = re.compile(r"^\$step(?P<step>\d+)(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


@dataclass(frozen=True)
class BindingRequest:
    question: str
    routed_operator: str
    operator_catalog: dict[str, dict[str, Any]]
    candidate_tokens: list[dict[str, Any]]
    question_constants: list[dict[str, Any]]
    prescreen: dict[str, Any]

    @property
    def token_ids(self) -> set[str]:
        return {str(row["id"]) for row in self.candidate_tokens}

    @property
    def question_constant_ids(self) -> set[str]:
        return {str(row["id"]) for row in self.question_constants}

    def token_by_id(self, token_id: str) -> AttributeValueToken | None:
        token = _TOKENS_BY_REQUEST_ID.get(id(self), {}).get(token_id)
        if token is not None:
            return token
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "routed_operator": self.routed_operator,
            "operator_catalog": self.operator_catalog,
            "candidate_tokens": self.candidate_tokens,
            "question_constants": self.question_constants,
            "prescreen": self.prescreen,
        }


@dataclass(frozen=True)
class BindingValidationReport:
    ok: bool
    error_codes: tuple[str, ...] = ()
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_codes": list(self.error_codes),
            "details": dict(self.details or {}),
        }


@dataclass(frozen=True)
class S4PrimeBindingResult:
    request: BindingRequest
    proposal: dict[str, Any]
    validation: BindingValidationReport
    plan: OperatorPlan | CompositeOperatorPlan | None
    validation_attempts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "proposal": dict(self.proposal),
            "validation": self.validation.to_dict(),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "validation_attempts": [dict(row) for row in self.validation_attempts],
        }


class S4PrimeInvalidProposalError(ValueError):
    def __init__(self, message: str, *, validation: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.validation = dict(validation or {})


class S4PrimeAbstainedError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        proposal: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.proposal = dict(proposal or {})
        self.validation = dict(validation or {})


class S4PrimeBinder:
    """Closed-set binding proposer interface.

    The default implementation is intentionally not a heuristic replacement for
    the baseline binder. It exists as the integration point for the LLM-backed
    proposer; tests and benchmark wiring can inject a concrete binder.
    """

    def bind(
        self,
        query: str,
        graph: AttributeValueGraph,
        routing: RoutingResult,
    ) -> S4PrimeBindingResult:
        raise RuntimeError("s4prime_binder_not_configured")


class S4PrimeLLMBinder(S4PrimeBinder):
    """LLM-backed closed-set binding proposer for S4prime."""

    def __init__(
        self,
        config: LLMGraphExtractorConfig | None = None,
        *,
        chat_completion: Any | None = None,
        model: str | None = None,
        max_candidate_tokens: int = 200,
        cache_path: str | None = None,
        cache_read_only: bool | None = None,
    ) -> None:
        self.config = config or LLMGraphExtractorConfig.from_env()
        self.chat_completion = chat_completion
        self.model = model or self.config.model
        self.max_candidate_tokens = max_candidate_tokens
        self.cache_path = cache_path
        self.cache_read_only = cache_read_only

    @property
    def available(self) -> bool:
        if self.chat_completion is not None:
            return bool(self.model)
        return self.config.available and bool(self.model)

    def bind(
        self,
        query: str,
        graph: AttributeValueGraph,
        routing: RoutingResult,
    ) -> S4PrimeBindingResult:
        if not self.available:
            raise RuntimeError("s4prime_llm_not_configured")
        binding_request = build_binding_request(
            question=query,
            graph=graph,
            routing=routing,
            max_candidate_tokens=self.max_candidate_tokens,
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(binding_request)},
            ],
        }
        proposal, validation = self._request_and_validate(payload, binding_request)
        attempts = [_validation_attempt(1, validation)]
        if not validation.ok:
            retry_payload = {
                **payload,
                "messages": [
                    payload["messages"][0],
                    {
                        "role": "user",
                        "content": (
                            f"{payload['messages'][1]['content']}\n\n"
                            + _validation_retry_prompt(
                                proposal,
                                validation,
                                binding_request,
                            )
                        ),
                    },
                ],
            }
            proposal, validation = self._request_and_validate(retry_payload, binding_request)
            retry_attempt = _validation_attempt(2, validation)
            retry_attempt[
                "validation_retry_recovered" if validation.ok else "validation_retry_failed"
            ] = True
            attempts.append(retry_attempt)
        if not validation.ok:
            failure = validation.to_dict()
            failure.update(
                {
                    "validation_first_failed": True,
                    "validation_retry_failed": True,
                    "attempts": attempts,
                }
            )
            raise S4PrimeInvalidProposalError("invalid_proposal", validation=failure)
        if proposal.get("status") == "abstain":
            raise S4PrimeAbstainedError(
                "s4prime_abstained",
                proposal=proposal,
                validation=validation.to_dict(),
            )
        plan = proposal_to_plan(proposal, binding_request)
        return S4PrimeBindingResult(
            request=binding_request,
            proposal=proposal,
            validation=validation,
            plan=plan,
            validation_attempts=tuple(attempts),
        )

    def _request_and_validate(
        self,
        payload: Mapping[str, Any],
        binding_request: BindingRequest,
    ) -> tuple[dict[str, Any], BindingValidationReport]:
        response = self._chat_completion(payload)
        proposal = parse_binding_proposal(_extract_content(response))
        return proposal, validate_binding_proposal(proposal, binding_request)

    def _chat_completion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.chat_completion is not None:
            return dict(self.chat_completion(dict(payload)))
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        def call() -> dict[str, Any]:
            req = request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:  # pragma: no cover - network path
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"S4prime binding failed with HTTP {exc.code}: {body}") from exc
            except error.URLError as exc:  # pragma: no cover - network path
                raise RuntimeError(f"S4prime binding failed: {exc.reason}") from exc

        return chat_completion_with_cache(
            cache_path=self.cache_path or default_prompt_cache_path(),
            namespace=f"s4prime_binding:{url}",
            request_payload=payload,
            call=call,
            read_only=self.cache_read_only,
        )


_TOKENS_BY_REQUEST_ID: dict[int, dict[str, AttributeValueToken]] = {}


def build_binding_request(
    *,
    question: str,
    graph: AttributeValueGraph,
    routing: RoutingResult,
    max_candidate_tokens: int = 200,
) -> BindingRequest:
    ordered_tokens = list(graph.tokens)
    prescreen = {
        "enabled": False,
        "original_count": len(ordered_tokens),
        "kept_count": len(ordered_tokens),
        "method": None,
    }
    if len(ordered_tokens) > max_candidate_tokens:
        ordered_tokens = _prescreen_tokens(question, ordered_tokens, max_candidate_tokens)
        prescreen = {
            "enabled": True,
            "original_count": len(graph.tokens),
            "kept_count": len(ordered_tokens),
            "method": "lexical_overlap_then_input_order",
        }

    request = BindingRequest(
        question=question,
        routed_operator=routing.operator,
        operator_catalog=_operator_catalog(),
        candidate_tokens=[_token_payload(token) for token in ordered_tokens],
        question_constants=_question_constant_payloads(question),
        prescreen=prescreen,
    )
    _TOKENS_BY_REQUEST_ID[id(request)] = {token.token_id: token for token in ordered_tokens}
    return request


def parse_binding_proposal(content: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(content, Mapping):
        return dict(content)
    text = str(content or "").strip()
    match = JSON_BLOCK_PATTERN.search(text)
    if match is not None:
        text = match.group(1)
    else:
        text = _slice_first_json_object(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid",
            "_parse_error": str(exc),
        }
    if not isinstance(parsed, dict):
        return {
            "status": "invalid",
            "_parse_error": "binding proposal must be a JSON object",
        }
    return parsed


def validate_binding_proposal(
    proposal: Mapping[str, Any],
    request: BindingRequest,
) -> BindingValidationReport:
    errors: list[str] = []
    details: dict[str, Any] = {}
    if proposal.get("_parse_error"):
        return BindingValidationReport(
            ok=False,
            error_codes=("json_parse_error",),
            details={"parse_error": proposal.get("_parse_error")},
        )

    status = proposal.get("status")
    if status not in {"bound", "abstain"}:
        errors.append("invalid_status")

    steps = proposal.get("steps")
    if status == "abstain":
        if steps:
            errors.append("abstain_contains_steps")
        return BindingValidationReport(ok=not errors, error_codes=tuple(dict.fromkeys(errors)), details=details)

    if not isinstance(steps, list) or not steps:
        errors.append("bound_missing_steps")
        return BindingValidationReport(ok=False, error_codes=tuple(dict.fromkeys(errors)), details=details)

    step_numbers: set[int] = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            errors.append("invalid_step")
            continue
        step_number = _step_number(step, index)
        operator = str(step.get("operator") or proposal.get("operator") or "").upper()
        if operator not in request.operator_catalog:
            errors.append("unknown_operator")
            continue
        bindings = step.get("bindings")
        if not isinstance(bindings, Mapping):
            errors.append("missing_bindings")
            bindings = {}
        _validate_step_bindings(
            errors,
            request=request,
            operator=operator,
            step_number=step_number,
            prior_steps=step_numbers,
            bindings=bindings,
        )
        step_numbers.add(step_number)

    return BindingValidationReport(
        ok=not errors,
        error_codes=tuple(dict.fromkeys(errors)),
        details=details,
    )


def proposal_to_plan(
    proposal: Mapping[str, Any],
    request: BindingRequest,
) -> OperatorPlan | CompositeOperatorPlan:
    report = validate_binding_proposal(proposal, request)
    if not report.ok:
        raise ValueError(f"Invalid S4prime binding proposal: {report.error_codes}")
    steps_payload = proposal.get("steps") or []
    plans = tuple(
        _step_to_plan(step, request=request, fallback_index=index)
        for index, step in enumerate(steps_payload, start=1)
        if isinstance(step, Mapping)
    )
    if len(plans) == 1:
        return plans[0]
    return CompositeOperatorPlan(
        steps=plans,
        confidence=1.0,
        trace={
            "solver": "s4prime.binding_proposal",
            "routed_operator": request.routed_operator,
            "proposal_operator": proposal.get("operator"),
        },
    )


def _system_prompt() -> str:
    return (
        "You are the S4prime binding proposer for a financial numerical reasoning system.\n"
        "Return JSON only. Do not compute the final answer.\n"
        "Your job is to bind operator slots to the closed-set token IDs in the binding_request.\n"
        "Allowed binding values are only candidate token ids, question constant ids, or $stepN references.\n"
        "If the provided candidates do not contain enough evidence, return status abstain.\n"
        "You may choose an operator from operator_catalog when routed_operator is semantically wrong.\n"
        "Use at most two steps. The final step is the proposed answer operator.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "status": "bound" or "abstain",\n'
        '  "operator": "OPERATOR_NAME",\n'
        '  "steps": [\n'
        '    {"step": 1, "operator": "SUM", "bindings": {"target_field": ["t1", "t2"]}, "note": "..."}\n'
        "  ],\n"
        '  "rationale": "one short audit note",\n'
        '  "abstain_reason": null or "insufficient_evidence" or "ambiguous_candidates"\n'
        "}\n\n"
        "FinQA conventions: increase from X to Y normally means percent change; "
        "five year change normally means absolute difference."
    )


def _user_prompt(binding_request: BindingRequest) -> str:
    return (
        "binding_request:\n"
        f"{json.dumps(binding_request.to_dict(), ensure_ascii=False, sort_keys=True)}"
    )


def _validation_retry_prompt(
    proposal: Mapping[str, Any],
    validation: BindingValidationReport,
    binding_request: BindingRequest,
) -> str:
    referenced_ids = sorted(
        {
            str(value)
            for step in proposal.get("steps") or []
            if isinstance(step, Mapping)
            for value in _flatten_binding_values(step.get("bindings") or {})
            if isinstance(value, str) and not value.startswith("$step")
        }
    )
    allowed_ids = sorted(binding_request.token_ids | binding_request.question_constant_ids)
    unknown_ids = sorted(set(referenced_ids) - set(allowed_ids))
    return (
        "validation_feedback:\n"
        f"error_codes={json.dumps(list(validation.error_codes), ensure_ascii=False)}\n"
        f"unknown_ids={json.dumps(unknown_ids, ensure_ascii=False)}\n"
        "The previous proposal failed V-layer validation. Correct it once. "
        "All bindings must use candidate token ids, question constant ids, or valid prior $stepN references. "
        f"Allowed candidate token ids and constants are: {json.dumps(allowed_ids, ensure_ascii=False)}. "
        "Return the complete corrected JSON proposal only."
    )


def _validation_attempt(attempt: int, validation: BindingValidationReport) -> dict[str, Any]:
    return {
        "attempt": attempt,
        **validation.to_dict(),
    }


def _extract_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping) and message.get("content") is not None:
                return str(message.get("content"))
            if first.get("text") is not None:
                return str(first.get("text"))
    if response.get("content") is not None:
        return str(response.get("content"))
    return json.dumps(response, ensure_ascii=False)


def _operator_catalog() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "required_slots": sorted(spec.required_slots),
            "optional_slots": sorted(spec.optional_slots),
            "output_type": spec.output_type,
            "executor_operator": spec.canonical_executor_operator,
            "description": spec.description,
        }
        for name, spec in sorted(OPERATOR_SPECS.items())
    }


def _token_payload(token: AttributeValueToken) -> dict[str, Any]:
    return {
        "id": token.token_id,
        "metric": token.field_name,
        "metric_label": token.field_label,
        "year": token.year,
        "period": _token_period(token),
        "value": token.value,
        "scale": _unit_scale_hint(token.unit),
        "unit": token.unit,
        "is_aggregate": bool(token.is_aggregate),
        "provenance": token.provenance_channel or _source_provenance(token),
        "entity": token.company_name,
        "dimensions": dict(token.dimensions or {}),
        "evidence": _token_evidence(token),
    }


def _question_constant_payloads(question: str) -> list[dict[str, Any]]:
    constants = []
    for index, row in enumerate(resolve_question_constant_slots(question), start=1):
        constants.append({"id": f"qc{index}", **row})
    return constants


def _prescreen_tokens(
    question: str,
    tokens: list[AttributeValueToken],
    max_candidate_tokens: int,
) -> list[AttributeValueToken]:
    query_terms = _meaningful_terms(question)
    scored: list[tuple[int, int, AttributeValueToken]] = []
    for index, token in enumerate(tokens):
        token_terms = _meaningful_terms(_token_text_for_screening(token))
        overlap = len(query_terms & token_terms)
        scored.append((-overlap, index, token))
    scored.sort(key=lambda item: (item[0], item[1], item[2].token_id))
    return [token for _, _, token in scored[:max_candidate_tokens]]


def _validate_step_bindings(
    errors: list[str],
    *,
    request: BindingRequest,
    operator: str,
    step_number: int,
    prior_steps: set[int],
    bindings: Mapping[str, Any],
) -> None:
    required = set(request.operator_catalog[operator]["required_slots"])
    provided = set(str(key) for key in bindings)
    missing = sorted(required - _provided_signature_slots(operator, provided))
    if missing:
        errors.append("missing_required_slots")
    allowed = required | set(request.operator_catalog[operator]["optional_slots"]) | _alias_slots_for_operator(operator)
    extra = sorted(provided - allowed)
    if extra:
        errors.append("extra_slots")
    for value in bindings.values():
        for item in _flatten_binding_values(value):
            _validate_binding_value(
                errors,
                request=request,
                value=item,
                step_number=step_number,
                prior_steps=prior_steps,
            )


def _validate_binding_value(
    errors: list[str],
    *,
    request: BindingRequest,
    value: Any,
    step_number: int,
    prior_steps: set[int],
) -> None:
    if isinstance(value, str) and value.startswith("$step"):
        match = STEP_REFERENCE_PATTERN.match(value)
        if match is None:
            errors.append("invalid_step_reference")
            return
        referenced = int(match.group("step"))
        if referenced >= step_number or referenced not in prior_steps:
            errors.append("future_step_reference")
        return
    if isinstance(value, str) and value.startswith("$"):
        errors.append("invalid_step_reference")
        return
    if isinstance(value, str) and value.startswith("qc"):
        if value not in request.question_constant_ids:
            errors.append("unknown_binding_id")
        return
    if isinstance(value, str) and value not in request.token_ids:
        errors.append("unknown_binding_id")


def _step_to_plan(
    step: Mapping[str, Any],
    *,
    request: BindingRequest,
    fallback_index: int,
) -> OperatorPlan:
    operator = str(step.get("operator") or "").upper()
    step_number = _step_number(step, fallback_index)
    bindings = step.get("bindings")
    if not isinstance(bindings, Mapping):
        bindings = {}
    slots: dict[str, Any] = {}
    depends_on: dict[str, Any] = {}
    proposal_token_bindings: dict[str, list[str]] = {}
    for slot_name, raw_value in bindings.items():
        slot_key = str(slot_name)
        token_ids = _binding_token_ids(raw_value, request)
        if token_ids:
            proposal_token_bindings[slot_key] = token_ids
        mapped = _map_binding_slot(operator, slot_key, raw_value, request)
        slots.update(mapped.slots)
        depends_on.update(mapped.depends_on)
    if proposal_token_bindings:
        slots["proposal_token_bindings"] = Slot(
            surface="s4prime binding proposal",
            grounded_value=proposal_token_bindings,
            confidence=1.0,
        )
    return OperatorPlan(
        operator=operator,
        slots=slots,
        depends_on=depends_on or None,
        confidence=1.0,
        trace={
            "solver": "s4prime.binding_proposal",
            "step_id": f"s{step_number}",
            "proposal_note": step.get("note"),
        },
    )


@dataclass(frozen=True)
class _MappedBinding:
    slots: dict[str, Any]
    depends_on: dict[str, Any]


def _map_binding_slot(
    operator: str,
    slot_key: str,
    raw_value: Any,
    request: BindingRequest,
) -> _MappedBinding:
    values = list(_flatten_binding_values(raw_value))
    if operator == "SUM" and slot_key == "target_field":
        token_ids = [value for value in values if isinstance(value, str) and value in request.token_ids]
        field = _field_for_token_ids(request, token_ids)
        return _MappedBinding(
            slots={
                "target_field": Slot(surface=slot_key, grounded_value=field, confidence=1.0),
                "selected_token_ids": Slot(surface=slot_key, grounded_value=token_ids, confidence=1.0),
            },
            depends_on={},
        )
    if operator == "PRODUCT" and slot_key in {"factor_token_ids", "factors", "target_field"}:
        token_ids = [value for value in values if isinstance(value, str) and value in request.token_ids]
        constant_ids = [value for value in values if isinstance(value, str) and value in request.question_constant_ids]
        slots = {
            "factor_token_ids": Slot(surface=slot_key, grounded_value=token_ids, confidence=1.0),
        }
        constants = [_constant_by_id(request, constant_id) for constant_id in constant_ids]
        constants = [row for row in constants if row is not None]
        if constants:
            slots["constant_slot"] = Slot(surface=slot_key, grounded_value=constants, confidence=1.0)
        return _MappedBinding(slots=slots, depends_on={})
    if operator in {"RATIO", "SHARE", "MARGIN"}:
        return _map_ratio_binding(slot_key, raw_value, request)

    if _is_step_reference(raw_value):
        return _MappedBinding(
            slots={},
            depends_on={slot_key: _step_reference_to_executor_reference(str(raw_value), default_attr="answer")},
        )
    token_ids = [value for value in values if isinstance(value, str) and value in request.token_ids]
    if token_ids and slot_key.endswith("_field"):
        return _MappedBinding(
            slots={slot_key: Slot(surface=slot_key, grounded_value=_field_for_token_ids(request, token_ids), confidence=1.0)},
            depends_on={},
        )
    semantic_value = _semantic_slot_value(slot_key, token_ids, request)
    if semantic_value is not None:
        return _MappedBinding(
            slots={slot_key: Slot(surface=slot_key, grounded_value=semantic_value, confidence=1.0)},
            depends_on={},
        )
    if len(values) == 1:
        value = values[0]
    else:
        value = values
    return _MappedBinding(
        slots={slot_key: Slot(surface=slot_key, grounded_value=value, confidence=1.0)},
        depends_on={},
    )


def _map_ratio_binding(
    slot_key: str,
    raw_value: Any,
    request: BindingRequest,
) -> _MappedBinding:
    step_reference = _single_step_reference(raw_value)
    if slot_key in {"numerator_field", "part"}:
        if step_reference is not None:
            return _MappedBinding(
                slots={
                    "numerator_field": Slot(
                        surface=slot_key,
                        grounded_value="__step_result__",
                        confidence=1.0,
                    ),
                },
                depends_on={
                    "numerator_value": _step_reference_to_executor_reference(
                        step_reference,
                        default_attr="answer",
                    )
                },
            )
        token_ids = _binding_token_ids(raw_value, request)
        return _MappedBinding(
            slots={
                "numerator_field": Slot(surface=slot_key, grounded_value=_field_for_token_ids(request, token_ids), confidence=1.0),
                "numerator_token_ids": Slot(surface=slot_key, grounded_value=token_ids, confidence=1.0),
            },
            depends_on={},
        )
    if slot_key in {"denominator_field", "whole"}:
        if step_reference is not None:
            return _MappedBinding(
                slots={
                    "denominator_field": Slot(
                        surface=slot_key,
                        grounded_value="__step_result__",
                        confidence=1.0,
                    ),
                },
                depends_on={
                    "denominator_value": _step_reference_to_executor_reference(
                        step_reference,
                        default_attr="answer",
                    )
                },
            )
        token_ids = _binding_token_ids(raw_value, request)
        return _MappedBinding(
            slots={
                "denominator_field": Slot(surface=slot_key, grounded_value=_field_for_token_ids(request, token_ids), confidence=1.0),
                "denominator_token_ids": Slot(surface=slot_key, grounded_value=token_ids, confidence=1.0),
            },
            depends_on={},
        )
    return _MappedBinding(
        slots={slot_key: Slot(surface=slot_key, grounded_value=raw_value, confidence=1.0)},
        depends_on={},
    )


def _binding_token_ids(value: Any, request: BindingRequest) -> list[str]:
    return [
        item
        for item in _flatten_binding_values(value)
        if isinstance(item, str) and item in request.token_ids
    ]


def _single_step_reference(value: Any) -> str | None:
    values = _flatten_binding_values(value)
    if len(values) != 1 or not _is_step_reference(values[0]):
        return None
    return str(values[0])


def _field_for_token_ids(request: BindingRequest, token_ids: list[str]) -> str | None:
    if not token_ids:
        return None
    token = request.token_by_id(token_ids[0])
    return token.field_name if token is not None else None


def _semantic_slot_value(
    slot_key: str,
    token_ids: list[str],
    request: BindingRequest,
) -> Any | None:
    if not token_ids:
        return None
    token = request.token_by_id(token_ids[0])
    if token is None:
        return None
    if slot_key in {"entity", "left_entity", "right_entity"}:
        return token.entity_id or token.company_name
    if slot_key in {"year", "from_time", "to_time", "left_time", "right_time"}:
        return token.year
    if slot_key == "unit":
        return token.unit
    if slot_key == "industry":
        return token.industry
    return None


def _numerator_field_hint(request: BindingRequest) -> str | None:
    for row in request.candidate_tokens:
        metric = row.get("metric")
        if metric is not None:
            return str(metric)
    return None


def _constant_by_id(request: BindingRequest, constant_id: str) -> dict[str, Any] | None:
    for row in request.question_constants:
        if row.get("id") == constant_id:
            constant = dict(row)
            constant.pop("id", None)
            return constant
    return None


def _step_reference_to_executor_reference(reference: str, *, default_attr: str) -> str:
    match = STEP_REFERENCE_PATTERN.match(reference)
    if match is None:
        return reference
    step = match.group("step")
    attr = reference.split(".", 1)[1] if "." in reference else default_attr
    if attr == "selected_token_ids":
        attr = "selected_token_ids"
    return f"$s{step}.{attr}"


def _provided_signature_slots(operator: str, provided: set[str]) -> set[str]:
    signature = set(provided)
    if operator in {"RATIO", "SHARE", "MARGIN"}:
        if "part" in provided:
            signature.add("numerator_field")
        if "whole" in provided:
            signature.add("denominator_field")
    return signature


def _alias_slots_for_operator(operator: str) -> set[str]:
    if operator in {"RATIO", "SHARE", "MARGIN"}:
        return {"part", "whole"}
    if operator == "PRODUCT":
        return {"factors"}
    return set()


def _flatten_binding_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for item in value:
            items.extend(_flatten_binding_values(item))
        return tuple(items)
    if isinstance(value, Mapping):
        items = []
        for item in value.values():
            items.extend(_flatten_binding_values(item))
        return tuple(items)
    return (value,)


def _is_step_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$step")


def _step_number(step: Mapping[str, Any], fallback_index: int) -> int:
    try:
        return int(step.get("step", fallback_index))
    except (TypeError, ValueError):
        return fallback_index


def _slice_first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _token_period(token: AttributeValueToken) -> Any:
    dimensions = token.dimensions or {}
    for key in ("period", "quarter", "date", "column_label"):
        if key in dimensions:
            return dimensions[key]
    return None


def _unit_scale_hint(unit: str | None) -> float | None:
    text = str(unit or "").lower()
    if "billion" in text:
        return 1e9
    if "million" in text:
        return 1e6
    if "thousand" in text:
        return 1e3
    return None


def _source_provenance(token: AttributeValueToken) -> str | None:
    if token.source is not None and token.source.table is not None:
        return "table"
    if token.source is not None and token.source.text_excerpt:
        return "text"
    return None


def _token_evidence(token: AttributeValueToken) -> str:
    source = token.source
    parts = [
        token.raw_label,
        token.field_label,
        token.company_name,
        str(token.year) if token.year is not None else None,
        str(token.value),
    ]
    if source is not None:
        parts.extend([source.column, source.text_excerpt])
    return " | ".join(str(part) for part in parts if part)


def _meaningful_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "by",
        "for",
        "from",
        "in",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
    }
    return {
        term
        for term in normalize_identifier(text).split("_")
        if term and term not in stopwords
    }


def _token_text_for_screening(token: AttributeValueToken) -> str:
    values: list[Any] = [
        token.field_name,
        token.field_label,
        token.raw_label,
        token.company_name,
        token.entity_id,
        token.unit,
        token.year,
    ]
    if token.dimensions:
        values.extend(token.dimensions.values())
    if token.source is not None:
        values.extend([token.source.column, token.source.text_excerpt])
    return " ".join(str(value) for value in values if value is not None)
