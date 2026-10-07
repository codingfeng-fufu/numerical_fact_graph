from __future__ import annotations

import math
import json
import re
from collections import Counter
from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph


_PROGRAM_NUMBER_RE = re.compile(
    r"(?<![#\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])"
)
_QUESTION_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_SOURCE_NUMBER_RE = re.compile(
    r"\(\s*[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\$?\s*\d[\d,]*(?:\.\d+)?"
)


def extract_program_operands(program: str | None) -> list[float]:
    """Return literal numeric operands, excluding step refs and named constants."""

    return [float(match.group(0)) for match in _PROGRAM_NUMBER_RE.finditer(program or "")]


def classify_program_operand_reachability(
    *,
    program: str | None,
    question: str,
    graph: AttributeValueGraph,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    candidates = [
        {
            "id": token.token_id,
            "value": token.value,
            "field_name": token.field_name,
            "field_label": token.field_label,
            "year": token.year,
            "table_id": token.table_id,
            "row_id": token.row_id,
            "col_id": token.col_id,
            "is_aggregate": token.is_aggregate,
            "provenance_channel": token.provenance_channel,
        }
        for token in graph.tokens
    ]
    return _classify_candidate_reachability(
        program=program,
        question=question,
        candidates=candidates,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def classify_cached_candidate_reachability(
    *,
    program: str | None,
    question: str,
    candidate_tokens: list[dict[str, Any]],
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    return _classify_candidate_reachability(
        program=program,
        question=question,
        candidates=candidate_tokens,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def binding_requests_from_prompt_cache(
    cache_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    requests: dict[str, list[dict[str, Any]]] = {}
    for entry in cache_payload.values():
        request = entry.get("request") if isinstance(entry, dict) else None
        messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(messages, list):
            continue
        user_content = next(
            (
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
        prefix = "binding_request:\n"
        if not user_content.startswith(prefix):
            continue
        try:
            binding_request = json.loads(user_content[len(prefix) :])
        except (json.JSONDecodeError, TypeError):
            continue
        question = str(binding_request.get("question") or "")
        candidates = binding_request.get("candidate_tokens")
        if not question or not isinstance(candidates, list):
            continue
        requests.setdefault(question, []).append(binding_request)
    return requests


def locate_operand_sources(
    *,
    operands: list[float],
    table: Any,
    text: Any,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> list[dict[str, Any]]:
    table_values = _source_values(table)
    text_values = _source_values(text)
    locations: list[dict[str, Any]] = []
    for operand in operands:
        in_table = any(
            math.isclose(abs(operand), abs(value), rel_tol=rel_tol, abs_tol=abs_tol)
            for value in table_values
        )
        in_text = any(
            math.isclose(abs(operand), abs(value), rel_tol=rel_tol, abs_tol=abs_tol)
            for value in text_values
        )
        location = (
            "both"
            if in_table and in_text
            else "table"
            if in_table
            else "text"
            if in_text
            else "unresolved"
        )
        locations.append({"operand": operand, "location": location})
    return locations


def _source_values(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_source_values(item))
        return result
    result = []
    for match in _SOURCE_NUMBER_RE.finditer(str(value or "")):
        surface = match.group(0).strip()
        negative = surface.startswith("(") or surface.startswith("-")
        digits = re.sub(r"[^0-9.]", "", surface)
        if not digits:
            continue
        parsed = float(digits)
        result.append(-parsed if negative else parsed)
    return result


def _classify_candidate_reachability(
    *,
    program: str | None,
    question: str,
    candidates: list[dict[str, Any]],
    rel_tol: float,
    abs_tol: float,
) -> dict[str, Any]:
    operands = extract_program_operands(program)
    question_values = [float(value) for value in _QUESTION_NUMBER_RE.findall(question or "")]
    question_counts = Counter(question_values)
    question_constants: list[float] = []
    document_operands: list[float] = []
    for operand in operands:
        matched_question_value = _matching_counter_key(
            operand,
            question_counts,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        if matched_question_value is not None and question_counts[matched_question_value] > 0:
            question_counts[matched_question_value] -= 1
            question_constants.append(operand)
        else:
            document_operands.append(operand)

    unique_operands: list[float] = []
    for operand in document_operands:
        if not any(
            math.isclose(operand, existing, rel_tol=rel_tol, abs_tol=abs_tol)
            for existing in unique_operands
        ):
            unique_operands.append(operand)
    missing_operands = [
        operand
        for operand in unique_operands
        if not any(
            math.isclose(
                float(candidate["value"]),
                operand,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for candidate in candidates
        )
    ]

    available = list(candidates)
    matched_token_ids: list[str] = []
    matched_tokens: list[dict[str, Any]] = []
    missing_operand_occurrences: list[float] = []
    for operand in document_operands:
        match_index = next(
            (
                index
                for index, token in enumerate(available)
                if math.isclose(
                    float(token["value"]),
                    operand,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
            ),
            None,
        )
        if match_index is None:
            missing_operand_occurrences.append(operand)
            continue
        token = available.pop(match_index)
        token_id = str(token.get("id") or token.get("token_id") or "")
        matched_token_ids.append(token_id)
        matched_tokens.append(
            {
                "token_id": token_id,
                "value": token.get("value"),
                "field_name": token.get("field_name") or token.get("metric"),
                "field_label": token.get("field_label") or token.get("metric_label"),
                "year": token.get("year"),
                "table_id": token.get("table_id") or (token.get("dimensions") or {}).get("table_id"),
                "row_id": token.get("row_id") or (token.get("dimensions") or {}).get("row_id"),
                "col_id": token.get("col_id") or (token.get("dimensions") or {}).get("col_id"),
                "is_aggregate": bool(token.get("is_aggregate")),
                "provenance_channel": token.get("provenance_channel") or token.get("provenance"),
            }
        )

    if not operands:
        classification = "non_numeric_or_unparsed_program"
    elif not document_operands:
        classification = "question_constants_only"
    elif missing_operands:
        classification = "s2_gold_token_missing"
    else:
        classification = "s4prime_binding_shortfall"
    return {
        "classification": classification,
        "program_operands": operands,
        "question_constants": question_constants,
        "document_operands": document_operands,
        "missing_operands": missing_operands,
        "missing_operand_occurrences": missing_operand_occurrences,
        "matched_token_ids": matched_token_ids,
        "matched_tokens": matched_tokens,
    }


def _matching_counter_key(
    value: float,
    counts: Counter[float],
    *,
    rel_tol: float,
    abs_tol: float,
) -> float | None:
    for candidate, count in counts.items():
        if count > 0 and math.isclose(
            value,
            candidate,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return candidate
    return None
