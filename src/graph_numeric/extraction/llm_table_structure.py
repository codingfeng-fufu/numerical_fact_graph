from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
    TokenSource,
    normalize_identifier,
)
from graph_numeric.extraction.llm_extraction import (
    JSON_BLOCK_PATTERN,
    LLMGraphExtractorConfig,
    _coerce_float,
    _first_non_empty,
    _normalize_unit,
    _optional_int,
    _slice_first_json_object,
    infer_table_unit_context,
)
from graph_numeric.runtime.llm_cache import chat_completion_with_cache, default_prompt_cache_path


@dataclass(frozen=True)
class LLMTableStructureResult:
    graph: AttributeValueGraph
    metadata: dict[str, Any]
    payload: dict[str, Any]


class LLMTableStructureParser:
    """LLM table structure parser that emits auditable numeric graph facts.

    The model is only allowed to normalize table structure. It does not answer
    the user question; graph execution remains deterministic downstream.
    """

    def __init__(
        self,
        config: LLMGraphExtractorConfig | None = None,
        *,
        max_chars: int = 12000,
    ) -> None:
        self.config = config or LLMGraphExtractorConfig.from_env()
        self.max_chars = max_chars

    @property
    def available(self) -> bool:
        return self.config.available

    def parse(
        self,
        document_text: str,
        *,
        query: str | None = None,
        source_name: str = "document",
    ) -> LLMTableStructureResult:
        if not self.available:
            raise RuntimeError("LLM table structure parsing is not configured.")
        if not document_text.strip():
            raise ValueError("Document text is empty.")
        payload = self._chat_completion(
            [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(
                        document_text=document_text,
                        query=query,
                        source_name=source_name,
                    ),
                },
            ]
        )
        content = _extract_content(payload)
        parsed = _parse_response_json(content)
        graph, validation = build_graph_from_normalized_table_payload(
            parsed,
            source_name=source_name,
            document_text=document_text,
        )
        metadata = {
            "source": "llm",
            "parser": "llm_table_structure",
            "model": self.config.model,
            "confidence": _coerce_float(parsed.get("confidence")),
            "record_count": len(graph.tokens),
            "warnings": list(parsed.get("warnings") or []) + validation["warnings"],
            "validation_errors": validation["validation_errors"],
            "parse_failure_count": validation.get("parse_failure_count", 0),
            "source_coverage": validation["source_coverage"],
        }
        return LLMTableStructureResult(graph=graph, metadata=metadata, payload=parsed)

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        def call() -> dict[str, Any]:
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:  # pragma: no cover - network path
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"LLM table structure parsing failed with HTTP {exc.code}: {body}") from exc
            except error.URLError as exc:  # pragma: no cover - network path
                raise RuntimeError(f"LLM table structure parsing failed: {exc.reason}") from exc

        return chat_completion_with_cache(
            cache_path=default_prompt_cache_path(),
            namespace=f"llm_table_structure:{url}",
            request_payload=payload,
            call=call,
        )

    def _system_prompt(self) -> str:
        return (
            "You are a table structure parser for financial documents.\n"
            "Return JSON only. Do not answer the user question.\n"
            "Your task is to convert tables into normalized numeric facts with auditable structure.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "document_id": string,\n'
            '  "parser": "llm_table_structure",\n'
            '  "confidence": number,\n'
            '  "tables": [\n'
            "    {\n"
            '      "table_id": string,\n'
            '      "table_label": string or null,\n'
            '      "unit_context": {"currency": string or null, "scale": string or null},\n'
            '      "facts": [\n'
            "        {\n"
            '          "value": number,\n'
            '          "raw_value": string,\n'
            '          "row_index": integer,\n'
            '          "column_index": integer,\n'
            '          "row_path": [string],\n'
            '          "column_path": [string],\n'
            '          "year": integer or null,\n'
            '          "unit": string or null,\n'
            '          "source_text": string\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            '  "warnings": [string]\n'
            "}\n\n"
            "Rules:\n"
            "- Extract numeric table cells, not prose-only statements.\n"
            "- Preserve row_path and column_path as the semantic header path for each value.\n"
            "- For multi-level headers, include every relevant level in order.\n"
            "- Do not collapse a date header into a metric header; keep both in column_path.\n"
            "- Keep value as a pure number without commas, currency symbols, or percent signs.\n"
            "- raw_value must be the exact original cell text, such as $44,705 or (226).\n"
            "- row_index and column_index are zero-based indexes in the visible table grid, including header rows.\n"
            "- source_text must quote the original row or local table fragment that supports the value.\n"
            "- If a unit such as in thousands appears in the table, put the normalized unit on every money fact.\n"
            "- If the user question is supplied, use it only to prioritize relevant tables and rows; do not answer it."
        )

    def _user_prompt(
        self,
        *,
        document_text: str,
        query: str | None,
        source_name: str,
    ) -> str:
        clipped = document_text[: self.max_chars]
        query_line = query.strip() if query and query.strip() else "None"
        return (
            f"Document id: {source_name}\n"
            f"User question for prioritization only: {query_line}\n\n"
            "Document text:\n"
            f"{clipped}"
        )


def build_graph_from_normalized_table_payload(
    payload: dict[str, Any],
    *,
    source_name: str,
    document_text: str | None = None,
) -> tuple[AttributeValueGraph, dict[str, Any]]:
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("Normalized table payload must contain a tables list.")

    tokens: list[AttributeValueToken] = []
    warnings: list[str] = []
    validation_errors: list[str] = []
    seen: set[tuple[object, ...]] = set()
    for table_index, table in enumerate(tables):
        if not isinstance(table, dict):
            raise ValueError(f"Normalized table {table_index} must be an object.")
        table_id = _first_non_empty([table.get("table_id"), f"table_{table_index}"])
        table_label = _first_non_empty([table.get("table_label"), table_id])
        unit_context = _unit_context(table, document_text)
        facts = table.get("facts")
        if not isinstance(facts, list):
            raise ValueError(f"Normalized table {table_id} must contain a facts list.")
        for fact_index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                validation_errors.append(
                    f"fact_parse_failure:{table_id}:{fact_index}:Normalized fact {table_id}:{fact_index} must be an object."
                )
                continue
            try:
                token = _token_from_normalized_fact(
                    fact,
                    fact_index=fact_index,
                    table_id=table_id,
                    table_label=table_label,
                    unit_context=unit_context,
                    source_name=source_name,
                    document_text=document_text,
                )
            except ValueError as exc:
                validation_errors.append(f"fact_parse_failure:{table_id}:{fact_index}:{exc}")
                continue
            candidate_tokens = [token]
            row_metric_projection = _row_metric_projection_token(token)
            if row_metric_projection is not None:
                candidate_tokens.append(row_metric_projection)
            for candidate in candidate_tokens:
                key = (
                    normalize_identifier(candidate.company_name),
                    candidate.field_name,
                    candidate.year,
                    candidate.source.row if candidate.source else None,
                    candidate.source.column if candidate.source else None,
                    candidate.value,
                )
                if key in seen:
                    warnings.append(
                        f"duplicate_fact_skipped:{candidate.field_name}:{candidate.company_name}:{candidate.year}"
                    )
                    continue
                seen.add(key)
                tokens.append(candidate)
    if not tokens:
        raise ValueError("Normalized table payload did not produce any graph tokens.")
    validation = {
        "warnings": warnings,
        "validation_errors": validation_errors,
        "parse_failure_count": len(validation_errors),
        "source_coverage": round(
            sum(1 for token in tokens if token.source and token.source.text_excerpt) / len(tokens),
            4,
        ),
    }
    return AttributeValueGraph(tuple(tokens), source_name=source_name), validation


def _token_from_normalized_fact(
    fact: dict[str, Any],
    *,
    fact_index: int,
    table_id: str,
    table_label: str,
    unit_context: dict[str, Any],
    source_name: str,
    document_text: str | None,
) -> AttributeValueToken:
    value = _coerce_float(fact.get("value"))
    if value is None:
        raise ValueError(f"Normalized fact {table_id}:{fact_index} has non-numeric value.")
    raw_value = _first_non_empty([fact.get("raw_value"), fact.get("source_value")])
    if not raw_value:
        raise ValueError(f"Normalized fact {table_id}:{fact_index} is missing raw_value.")
    if document_text is not None and raw_value not in document_text:
        raise ValueError(f"Normalized fact {table_id}:{fact_index} raw_value not found in document text: {raw_value}")
    row_index = _optional_int(fact.get("row_index"))
    column_index = _optional_int(fact.get("column_index"))
    if row_index is None or column_index is None:
        raise ValueError(f"Normalized fact {table_id}:{fact_index} is missing row_index or column_index.")
    row_path = _string_list(fact.get("row_path"))
    column_path = _string_list(fact.get("column_path"))
    if not row_path:
        row_path = [_first_non_empty([fact.get("row_label"), f"row_{row_index}"])]
    if not column_path:
        column_path = [_first_non_empty([fact.get("column_label"), f"column_{column_index}"])]
    field_label = _field_label_from_column_path(column_path)
    field_name = normalize_identifier(_first_non_empty([fact.get("field_name"), field_label]))
    entity = _first_non_empty([fact.get("entity"), row_path[-1], f"row_{row_index}"])
    year = _optional_int(fact.get("year")) or _year_from_labels(column_path)
    unit = _normalize_unit(_first_non_empty([fact.get("unit")]), unit_context) or None
    column_label = " / ".join(column_path)
    raw_excerpt = raw_value
    source_text = _first_non_empty([fact.get("source_text"), fact.get("evidence_text")])
    char_start = document_text.find(raw_value) if document_text is not None else -1
    source = TokenSource(
        document_id=source_name,
        table=table_label or table_id,
        row=row_index,
        column=column_label,
        char_start=char_start if char_start >= 0 else None,
        char_end=char_start + len(raw_value) if char_start >= 0 else None,
        text_excerpt=raw_excerpt,
    )
    dimensions: dict[str, object] = {
        "row_path": " / ".join(row_path),
        "column_path": column_label,
        "row_label": row_path[-1],
        "column_label": field_label,
        "table_id": table_id,
        "row_id": row_index,
        "col_id": column_index,
        "provenance_channel": "table",
    }
    is_aggregate = _is_aggregate_row_path(row_path)
    if is_aggregate:
        dimensions["is_aggregate"] = True
        dimensions["is_total_row"] = True
    if source_text:
        dimensions["source_row_text"] = source_text
    if year is not None:
        dimensions["period_type"] = "fiscal_year"
    entity_id = f"{normalize_identifier(entity)}:{year if year is not None else row_index}"
    return AttributeValueToken(
        token_id=f"{entity_id}:{field_name}:{fact_index}",
        entity_id=entity_id,
        company_name=entity,
        field_name=field_name,
        field_label=field_label,
        value=float(value),
        year=year,
        source=source,
        unit=unit,
        dimensions=dimensions,
        table_id=table_id,
        row_id=row_index,
        col_id=column_index,
        is_aggregate=is_aggregate,
        provenance_channel="table",
        raw_label=row_path[-1] if row_path else None,
    )


def _unit_context(table: dict[str, Any], document_text: str | None) -> dict[str, Any]:
    context = table.get("unit_context")
    if isinstance(context, dict) and context:
        return dict(context)
    return infer_table_unit_context(document_text or "")


def _row_metric_projection_token(token: AttributeValueToken) -> AttributeValueToken | None:
    dimensions = dict(token.dimensions or {})
    row_label = str(dimensions.get("row_label") or "").strip()
    if not row_label:
        return None
    row_field_name = normalize_identifier(row_label)
    if not row_field_name or row_field_name == token.field_name:
        return None
    column_label = str(dimensions.get("column_label") or token.field_label or "").strip()
    if not _column_label_is_period_or_total(column_label):
        return None
    projected_dimensions = {
        **dimensions,
        "projection": "row_metric",
        "column_metric_field": token.field_name,
    }
    table_scope = _first_non_empty([
        dimensions.get("table_id"),
        token.source.table if token.source is not None else None,
        token.company_name,
    ])
    period_scope = token.year if token.year is not None else normalize_identifier(column_label)
    return AttributeValueToken(
        token_id=f"{token.token_id}:row_metric",
        entity_id=f"{normalize_identifier(str(table_scope))}:{period_scope}",
        company_name=token.company_name,
        field_name=row_field_name,
        field_label=row_label,
        value=token.value,
        year=token.year,
        industry=token.industry,
        source=token.source,
        unit=token.unit,
        canonical_concept_id=token.canonical_concept_id,
        dimensions=projected_dimensions,
        table_id=token.table_id,
        row_id=token.row_id,
        col_id=token.col_id,
        is_aggregate=token.is_aggregate,
        provenance_channel=token.provenance_channel,
        raw_label=row_label,
        external_concept_ids=token.external_concept_ids,
    )


def _column_label_is_period_or_total(label: str) -> bool:
    normalized = normalize_identifier(label)
    if normalized in {"total", "subtotal"}:
        return True
    if _is_period_label(label):
        return True
    return bool(
        re.search(
            r"(?:^|_)(?:less_than|more_than|thereafter|after|before|remaining|remainder)(?:_|$)",
            normalized,
        )
    )


def _field_label_from_column_path(column_path: list[str]) -> str:
    for label in reversed(column_path):
        if not _is_period_label(label) and not _looks_like_unit_label(label):
            return label
    return column_path[-1]


def _is_aggregate_row_path(row_path: list[str]) -> bool:
    if not row_path:
        return False
    normalized = normalize_identifier(" ".join(row_path))
    return bool(re.search(r"(?:^|_)total(?:_|$)|(?:^|_)aggregate(?:_|$)|(?:^|_)overall(?:_|$)", normalized))


def _is_period_label(label: str) -> bool:
    return bool(re.search(r"\b(?:19\d{2}|20\d{2})\b", label))


def _looks_like_unit_label(label: str) -> bool:
    return bool(re.search(r"\b(?:in\s+)?(?:thousands?|millions?|billions?)\b", label, re.IGNORECASE))


def _year_from_labels(labels: list[str]) -> int | None:
    for label in labels:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", label)
        if match:
            return int(match.group(1))
    return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response does not contain choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        if parts:
            return "\n".join(parts)
    raise ValueError("LLM response does not contain text content.")


def _parse_response_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    candidates = [stripped]
    fenced = JSON_BLOCK_PATTERN.search(stripped)
    if fenced:
        candidates.insert(0, fenced.group(1))
    brace_candidate = _slice_first_json_object(stripped)
    if brace_candidate:
        candidates.insert(0, brace_candidate)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("LLM response is not valid JSON.")
