from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource, normalize_identifier
from graph_numeric.core.canonical_concepts import (
    CANONICAL_DIMENSION_KEYS,
    UNSUPPORTED_CONCEPT_ID,
    canonical_external_mappings,
    concept_prompt_catalog,
    get_canonical_concept,
    resolve_canonical_concept,
    validate_concept_record,
)
from graph_numeric.learning.question_intent import classify_question_intent
from graph_numeric.runtime.llm_cache import chat_completion_with_cache, default_prompt_cache_path


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
NUMERIC_CANDIDATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<number>"
    r"(?:[$€£]\s*)?"
    r"\(?[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?\)?"
    r"(?:\s*(?:%|percent(?:age)?|bps|basis points?|million|millions|billion|billions|"
    r"thousand|thousands|USD|U\.S\. dollars?|dollars?|shares?))?"
    r")"
    r"(?![A-Za-z0-9_]|\.\d)",
    re.IGNORECASE,
)
MONTH_NAME_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\b",
    re.IGNORECASE,
)
REFERENCE_NUMBER_PREFIX_PATTERN = re.compile(
    r"(?:\b(?:item|note|footnote|table|page|section|exhibit|figure|fig|appendix|"
    r"part|schedule|chapter|no|number|p)\.?\s*|[#])$",
    re.IGNORECASE,
)
QUERY_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "between",
        "by",
        "change",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
    }
)


@dataclass(frozen=True)
class LLMGraphExtractorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 45.0

    @classmethod
    def from_env(cls) -> "LLMGraphExtractorConfig":
        env_file_values = _load_env_file(Path.cwd() / ".env")

        def get_value(*keys: str) -> str:
            return _first_non_empty(os.getenv(key) or env_file_values.get(key) for key in keys)

        explicit_base_url = get_value("LLM_GRAPH_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL")
        if explicit_base_url:
            base_url = explicit_base_url.rstrip("/")
        elif get_value("DASHSCOPE_API_KEY"):
            base_url = DEFAULT_DASHSCOPE_BASE_URL
        elif get_value("DEEPSEEK_API_KEY"):
            base_url = "https://api.deepseek.com"
        else:
            base_url = ""

        if "deepseek" in base_url.lower():
            api_key = get_value("LLM_GRAPH_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY") or None
        else:
            api_key = get_value("LLM_GRAPH_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY") or None
        if not base_url and api_key:
            base_url = DEFAULT_OPENAI_BASE_URL

        model = (
            get_value("LLM_GRAPH_MODEL", "DEEPSEEK_MODEL_SMART", "DEEPSEEK_MODEL_FAST", "OPENAI_MODEL")
            or ("qwen-plus" if get_value("DASHSCOPE_API_KEY") else "")
        ).strip()
        try:
            timeout_seconds = float(get_value("LLM_GRAPH_TIMEOUT_SECONDS") or "45")
        except ValueError:
            timeout_seconds = 45.0
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    @property
    def available(self) -> bool:
        if not self.base_url or not self.model:
            return False
        if self.base_url == DEFAULT_OPENAI_BASE_URL and not self.api_key:
            return False
        return True


@dataclass(frozen=True)
class LLMGraphExtractionResult:
    graph: AttributeValueGraph
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    kind: str
    text: str
    table_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkSelection:
    chunks: list[DocumentChunk]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class NumericEvidenceChunk:
    chunk_id: str
    source_chunk_id: str
    source_kind: str
    table_index: int | None
    text: str
    value_surface: str
    normalized_value: float | None
    unit_hint: str | None
    year_hint: int | None
    source_value_start: int
    source_value_end: int
    source_context_start: int
    source_context_end: int
    text_value_start: int
    text_value_end: int


@dataclass(frozen=True)
class RejectedNumericCandidate:
    source_chunk_id: str
    source_kind: str
    table_index: int | None
    text: str
    value_surface: str
    normalized_value: float | None
    unit_hint: str | None
    reject_reason: str
    source_value_start: int
    source_value_end: int
    source_context_start: int
    source_context_end: int


@dataclass(frozen=True)
class NumericEvidenceExtraction:
    chunks: list[NumericEvidenceChunk]
    rejected_candidates: list[RejectedNumericCandidate]


@dataclass(frozen=True)
class NumericEvidenceSelection:
    chunks: list[NumericEvidenceChunk]
    diagnostics: dict[str, Any]


class LLMGraphExtractor:
    """Query-conditioned LLM compiler from real document chunks to AttributeValueGraph.

    The extractor deliberately does not answer the user query. It asks the LLM
    for grounded records, validates those records, and then hands a normal graph
    to the deterministic router/solver/executor stack.
    """

    def __init__(
        self,
        config: LLMGraphExtractorConfig | None = None,
        *,
        max_chunks: int = 6,
        max_chars_per_chunk: int = 5000,
        max_numeric_candidates: int | None = 80,
        intent_conditioned_numeric_budget: bool = False,
        max_retries: int = 1,
        retry_expand_chunks: int = 4,
    ) -> None:
        self.config = config or LLMGraphExtractorConfig.from_env()
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_numeric_candidates = max_numeric_candidates
        self.intent_conditioned_numeric_budget = intent_conditioned_numeric_budget
        self.max_retries = max_retries
        self.retry_expand_chunks = retry_expand_chunks

    @property
    def available(self) -> bool:
        return self.config.available

    def extract(
        self,
        document_text: str,
        query: str,
        *,
        source_name: str = "document",
    ) -> LLMGraphExtractionResult:
        if not self.available:
            raise RuntimeError("LLM graph extraction is not configured.")
        if not document_text.strip():
            raise ValueError("Document text is empty.")
        if not query.strip():
            raise ValueError("Query is empty.")

        all_chunks = extract_document_chunks(
            document_text,
            source_name=source_name,
            max_chars_per_chunk=self.max_chars_per_chunk,
        )
        numeric_extraction = extract_numeric_evidence_chunks_with_diagnostics(all_chunks)
        all_numeric_chunks = numeric_extraction.chunks
        numeric_selection = _select_numeric_chunks_for_llm_calls(
            query,
            all_numeric_chunks,
            max_numeric_candidates=self.max_numeric_candidates,
            intent_conditioned_numeric_budget=self.intent_conditioned_numeric_budget,
        )
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        max_attempts = max(1, self.max_retries + 1)
        parsed: dict[str, Any] | None = None
        graph: AttributeValueGraph | None = None
        validation: dict[str, Any] | None = None
        numeric_chunks: list[NumericEvidenceChunk] = numeric_selection.chunks
        numeric_call_groups = _group_numeric_chunks_for_llm_calls(numeric_chunks)
        selected_source_chunks: list[DocumentChunk] = []
        extraction_drop_rows: list[dict[str, Any]] = []

        for attempt_index in range(max_attempts):
            selected_source_chunks = _source_chunks_for_numeric_evidence(all_chunks, numeric_chunks)
            try:
                parsed_payloads: list[dict[str, Any]] = []
                current_drop_rows: list[dict[str, Any]] = []
                for numeric_chunk_group in numeric_call_groups:
                    payload = self._chat_completion(
                        messages=[
                            {"role": "system", "content": self._system_prompt()},
                            {
                                "role": "user",
                                "content": self._user_prompt(
                                    source_name=source_name,
                                    query=query,
                                    source_chunks=_source_chunks_for_numeric_evidence(
                                        all_chunks,
                                        numeric_chunk_group,
                                    ),
                                    numeric_chunks=numeric_chunk_group,
                                    validation_errors=_retryable_error_messages(attempts),
                                ),
                            },
                        ]
                    )
                    content = self._extract_content(payload)
                    parsed_payload = self._parse_response_json(content)
                    parsed_payloads.append(parsed_payload)
                    drop_row = _numeric_group_drop_row(numeric_chunk_group, parsed_payload)
                    if drop_row is not None:
                        current_drop_rows.append(drop_row)
                parsed = _merge_numeric_chunk_payloads(
                    parsed_payloads,
                    source_name=source_name,
                )
                extraction_drop_rows = current_drop_rows
                if extraction_drop_rows:
                    parsed.setdefault("warnings", []).extend(
                        _extraction_drop_warning(row) for row in extraction_drop_rows
                    )
                graph, validation = build_graph_from_llm_payload(
                    parsed,
                    source_name=source_name,
                    query=query,
                )
                retryable = _should_retry_extraction(validation)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "retryable_validation_error" if retryable else "ok",
                        "source_chunk_count": len(selected_source_chunks),
                        "numeric_evidence_chunk_count": len(numeric_chunks),
                        "llm_call_count": len(numeric_call_groups),
                        "validation_errors": list(validation.get("validation_errors", [])),
                        "warnings": list(validation.get("warnings", [])),
                    }
                )
                if not retryable or attempt_index == max_attempts - 1:
                    break
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "error",
                        "source_chunk_count": len(selected_source_chunks),
                        "numeric_evidence_chunk_count": len(numeric_chunks),
                        "llm_call_count": len(numeric_call_groups),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if attempt_index == max_attempts - 1:
                    raise

        if graph is None or validation is None or parsed is None:
            if last_error is not None:
                raise last_error
            raise ValueError("LLM graph extraction did not produce a graph.")
        metadata = {
            "source": "llm",
            "parser": "llm_query_conditioned",
            "model": self.config.model,
            "confidence": _optional_float(parsed.get("confidence")),
            "warnings": list(parsed.get("warnings") or []) + validation["warnings"],
            "validation_errors": list(validation.get("validation_errors", [])),
            "chunks_used": [
                {
                    "chunk_id": chunk.chunk_id,
                    "kind": chunk.kind,
                    "table_index": chunk.table_index,
                    "caption": chunk.metadata.get("caption"),
                    "context": chunk.metadata.get("context"),
                    "statement_type": chunk.metadata.get("statement_type"),
                    "unit_context": chunk.metadata.get("unit_context"),
                }
                for chunk in selected_source_chunks
            ],
            "numeric_chunks_used": [
                _numeric_evidence_chunk_metadata(chunk)
                for chunk in numeric_chunks
            ],
            "numeric_evidence_chunk_count": len(numeric_chunks),
            "numeric_candidates_used": [
                _numeric_evidence_chunk_metadata(chunk)
                for chunk in numeric_chunks
            ],
            "numeric_candidate_count": len(numeric_chunks),
            "rejected_numeric_candidates": [
                _rejected_numeric_candidate_metadata(candidate)
                for candidate in numeric_extraction.rejected_candidates
            ],
            "numeric_candidate_filter_summary": {
                "kept": len(all_numeric_chunks),
                "rejected": len(numeric_extraction.rejected_candidates),
            },
            "numeric_retrieval": dict(numeric_selection.diagnostics),
            "retrieval_diagnostics": {
                "strategy": "one_llm_call_per_numeric_evidence_chunk",
                "available_chunks": len(all_numeric_chunks),
                "selected_chunk_ids": [chunk.chunk_id for chunk in numeric_chunks],
                "llm_call_count": len(numeric_call_groups),
                "numeric_selection": dict(numeric_selection.diagnostics),
            },
            "retry_count": max(0, len(attempts) - 1),
            "attempts": attempts,
            "record_count": len(graph.tokens),
            "source_coverage": validation["source_coverage"],
            "unit_coverage": validation["unit_coverage"],
            "field_coverage": validation.get("field_coverage", {}),
            "canonical_coverage": validation.get("canonical_coverage", {}),
            "candidate_completeness": validation.get("candidate_completeness", {}),
            "extraction_drop": _extraction_drop_summary(extraction_drop_rows),
        }
        return LLMGraphExtractionResult(graph=graph, metadata=metadata)

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": messages,
        }

        def call() -> dict[str, Any]:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            req = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:  # pragma: no cover - network path
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"LLM graph extraction failed with HTTP {exc.code}: {body}") from exc
            except error.URLError as exc:  # pragma: no cover - network path
                raise RuntimeError(f"LLM graph extraction failed: {exc.reason}") from exc

        return chat_completion_with_cache(
            cache_path=default_prompt_cache_path(),
            namespace=f"llm_graph_extraction:{url}",
            request_payload=payload,
            call=call,
        )

    def _system_prompt(self) -> str:
        concepts = concept_prompt_catalog()
        return (
            "You compile financial document facts into a structured numeric graph.\n"
            "Return JSON only. Do not answer the user's question.\n"
            "You must map every extracted fact into the closed canonical concept registry below.\n\n"
            "Canonical concept registry:\n"
            f"{concepts}\n\n"
            "The JSON schema is:\n"
            "{\n"
            '  "document_id": string,\n'
            '  "extraction_scope": "query_conditioned",\n'
            '  "unit_context": {"currency": string or null, "scale": string or null, "percent_is_ratio": boolean},\n'
            '  "records": [\n'
            "    {\n"
            '      "entity": string,\n'
            '      "entity_type": "company" | "product" | "segment" | "region" | string,\n'
            '      "canonical_concept_id": string,\n'
            '      "raw_label": string,\n'
            '      "field_name": string or null,\n'
            '      "field_label": string or null,\n'
            '      "dimensions": {\n'
            '        "entity_scope": "company" | "product" | "segment" | "region" | string,\n'
            '        "statement_type": "income_statement" | "balance_sheet" | "cash_flow" | "segment" | "product" | null,\n'
            '        "period_type": "fiscal_year" | "quarter" | "point_in_time" | null,\n'
            '        "fiscal_period": string or null,\n'
            '        "segment": string or null,\n'
            '        "product": string or null,\n'
            '        "region": string or null,\n'
            '        "attributable_to": string or null\n'
            "      },\n"
            '      "external_concept_id": string or null,\n'
            '      "year": integer or null,\n'
            '      "value": number,\n'
            '      "unit": string,\n'
            '      "source": {\n'
            '        "table_label": string or null,\n'
            '        "row_label": string or null,\n'
            '        "column_label": string or null,\n'
            '        "text_excerpt": string\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "warnings": [string],\n'
            '  "confidence": number\n'
            "}\n"
            "Rules:\n"
            "- Only include facts directly supported by the supplied numeric evidence chunks.\n"
            "- Use only numeric evidence chunks listed in the user message. Do not emit a record for an unlisted number.\n"
            "- Every listed numeric evidence chunk must produce exactly one independent record; never merge multiple input numbers into one record and never drop an input number.\n"
            "- Treat year-like candidates as time evidence unless the user explicitly asks for a year value.\n"
            f"- canonical_concept_id must be one of the listed IDs or {UNSUPPORTED_CONCEPT_ID}. Never invent IDs.\n"
            f"- If no listed concept applies, emit canonical_concept_id={UNSUPPORTED_CONCEPT_ID} and do not force it into a nearby concept.\n"
            "- Use raw_label for the exact row or source label; do not normalize raw_label.\n"
            "- Put qualifiers such as product, segment, region, period type, and attributable owner into dimensions, not into canonical_concept_id.\n"
            "- The deterministic executor derives field_name from canonical_concept_id. If you include field_name, it must match the default field for that canonical concept.\n"
            "- Extract facts needed for the current query plus enough peers for sums, counts, averages, and rankings.\n"
            "- For growth/change questions, extract both the starting year and ending year records for the same metric.\n"
            "- For company-level Revenue, Revenues, Total revenue, Sales, or Net sales, use canonical_concept_id revenue unless the user explicitly asks for total_net_sales.\n"
            "- If the user asks for revenue and the source row label is Total net sales or Net sales, canonical_concept_id MUST still be revenue.\n"
            "- For product, segment, or region sales, use canonical_concept_id revenue with entity_type and dimensions set to product/segment/region.\n"
            "- For questions shaped as A as a share/proportion/ratio of B, extract the raw numeric A and B records. Do not replace them with a precomputed percentage row.\n"
            "- For threshold count questions, extract every candidate record needed to evaluate the threshold, not just the final count.\n"
            "- Keep value as a pure number with no commas, currency symbols, or percent signs.\n"
            "- Emit explicit units such as million USD, percent, bps, or count.\n"
            "- Preserve per-share, shares, and basis-point units instead of converting them to ordinary currency.\n"
            "- For SEC 10-K/annual-report financial statement tables, dollar amounts are typically in millions; use million USD unless the supplied chunk explicitly says thousands, per-share, shares, or another scale.\n"
            "- Never infer thousand USD only because a number contains comma separators.\n"
            "- Skip Change and % columns unless the query directly asks for those columns.\n"
            "- Avoid mixing detail rows with total rows for the same aggregation.\n"
            "- Every record must include source text_excerpt copied from a numeric evidence chunk and row/column evidence when available."
        )

    def _user_prompt(
        self,
        *,
        source_name: str,
        query: str,
        source_chunks: list[DocumentChunk],
        numeric_chunks: list[NumericEvidenceChunk],
        validation_errors: list[str] | None = None,
    ) -> str:
        chunk_text = "\n".join(
            f"[{chunk.chunk_id} | kind={chunk.kind} | table_index={chunk.table_index} | "
            f"statement_type={chunk.metadata.get('statement_type')} | "
            f"unit_context={chunk.metadata.get('unit_context')}]"
            for chunk in source_chunks
        )
        numeric_chunk_text = _format_numeric_evidence_chunks(numeric_chunks)
        retry_note = ""
        if validation_errors:
            retry_note = (
                "\n\nPrevious extraction failed validation. Fix these issues without inventing data:\n"
                + "\n".join(f"- {item}" for item in validation_errors[:8])
            )
        return (
            f"Document id: {source_name}\n"
            f"User query: {query}\n\n"
            "Use only the numeric evidence chunks listed below to compile graph records. "
            "Each chunk was created deterministically by first extracting one numeric value and then clipping its local text. "
            "Every listed numeric evidence chunk must produce exactly one independent record. "
            "Do not merge records; the records array must cover every listed chunk. "
            "Select the chunks needed by the query, preserve the chunk text as source.text_excerpt, "
            "and do not calculate the final answer.\n\n"
            "Selected source chunk metadata:\n"
            f"{chunk_text}\n\n"
            "Numeric evidence chunks:\n"
            f"{numeric_chunk_text}"
            f"{retry_note}"
        )

    def _extract_content(self, payload: dict[str, Any]) -> str:
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

    def _parse_response_json(self, content: str) -> dict[str, Any]:
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


def build_graph_from_llm_payload(
    payload: dict[str, Any],
    *,
    source_name: str,
    query: str | None = None,
) -> tuple[AttributeValueGraph, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("LLM JSON payload must contain a records list.")

    context = payload.get("unit_context") if isinstance(payload.get("unit_context"), dict) else {}
    tokens: list[AttributeValueToken] = []
    warnings: list[str] = []
    seen: set[tuple[object, ...]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"LLM record {index} must be an object.")
        entity = _first_present(record, "entity", "company_name", "entity_name")
        concept_value = _first_present(record, "canonical_concept_id", "concept_id")
        legacy_field = _first_present(record, "field_name", "field")
        raw_label = _first_non_empty(
            [
                record.get("raw_label"),
                record.get("field_label"),
                record.get("label"),
                _source_label_from_record(record),
                legacy_field,
            ]
        )
        mapped_unsupported_text_metric = False
        if _is_unsupported_concept(concept_value):
            if _unsupported_record_is_query_relevant(record, raw_label, query):
                concept = get_canonical_concept("text_metric")
                concept_value = "text_metric"
                mapped_unsupported_text_metric = True
                warnings.append(f"unsupported_concept_mapped_to_text_metric:{raw_label or index}")
            else:
                warnings.append(f"unsupported_concept_skipped:{raw_label or index}")
                continue
        external_concept_ids = _external_concept_ids_from_record(record)
        if mapped_unsupported_text_metric:
            if concept is None:
                raise ValueError("text_metric canonical concept is not registered.")
        elif concept_value is not None:
            concept = get_canonical_concept(concept_value)
            if concept is None:
                raw_concept = _first_non_empty([concept_value]) or "<missing>"
                raise ValueError(f"LLM record {index} has unsupported canonical_concept_id: {raw_concept}")
        else:
            concept = resolve_canonical_concept(
                *external_concept_ids,
                legacy_field,
                raw_label,
            )
        value = _coerce_float(record.get("value"))
        source = _source_from_record(record, index=index, source_name=source_name)
        unit = _normalize_unit(str(record.get("unit") or ""), context)
        if not entity:
            raise ValueError(f"LLM record {index} is missing entity.")
        if concept is None:
            raw_concept = _first_non_empty([concept_value, legacy_field, raw_label]) or "<missing>"
            raise ValueError(f"LLM record {index} has unsupported canonical_concept_id: {raw_concept}")
        if _should_coerce_total_net_sales_to_revenue(concept.concept_id, query):
            warnings.append(f"canonical_concept_coerced:total_net_sales_to_revenue:{raw_label or index}")
            revenue_concept = get_canonical_concept("revenue")
            if revenue_concept is not None:
                concept = revenue_concept
        if value is None:
            raise ValueError(f"LLM record {index} has non-numeric value.")
        if not unit:
            raise ValueError(f"LLM record {index} is missing unit.")

        field_name = (
            _text_metric_field_name(record, raw_label)
            if concept.concept_id == "text_metric"
            else concept.default_field_name
        )
        year = _optional_int(record.get("year"))
        table = source.table or ""
        row_label = str(record.get("source", {}).get("row_label", "")) if isinstance(record.get("source"), dict) else ""
        entity_type = _first_non_empty(
            [
                record.get("entity_type"),
                _dimension_value(record, "entity_scope"),
                record.get("industry"),
                "company",
            ]
        )
        dimensions = _normalize_dimensions(record, source=source, entity_type=entity_type, year=year)
        if concept.concept_id == "text_metric":
            dimensions = {
                **(dimensions or {}),
                "concept_source": "unsupported_concept" if mapped_unsupported_text_metric else "text_metric",
            }
        external_ids = external_concept_ids or canonical_external_mappings(concept.concept_id)
        key = (
            normalize_identifier(str(entity)),
            field_name,
            concept.concept_id,
            year,
            table,
            row_label,
            source.column,
            float(value),
        )
        if key in seen:
            warnings.append(f"duplicate_record_skipped:{field_name}:{entity}:{year}")
            continue
        seen.add(key)

        entity_id = f"{normalize_identifier(str(entity))}:{year if year is not None else index}:{normalize_identifier(entity_type)}"
        tokens.append(
            AttributeValueToken(
                token_id=f"{entity_id}:{field_name}:{index}",
                entity_id=entity_id,
                company_name=str(entity),
                field_name=field_name,
                field_label=str(record.get("field_label") or concept.display_name),
                value=float(value),
                year=year,
                industry=entity_type,
                unit=unit,
                source=source,
                canonical_concept_id=concept.concept_id,
                dimensions=dimensions,
                raw_label=raw_label or None,
                external_concept_ids=external_ids,
            )
        )

    if not tokens:
        raise ValueError("LLM JSON payload did not produce any graph tokens.")
    graph = AttributeValueGraph(tuple(tokens), source_name=source_name)
    semantic_validation = _validate_llm_extraction(graph, query or "")
    warnings.extend(semantic_validation["warnings"])
    validation = {
        "warnings": warnings,
        "validation_errors": semantic_validation["validation_errors"],
        "source_coverage": _ratio(sum(1 for token in tokens if token.source is not None), len(tokens)),
        "unit_coverage": _ratio(sum(1 for token in tokens if token.unit), len(tokens)),
        "field_coverage": semantic_validation["field_coverage"],
        "canonical_coverage": semantic_validation["canonical_coverage"],
        "candidate_completeness": semantic_validation["candidate_completeness"],
    }
    return graph, validation


def _merge_numeric_chunk_payloads(
    payloads: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, Any]:
    records: list[Any] = []
    warnings: list[str] = []
    unit_context: dict[str, Any] = {}
    confidences: list[float] = []
    for index, payload in enumerate(payloads):
        payload_records = payload.get("records")
        if not isinstance(payload_records, list):
            raise ValueError(f"LLM JSON payload for numeric chunk {index} must contain a records list.")
        records.extend(payload_records)
        payload_warnings = payload.get("warnings")
        if isinstance(payload_warnings, list):
            warnings.extend(str(item) for item in payload_warnings if str(item).strip())
        context = payload.get("unit_context")
        if isinstance(context, dict):
            for key, value in context.items():
                if key not in unit_context and value is not None and str(value).strip():
                    unit_context[key] = value
        confidence = _optional_float(payload.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
    return {
        "document_id": source_name,
        "extraction_scope": "query_conditioned",
        "unit_context": unit_context,
        "records": records,
        "warnings": warnings,
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
    }


def extract_document_chunks(
    text: str,
    *,
    source_name: str,
    max_chars_per_chunk: int = 5000,
) -> list[DocumentChunk]:
    if "<table" in text.lower():
        table_chunks = _html_table_chunks(text, max_chars_per_chunk=max_chars_per_chunk)
        if table_chunks:
            return table_chunks
    return _text_chunks(text, source_name=source_name, max_chars_per_chunk=max_chars_per_chunk)


def select_relevant_chunks(
    query: str,
    chunks: list[DocumentChunk],
    *,
    max_chunks: int = 6,
) -> list[DocumentChunk]:
    return select_relevant_chunks_with_diagnostics(query, chunks, max_chunks=max_chunks).chunks


def select_relevant_chunks_with_diagnostics(
    query: str,
    chunks: list[DocumentChunk],
    *,
    max_chunks: int = 6,
) -> ChunkSelection:
    if len(chunks) <= max_chunks:
        return ChunkSelection(
            chunks=chunks,
            diagnostics={
                "strategy": "all_chunks",
                "available_chunks": len(chunks),
                "selected_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "scored_chunks": [_chunk_score_row(query, chunk, index) for index, chunk in enumerate(chunks)],
            },
        )
    term_weights = _query_term_weights(query)
    scored: list[tuple[float, int, DocumentChunk, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        row = _chunk_score_row(query, chunk, index, term_weights=term_weights)
        scored.append((float(row["score"]), -index, chunk, row))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [chunk for score, _, chunk, _ in scored[:max_chunks] if score > 0]
    if len(selected) < max_chunks:
        already = {chunk.chunk_id for chunk in selected}
        for _, _, chunk, _ in scored:
            if chunk.chunk_id not in already:
                selected.append(chunk)
            if len(selected) == max_chunks:
                break
    selected_ids = {chunk.chunk_id for chunk in selected}
    return ChunkSelection(
        chunks=selected,
        diagnostics={
            "strategy": "weighted_financial_table_retrieval",
            "available_chunks": len(chunks),
            "selected_chunk_ids": [chunk.chunk_id for chunk in selected],
            "top_scored_chunks": [
                {**row, "selected": chunk.chunk_id in selected_ids}
                for _, _, chunk, row in scored[: max(max_chunks * 2, 10)]
            ],
            "query_terms": sorted(term_weights),
        },
    )


def extract_numeric_evidence_chunks(
    chunks: list[DocumentChunk],
    *,
    radius: int = 160,
) -> list[NumericEvidenceChunk]:
    return extract_numeric_evidence_chunks_with_diagnostics(chunks, radius=radius).chunks


def extract_numeric_evidence_chunks_with_diagnostics(
    chunks: list[DocumentChunk],
    *,
    radius: int = 160,
) -> NumericEvidenceExtraction:
    numeric_chunks: list[NumericEvidenceChunk] = []
    rejected_candidates: list[RejectedNumericCandidate] = []
    order = 0
    for chunk in chunks:
        unit_context = chunk.metadata.get("unit_context")
        if not isinstance(unit_context, dict):
            unit_context = {}
        for match in NUMERIC_CANDIDATE_PATTERN.finditer(chunk.text):
            start, end = match.span("number")
            surface = _compact_whitespace(match.group("number"))
            value = _numeric_surface_value(surface)
            if value is None:
                continue
            unit_hint = _numeric_unit_hint(surface, unit_context)
            reject_reason = _numeric_evidence_reject_reason(
                chunk.text,
                start,
                end,
                surface,
                unit_hint=unit_hint,
            )
            if reject_reason is not None:
                context_start, context_end = _numeric_context_bounds(chunk.text, start, end, radius=radius)
                rejected_candidates.append(
                    RejectedNumericCandidate(
                        source_chunk_id=chunk.chunk_id,
                        source_kind=chunk.kind,
                        table_index=chunk.table_index,
                        text=_compact_whitespace(chunk.text[context_start:context_end]),
                        value_surface=surface,
                        normalized_value=value,
                        unit_hint=unit_hint,
                        reject_reason=reject_reason,
                        source_value_start=start,
                        source_value_end=end,
                        source_context_start=context_start,
                        source_context_end=context_end,
                    )
                )
                continue
            context_start, context_end = _numeric_context_bounds(chunk.text, start, end, radius=radius)
            raw_context = chunk.text[context_start:context_end]
            context = _compact_whitespace(raw_context)
            relative_start = max(0, start - context_start)
            relative_end = max(relative_start, end - context_start)
            prefix_compacted = _compact_whitespace(raw_context[:relative_start])
            surface_in_context = _compact_whitespace(raw_context[relative_start:relative_end])
            if prefix_compacted:
                text_value_start = len(prefix_compacted) + 1
            else:
                text_value_start = 0
            text_value_end = text_value_start + len(surface_in_context)
            numeric_chunks.append(
                NumericEvidenceChunk(
                    chunk_id=f"num_chunk_{order}",
                    source_chunk_id=chunk.chunk_id,
                    source_kind=chunk.kind,
                    table_index=chunk.table_index,
                    text=context,
                    value_surface=surface,
                    normalized_value=value,
                    unit_hint=unit_hint,
                    year_hint=_numeric_year_hint(chunk.text, start, end, context_start, context_end),
                    source_value_start=start,
                    source_value_end=end,
                    source_context_start=context_start,
                    source_context_end=context_end,
                    text_value_start=text_value_start,
                    text_value_end=text_value_end,
                )
            )
            order += 1
    return NumericEvidenceExtraction(
        chunks=numeric_chunks,
        rejected_candidates=rejected_candidates,
    )


def select_relevant_numeric_evidence_chunks(
    query: str,
    chunks: list[NumericEvidenceChunk],
    *,
    max_chunks: int = 80,
) -> NumericEvidenceSelection:
    if len(chunks) <= max_chunks:
        return NumericEvidenceSelection(
            chunks=chunks,
            diagnostics={
                "strategy": "all_numeric_evidence_chunks",
                "available_chunks": len(chunks),
                "selected_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "scored_chunks": [
                    _numeric_evidence_score_row(query, chunk, index)
                    for index, chunk in enumerate(chunks)
                ],
            },
        )
    scored: list[tuple[float, int, NumericEvidenceChunk, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        row = _numeric_evidence_score_row(query, chunk, index)
        scored.append((float(row["score"]), -index, chunk, row))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [chunk for score, _, chunk, _ in scored[:max_chunks] if score > 0]
    if len(selected) < max_chunks:
        already = {chunk.chunk_id for chunk in selected}
        for _, _, chunk, _ in scored:
            if chunk.chunk_id not in already:
                selected.append(chunk)
            if len(selected) == max_chunks:
                break
    selected_ids = {chunk.chunk_id for chunk in selected}
    return NumericEvidenceSelection(
        chunks=selected,
        diagnostics={
            "strategy": "weighted_numeric_evidence_retrieval",
            "available_chunks": len(chunks),
            "selected_chunk_ids": [chunk.chunk_id for chunk in selected],
            "top_scored_chunks": [
                {**row, "selected": chunk.chunk_id in selected_ids}
                for _, _, chunk, row in scored[: max(max_chunks * 2, 10)]
            ],
            "query_terms": sorted(_query_term_weights(query)),
        },
    )


def _select_numeric_chunks_for_llm_calls(
    query: str,
    chunks: list[NumericEvidenceChunk],
    *,
    max_numeric_candidates: int | None,
    intent_conditioned_numeric_budget: bool = False,
) -> NumericEvidenceSelection:
    if not intent_conditioned_numeric_budget and (
        max_numeric_candidates is None or max_numeric_candidates <= 0
    ):
        return NumericEvidenceSelection(
            chunks=chunks,
            diagnostics={
                "strategy": "all_numeric_evidence_chunks_unlimited",
                "budget_mode": "fixed",
                "intent_operator": None,
                "seed_budget": None,
                "hard_limit": None,
                "available_chunks": len(chunks),
                "selected_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "expanded_context_sibling_count": 0,
                "hard_limit_truncated_count": 0,
            },
        )
    intent_operator: str | None = None
    if intent_conditioned_numeric_budget:
        intent_operator, policy_budget = numeric_candidate_seed_budget(query)
        hard_limit = (
            max_numeric_candidates
            if max_numeric_candidates is not None and max_numeric_candidates > 0
            else 16
        )
        seed_budget = min(policy_budget, hard_limit)
    else:
        hard_limit = None
        seed_budget = int(max_numeric_candidates or 0)
    seed_selection = select_relevant_numeric_evidence_chunks(
        query,
        chunks,
        max_chunks=seed_budget,
    )
    diagnostics = dict(seed_selection.diagnostics)
    diagnostics.update(
        {
            "budget_mode": (
                "intent_conditioned" if intent_conditioned_numeric_budget else "fixed"
            ),
            "intent_operator": intent_operator,
            "seed_budget": seed_budget,
            "hard_limit": hard_limit,
        }
    )
    seed_selection = NumericEvidenceSelection(
        chunks=seed_selection.chunks,
        diagnostics=diagnostics,
    )
    return _expand_numeric_selection_to_sibling_contexts(
        chunks,
        seed_selection,
        hard_limit=hard_limit,
        prioritize_selected_contexts=intent_conditioned_numeric_budget,
    )


def numeric_candidate_seed_budget(query: str) -> tuple[str, int]:
    normalized = normalize_identifier(query)
    if any(term in normalized for term in ("highest", "largest", "maximum", "max")):
        return "MAX", 12
    if any(term in normalized for term in ("lowest", "smallest", "minimum", "min")):
        return "MIN", 12
    if "how_many" in normalized or "number_of" in normalized:
        return "COUNT", 12
    intent = classify_question_intent(query)
    operator = intent.preferred_operators[0]
    if operator in {"LOOKUP", "BOOLEAN", "TEXT_SPAN", "YEAR_LIST"}:
        return operator, 3
    if operator == "DIFFERENCE":
        return operator, 10
    if operator in {"PERCENT_CHANGE", "RATIO", "SHARE", "MARGIN"}:
        return operator, 6
    if operator in {"SUM", "AVG", "MAX", "MIN", "COUNT"}:
        return operator, 12
    return operator, 4


def _expand_numeric_selection_to_sibling_contexts(
    all_chunks: list[NumericEvidenceChunk],
    selection: NumericEvidenceSelection,
    *,
    hard_limit: int | None = None,
    prioritize_selected_contexts: bool = False,
) -> NumericEvidenceSelection:
    if not selection.chunks:
        return selection
    sibling_ids = {
        chunk.chunk_id
        for seed in selection.chunks
        for chunk in all_chunks
        if _numeric_chunks_are_siblings(seed, chunk)
    }
    if prioritize_selected_contexts:
        expanded = list(selection.chunks)
        seen_ids: set[str] = {chunk.chunk_id for chunk in expanded}
        seed_rank = {chunk.chunk_id: index for index, chunk in enumerate(selection.chunks)}
        sibling_candidates = []
        for chunk in all_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            matching_ranks = [
                seed_rank[seed.chunk_id]
                for seed in selection.chunks
                if _numeric_chunks_are_siblings(seed, chunk)
            ]
            if matching_ranks:
                sibling_candidates.append((min(matching_ranks), chunk))
        sibling_candidates.sort(
            key=lambda item: (
                0 if _surface_has_explicit_metric_unit(item[1].value_surface) else 1,
                item[0],
                item[1].source_value_start,
                item[1].chunk_id,
            )
        )
        expanded.extend(chunk for _, chunk in sibling_candidates)
    else:
        expanded = [chunk for chunk in all_chunks if chunk.chunk_id in sibling_ids]
    expanded_count = max(0, len(expanded) - len(selection.chunks))
    truncated_count = 0
    if hard_limit is not None and hard_limit > 0 and len(expanded) > hard_limit:
        truncated_count = len(expanded) - hard_limit
        expanded = expanded[:hard_limit]
    diagnostics = dict(selection.diagnostics)
    diagnostics["seed_selected_chunk_ids"] = [chunk.chunk_id for chunk in selection.chunks]
    diagnostics["selected_chunk_ids"] = [chunk.chunk_id for chunk in expanded]
    diagnostics["expanded_context_sibling_count"] = expanded_count
    diagnostics["hard_limit_truncated_count"] = truncated_count
    return NumericEvidenceSelection(chunks=expanded, diagnostics=diagnostics)


def _group_numeric_chunks_for_llm_calls(
    chunks: list[NumericEvidenceChunk],
) -> list[list[NumericEvidenceChunk]]:
    groups: list[list[NumericEvidenceChunk]] = []
    for chunk in chunks:
        matching = [
            index
            for index, group in enumerate(groups)
            if any(_numeric_chunks_are_siblings(chunk, existing) for existing in group)
        ]
        if not matching:
            groups.append([chunk])
            continue
        first = matching[0]
        groups[first].append(chunk)
        for index in reversed(matching[1:]):
            groups[first].extend(groups.pop(index))
    return groups


def _numeric_context_key(chunk: NumericEvidenceChunk) -> tuple[str, str]:
    return (chunk.source_chunk_id, chunk.text)


def _numeric_chunks_are_siblings(
    left: NumericEvidenceChunk,
    right: NumericEvidenceChunk,
) -> bool:
    if left.source_chunk_id != right.source_chunk_id:
        return False
    if left.text == right.text:
        return True
    return max(left.source_context_start, right.source_context_start) < min(
        left.source_context_end,
        right.source_context_end,
    )


def _numeric_group_drop_row(
    numeric_chunks: list[NumericEvidenceChunk],
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    records = payload.get("records")
    output_count = len(records) if isinstance(records, list) else 0
    input_count = len(numeric_chunks)
    if output_count >= input_count:
        return None
    return {
        "input_numeric_count": input_count,
        "output_record_count": output_count,
        "dropped_record_count": input_count - output_count,
        "input_chunk_ids": [chunk.chunk_id for chunk in numeric_chunks],
        "input_values": [chunk.value_surface for chunk in numeric_chunks],
    }


def _extraction_drop_warning(row: Mapping[str, Any]) -> str:
    return (
        "extraction_drop:"
        f"input={row.get('input_numeric_count')}:"
        f"output={row.get('output_record_count')}:"
        f"chunks={','.join(str(item) for item in row.get('input_chunk_ids') or [])}:"
        f"values={','.join(str(item) for item in row.get('input_values') or [])}"
    )


def _extraction_drop_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "triggered": bool(rows),
        "call_count": len(rows),
        "dropped_record_count": sum(int(row["dropped_record_count"]) for row in rows),
        "rows": [dict(row) for row in rows],
    }


def extract_numeric_candidate_windows(
    query: str,
    chunks: list[DocumentChunk],
    *,
    radius: int = 160,
    max_candidates: int | None = 80,
) -> list[NumericEvidenceChunk]:
    numeric_chunks = extract_numeric_evidence_chunks(chunks, radius=radius)
    if max_candidates is None:
        return numeric_chunks
    return select_relevant_numeric_evidence_chunks(
        query,
        numeric_chunks,
        max_chunks=max_candidates,
    ).chunks


def _format_numeric_evidence_chunks(chunks: list[NumericEvidenceChunk]) -> str:
    if not chunks:
        return "(no numeric evidence chunks found)"
    lines: list[str] = []
    for chunk in chunks:
        normalized = (
            str(int(chunk.normalized_value))
            if chunk.normalized_value is not None and chunk.normalized_value.is_integer()
            else str(chunk.normalized_value)
        )
        lines.append(
            f"[{chunk.chunk_id} | source_chunk={chunk.source_chunk_id} | kind={chunk.source_kind} | "
            f"table_index={chunk.table_index} | value={chunk.value_surface} | "
            f"normalized_value={normalized} | unit_hint={chunk.unit_hint or 'unknown'} | "
            f"year_hint={chunk.year_hint if chunk.year_hint is not None else 'unknown'} | "
            f"source_chars={chunk.source_value_start}:{chunk.source_value_end} | "
            f"chunk_value_chars={chunk.text_value_start}:{chunk.text_value_end}]\n"
            f"chunk_text: {chunk.text}"
        )
    return "\n\n".join(lines)


def _format_numeric_candidate_windows(candidates: list[NumericEvidenceChunk]) -> str:
    return _format_numeric_evidence_chunks(candidates)


def _numeric_evidence_chunk_metadata(chunk: NumericEvidenceChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_chunk_id": chunk.source_chunk_id,
        "source_kind": chunk.source_kind,
        "table_index": chunk.table_index,
        "value_surface": chunk.value_surface,
        "normalized_value": chunk.normalized_value,
        "unit_hint": chunk.unit_hint,
        "year_hint": chunk.year_hint,
        "source_value_start": chunk.source_value_start,
        "source_value_end": chunk.source_value_end,
        "source_context_start": chunk.source_context_start,
        "source_context_end": chunk.source_context_end,
        "text_value_start": chunk.text_value_start,
        "text_value_end": chunk.text_value_end,
        "text": chunk.text,
    }


def _rejected_numeric_candidate_metadata(candidate: RejectedNumericCandidate) -> dict[str, Any]:
    return {
        "source_chunk_id": candidate.source_chunk_id,
        "source_kind": candidate.source_kind,
        "table_index": candidate.table_index,
        "value_surface": candidate.value_surface,
        "normalized_value": candidate.normalized_value,
        "unit_hint": candidate.unit_hint,
        "reject_reason": candidate.reject_reason,
        "source_value_start": candidate.source_value_start,
        "source_value_end": candidate.source_value_end,
        "source_context_start": candidate.source_context_start,
        "source_context_end": candidate.source_context_end,
        "text": candidate.text,
    }


def _source_chunks_for_numeric_evidence(
    source_chunks: list[DocumentChunk],
    numeric_chunks: list[NumericEvidenceChunk],
) -> list[DocumentChunk]:
    by_id = {chunk.chunk_id: chunk for chunk in source_chunks}
    selected: list[DocumentChunk] = []
    seen: set[str] = set()
    for numeric_chunk in numeric_chunks:
        if numeric_chunk.source_chunk_id in seen:
            continue
        source_chunk = by_id.get(numeric_chunk.source_chunk_id)
        if source_chunk is None:
            continue
        selected.append(source_chunk)
        seen.add(numeric_chunk.source_chunk_id)
    return selected


def _numeric_context_bounds(text: str, start: int, end: int, *, radius: int) -> tuple[int, int]:
    window_start = max(0, start - radius)
    window_end = min(len(text), end + radius)
    sentence_start = max(text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n", ";"))
    if sentence_start >= window_start:
        window_start = sentence_start + 1
    sentence_ends = [text.find(mark, end) for mark in (".", "!", "?", "\n", ";")]
    sentence_ends = [position for position in sentence_ends if position != -1]
    if sentence_ends:
        nearest_end = min(sentence_ends) + 1
        if nearest_end <= window_end:
            window_end = nearest_end
    while window_start > 0 and not text[window_start - 1].isspace():
        window_start -= 1
    while window_end < len(text) and not text[window_end].isspace():
        window_end += 1
    return window_start, window_end


def _numeric_surface_value(surface: str) -> float | None:
    match = re.search(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?", surface)
    if match is None:
        return None
    value = _coerce_float(match.group(0))
    if value is None:
        return None
    stripped = surface.strip()
    if stripped.startswith("(") and ")" in stripped:
        return -value
    return value


def _numeric_unit_hint(surface: str, unit_context: dict[str, Any]) -> str | None:
    stripped = surface.strip()
    if _surface_is_year(stripped):
        return "year"
    lowered = stripped.casefold()
    if "%" in stripped or "percent" in lowered:
        return "percent"
    if "basis point" in lowered or re.search(r"\bbps\b", lowered):
        return "bps"
    inferred = _normalize_unit(stripped, unit_context)
    if inferred == stripped:
        context_unit = _normalize_unit("", unit_context)
        return context_unit or None
    return inferred or None


def _should_skip_numeric_evidence_candidate(
    text: str,
    start: int,
    end: int,
    surface: str,
    *,
    unit_hint: str | None,
) -> bool:
    return _numeric_evidence_reject_reason(
        text,
        start,
        end,
        surface,
        unit_hint=unit_hint,
    ) is not None


def _numeric_evidence_reject_reason(
    text: str,
    start: int,
    end: int,
    surface: str,
    *,
    unit_hint: str | None,
) -> str | None:
    if _surface_has_explicit_metric_unit(surface):
        return None
    if _surface_is_year(surface):
        return "standalone_year"
    if _looks_like_slash_date_fragment(text, start, end, surface):
        return "slash_date_fragment"
    if _looks_like_date_day(text, start, end, surface):
        return "date_day"
    if _looks_like_reference_number(text, start, end, surface):
        return "reference_number"
    if unit_hint is None and re.fullmatch(r"\(\d{1,2}\)", surface.strip()):
        return "footnote_marker"
    return None


def _surface_has_explicit_metric_unit(surface: str) -> bool:
    lowered = surface.casefold()
    return bool(
        re.search(
            r"[$€£%]|\b(?:percent(?:age)?|bps|basis points?|million|millions|billion|billions|"
            r"thousand|thousands|usd|u\.s\. dollars?|dollars?|shares?)\b",
            lowered,
        )
    )


def _looks_like_date_day(text: str, start: int, end: int, surface: str) -> bool:
    stripped = surface.strip("() ")
    if not re.fullmatch(r"\d{1,2}", stripped):
        return False
    day = int(stripped)
    if not 1 <= day <= 31:
        return False
    before = text[max(0, start - 32) : start]
    after = text[end : min(len(text), end + 32)]
    if _month_name_ends_context(before):
        return True
    return _month_name_starts_context(after)


def _looks_like_slash_date_fragment(text: str, start: int, end: int, surface: str) -> bool:
    if not re.fullmatch(r"\d{1,2}", surface.strip("() ")):
        return False
    window_start = max(0, start - 32)
    window_end = min(len(text), end + 32)
    window = text[window_start:window_end]
    matches = list(re.finditer(r"(?<!\d)\d{1,2}/\d{1,2}(?:/\d{1,4})?(?!\d)", window))
    for match in matches:
        absolute_start = window_start + match.start()
        absolute_end = window_start + match.end()
        if not (absolute_start <= start and end <= absolute_end):
            continue
        if match.group(0).count("/") >= 2 or len(matches) >= 2:
            return True
        if re.search(r"\b(?:date|dated|chart|column|as of|on)\b", window, re.IGNORECASE):
            return True
    return False


def _month_name_ends_context(text: str) -> bool:
    stripped = text.rstrip(" ,-/")
    match = MONTH_NAME_PATTERN.search(stripped)
    return match is not None and match.end() == len(stripped)


def _month_name_starts_context(text: str) -> bool:
    return MONTH_NAME_PATTERN.match(text.lstrip(" ,-/")) is not None


def _looks_like_reference_number(text: str, start: int, end: int, surface: str) -> bool:
    stripped = surface.strip("() ")
    if not re.fullmatch(r"\d{1,3}", stripped):
        return False
    before = text[max(0, start - 28) : start]
    after = text[end : min(len(text), end + 16)]
    if REFERENCE_NUMBER_PREFIX_PATTERN.search(before):
        return True
    if re.match(r"^\s*(?:of\s+\d{1,3}\b|[-–—])", after):
        return True
    return False


def _numeric_year_hint(
    text: str,
    start: int,
    end: int,
    context_start: int,
    context_end: int,
) -> int | None:
    surface = text[start:end]
    if _surface_is_year(surface):
        return int(_numeric_surface_value(surface) or 0)
    context = text[context_start:context_end]
    center = (start + end) / 2
    closest: tuple[float, int] | None = None
    for match in re.finditer(r"\b(?:19|20)\d{2}\b", context):
        year = int(match.group(0))
        absolute_center = context_start + (match.start() + match.end()) / 2
        distance = abs(absolute_center - center)
        if closest is None or distance < closest[0]:
            closest = (distance, year)
    return closest[1] if closest else None


def _surface_is_year(surface: str) -> bool:
    stripped = surface.strip()
    return bool(re.fullmatch(r"(?:19|20)\d{2}", stripped))


def _numeric_evidence_score_row(
    query: str,
    chunk: NumericEvidenceChunk,
    index: int,
) -> dict[str, Any]:
    term_weights = _query_term_weights(query)
    lowered_context = chunk.text.casefold()
    matched_terms = {term: 1 for term in term_weights if term in lowered_context}
    lexical_score = sum(count * term_weights[term] for term, count in matched_terms.items())
    unit_bonus = 0.0
    query_lower = query.casefold()
    if chunk.year_hint is not None and str(chunk.year_hint) in query:
        unit_bonus += 1.5
    if chunk.unit_hint in {"percent", "bps"} and any(
        term in query_lower
        for term in ("percent", "%", "bps", "比例", "占比")
    ):
        unit_bonus += 1.0
    if chunk.unit_hint and "USD" in chunk.unit_hint and any(
        term in query_lower
        for term in ("cash", "revenue", "sales", "income", "资产", "收入", "现金")
    ):
        unit_bonus += 0.5
    score = lexical_score + unit_bonus + _financial_chunk_bonus(query, lowered_context)
    return {
        "chunk_id": chunk.chunk_id,
        "source_chunk_id": chunk.source_chunk_id,
        "source_kind": chunk.source_kind,
        "table_index": chunk.table_index,
        "score": round(float(score), 4),
        "lexical_score": round(float(lexical_score), 4),
        "unit_bonus": round(float(unit_bonus), 4),
        "matched_terms": matched_terms,
        "value_surface": chunk.value_surface,
        "normalized_value": chunk.normalized_value,
        "unit_hint": chunk.unit_hint,
        "year_hint": chunk.year_hint,
        "rank_source_index": index,
    }


def _numeric_candidate_score(query: str, candidate: NumericEvidenceChunk) -> float:
    return float(_numeric_evidence_score_row(query, candidate, 0)["score"])


def _compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _html_table_chunks(text: str, *, max_chars_per_chunk: int) -> list[DocumentChunk]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover - bs4 is available in the project env
        return []
    soup = BeautifulSoup(text, "lxml")
    chunks: list[DocumentChunk] = []
    for table_index, table in enumerate(soup.find_all("table")):
        caption = ""
        caption_tag = table.find("caption")
        if caption_tag is not None:
            caption = caption_tag.get_text(" ", strip=True)
        context = _nearby_table_context(table)
        rows: list[str] = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        if not rows:
            continue
        statement_type = _infer_statement_type(" ".join([caption, context, *rows[:5]]))
        unit_context = infer_table_unit_context(" ".join([caption, context, *rows[:5]]))
        header_lines = []
        if context:
            header_lines.append(f"Context: {context}")
        if caption:
            header_lines.append(f"Caption: {caption}")
        if unit_context:
            header_lines.append(f"Unit context: {unit_context}")
        table_text = "\n".join([*header_lines, *rows])
        chunks.append(
            DocumentChunk(
                chunk_id=f"table_{table_index}",
                kind="table",
                table_index=table_index,
                text=table_text[:max_chars_per_chunk],
                metadata={
                    "caption": caption or None,
                    "context": context or None,
                    "statement_type": statement_type,
                    "unit_context": unit_context,
                },
            )
        )
    return chunks


def _text_chunks(text: str, *, source_name: str, max_chars_per_chunk: int) -> list[DocumentChunk]:
    cleaned = text.strip()
    if not cleaned:
        return []
    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(start + max_chars_per_chunk, len(cleaned))
        chunks.append(
            DocumentChunk(
                chunk_id=f"text_{index}",
                kind="text",
                text=cleaned[start:end],
                table_index=None,
                metadata={"unit_context": infer_table_unit_context(cleaned[start:end])},
            )
        )
        start = end
        index += 1
    return chunks or [DocumentChunk(chunk_id="text_0", kind="text", text=source_name)]


def _source_from_record(record: dict[str, Any], *, index: int, source_name: str) -> TokenSource:
    source = record.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"LLM record {index} is missing source evidence.")
    excerpt = _first_non_empty(
        [
            source.get("text_excerpt"),
            source.get("evidence"),
            record.get("evidence"),
        ]
    )
    table = _first_non_empty([source.get("table"), source.get("table_label")])
    row_label = _first_non_empty([source.get("row"), source.get("row_index"), source.get("row_label")])
    column = _first_non_empty([source.get("column"), source.get("column_label")])
    if not excerpt:
        raise ValueError(f"LLM record {index} source evidence is missing text_excerpt.")
    if not (table or row_label or column):
        table = "llm_evidence"
    return TokenSource(
        document_id=source_name,
        table=table or None,
        row=_optional_int(row_label),
        column=column or None,
        text_excerpt=excerpt,
    )


def _source_label_from_record(record: dict[str, Any]) -> str:
    source = record.get("source")
    if not isinstance(source, dict):
        return ""
    return _first_non_empty([source.get("row_label"), source.get("row"), source.get("label")])


def _is_unsupported_concept(value: object) -> bool:
    return value is not None and normalize_identifier(str(value)) == UNSUPPORTED_CONCEPT_ID


def _unsupported_record_is_query_relevant(
    record: dict[str, Any],
    raw_label: str | None,
    query: str | None,
) -> bool:
    if not query:
        return False
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    evidence = " ".join(
        str(value)
        for value in (
            raw_label,
            record.get("field_label"),
            _source_label_from_record(record),
            source.get("text_excerpt"),
            source.get("evidence"),
        )
        if value
    )
    query_terms = _text_metric_terms(query)
    evidence_terms = _text_metric_terms(evidence)
    if len(query_terms & evidence_terms) >= 2:
        return True
    normalized_label = normalize_identifier(raw_label or "")
    normalized_query = normalize_identifier(query)
    return bool(normalized_label and normalized_label in normalized_query)


def _text_metric_terms(value: object) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "between",
        "by",
        "change",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "total",
        "was",
        "were",
        "what",
        "which",
    }
    return {
        term
        for term in normalize_identifier(str(value)).split("_")
        if len(term) > 1 and not term.isdigit() and term not in stopwords
    }


def _text_metric_field_name(record: dict[str, Any], raw_label: str | None) -> str:
    field = _first_non_empty(
        [
            raw_label,
            record.get("field_label"),
            _source_label_from_record(record),
            record.get("field_name"),
            "text_metric",
        ]
    )
    normalized = normalize_identifier(str(field))
    return normalized or "text_metric"


def _should_coerce_total_net_sales_to_revenue(concept_id: str, query: str | None) -> bool:
    if concept_id != "total_net_sales":
        return False
    query_norm = normalize_identifier(query or "")
    if "total_net_sales" in query_norm or "totalnetsales" in query_norm.replace("_", ""):
        return False
    return "revenue" in query_norm or any(term in query_norm for term in ("营收", "收入", "营业收入"))


def _external_concept_ids_from_record(record: dict[str, Any]) -> tuple[str, ...]:
    raw_values: list[object] = []
    for key in ("external_concept_id", "xbrl_tag", "taxonomy_concept"):
        value = record.get(key)
        if value is not None:
            raw_values.append(value)
    many = record.get("external_concept_ids")
    if isinstance(many, (list, tuple)):
        raw_values.extend(many)
    elif many is not None:
        raw_values.append(many)
    values = tuple(str(value).strip() for value in raw_values if str(value).strip())
    return tuple(dict.fromkeys(values))


def _dimension_value(record: dict[str, Any], key: str) -> object | None:
    dimensions = record.get("dimensions")
    if isinstance(dimensions, dict):
        value = dimensions.get(key)
        if value is not None and str(value).strip():
            return value
    value = record.get(key)
    if value is not None and str(value).strip():
        return value
    return None


def _normalize_dimensions(
    record: dict[str, Any],
    *,
    source: TokenSource,
    entity_type: str,
    year: int | None,
) -> dict[str, object]:
    dimensions: dict[str, object] = {}
    raw_dimensions = record.get("dimensions")
    if isinstance(raw_dimensions, dict):
        for key, value in raw_dimensions.items():
            normalized_key = normalize_identifier(str(key))
            if normalized_key in CANONICAL_DIMENSION_KEYS and value is not None and str(value).strip():
                dimensions[normalized_key] = value
    for key in CANONICAL_DIMENSION_KEYS:
        value = record.get(key)
        if key not in dimensions and value is not None and str(value).strip():
            dimensions[key] = value
    if "entity_scope" not in dimensions:
        dimensions["entity_scope"] = normalize_identifier(entity_type) or "company"
    if "statement_type" not in dimensions:
        statement_type = _infer_statement_type(
            " ".join(str(item or "") for item in (source.table, source.text_excerpt))
        )
        if statement_type:
            dimensions["statement_type"] = statement_type
    if "period_type" not in dimensions and year is not None:
        dimensions["period_type"] = "fiscal_year"
    return dimensions


def _normalize_unit(unit: str, context: dict[str, Any]) -> str:
    raw = unit.strip()
    context_scale = str(context.get("scale") or "").strip()
    context_currency = str(context.get("currency") or "").strip().upper()
    lowered = raw.casefold()
    if not raw and context_scale and context_currency:
        return f"{_singular_scale(context_scale)} {context_currency}"
    if not raw:
        return ""
    if lowered in {"%", "percent", "percentage"}:
        return "percent"
    if "basis point" in lowered or lowered == "bps":
        return "bps"
    if lowered in {"count", "counts", "number"}:
        return "count"
    if "per share" in lowered or "per-share" in lowered:
        currency = _detect_currency(raw) or context_currency
        return f"{currency} per share" if currency else "per share"
    if re.search(r"\bshares?\b", lowered):
        scale = _detect_scale(raw) or context_scale
        return f"{_singular_scale(scale)} shares" if scale else "shares"

    currency = _detect_currency(raw) or context_currency
    scale = _detect_scale(raw) or context_scale
    if currency and scale:
        return f"{_singular_scale(scale)} {currency}"
    if currency:
        return currency
    if scale:
        return _singular_scale(scale)
    return raw


def _detect_currency(text: str) -> str | None:
    lowered = text.casefold()
    if re.search(r"\busd\b|us\$|\$|dollars?|美元", lowered):
        return "USD"
    if re.search(r"\beur\b|€|euros?|欧元", lowered):
        return "EUR"
    if re.search(r"\bcny\b|\brmb\b|¥|￥|yuan|人民币|元", lowered):
        return "CNY"
    if re.search(r"\bgbp\b|£|pounds?", lowered):
        return "GBP"
    if re.search(r"\bjpy\b|yen|日元", lowered):
        return "JPY"
    return None


def _detect_scale(text: str) -> str | None:
    lowered = text.casefold()
    for pattern, label in (
        (r"\bbillions?\b|\bbn\b", "billion"),
        (r"\bmillions?\b|\bmn\b", "million"),
        (r"\bthousands?\b", "thousand"),
    ):
        if re.search(pattern, lowered):
            return label
    return None


def _singular_scale(scale: str) -> str:
    lowered = scale.casefold()
    if lowered.startswith("million") or lowered == "mn":
        return "million"
    if lowered.startswith("billion") or lowered == "bn":
        return "billion"
    if lowered.startswith("thousand"):
        return "thousand"
    return scale.strip().rstrip("s")


def _query_terms(query: str) -> set[str]:
    terms = {
        item.casefold()
        for item in re.findall(r"\d{4}|[A-Za-z_][A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query)
        if item.casefold() not in QUERY_RETRIEVAL_STOPWORDS
    }
    for item in re.split(r"[_\s]+", query):
        lowered = item.casefold()
        if len(item) >= 3 and lowered not in QUERY_RETRIEVAL_STOPWORDS:
            terms.add(lowered)
    terms.update(_field_alias_terms(terms))
    return terms


def _query_term_weights(query: str) -> dict[str, float]:
    terms = _query_terms(query)
    weights = {
        term: 1.0 if re.fullmatch(r"(?:19|20)\d{2}", term) else 3.0
        for term in terms
    }
    financial_terms = {
        "net sales",
        "total net sales",
        "revenue",
        "revenues",
        "sales",
        "operating income",
        "operating income/(loss)",
        "gross margin",
        "gross profit",
        "net income",
        "research and development",
        "cash and cash equivalents",
        "net cash provided by operating activities",
        "operating cash flow",
        "total assets",
        "total liabilities",
        "net income",
        "diluted earnings per share",
        "basic earnings per share",
    }
    for term in financial_terms:
        if term in terms:
            weights[term] = 6.0
    for term in list(weights):
        if re.fullmatch(r"20\d{2}", term):
            weights[term] = 0.7
        elif term in {"amazon", "microsoft", "nvidia", "meta", "apple"}:
            weights[term] = 0.5
    return weights


def _financial_chunk_bonus(query: str, lowered_chunk: str) -> int:
    query_lower = query.casefold()
    bonus = 0
    if any(term in query_lower for term in ("revenue", "sales", "收入", "营收")):
        if "net sales" in lowered_chunk or "total net sales" in lowered_chunk:
            bonus += 16
        if "sales revenue" in lowered_chunk or "revenues" in lowered_chunk:
            bonus += 10
    if "operating_income" in query_lower or "operating income" in query_lower or "经营" in query_lower:
        if "operating income" in lowered_chunk or "operating income/(loss)" in lowered_chunk:
            bonus += 14
    if "gross_margin" in query_lower or "gross margin" in query_lower or "毛利" in query_lower:
        if "gross margin" in lowered_chunk or "gross profit" in lowered_chunk:
            bonus += 14
    if "net_income" in query_lower or "net income" in query_lower or "净利润" in query_lower:
        if "net income" in lowered_chunk or "net earnings" in lowered_chunk:
            bonus += 12
    if "total_assets" in query_lower or "assets" in query_lower or "资产" in query_lower:
        if "total assets" in lowered_chunk:
            bonus += 14
        if "consolidated balance sheets" in lowered_chunk:
            bonus += 6
    if "cash" in query_lower or "现金" in query_lower:
        if "cash and cash equivalents" in lowered_chunk:
            bonus += 12
        if "cash flows" in lowered_chunk:
            bonus += 6
    if "liabilities" in query_lower or "负债" in query_lower:
        if "total liabilities" in lowered_chunk:
            bonus += 12
    if "exhibit number" in lowered_chunk or "certification" in lowered_chunk:
        bonus -= 12
    if "signature" in lowered_chunk or "power of attorney" in lowered_chunk:
        bonus -= 8
    return bonus


def _field_alias_terms(terms: set[str]) -> set[str]:
    joined = " ".join(sorted(terms))
    aliases: set[str] = set()
    field_aliases: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("total_net_sales", "net_sales", "sales", "收入", "营收"), ("net sales", "total net sales")),
        (("gross_margin", "毛利"), ("gross margin",)),
        (("operating_income", "经营利润", "营业利润"), ("operating income", "operating income/(loss)")),
        (("net_income", "净利润"), ("net income",)),
        (("net_margin", "净利率"), ("net income", "net margin")),
        (("research_and_development", "研发"), ("research and development",)),
        (("cash_and_cash_equivalents", "现金"), ("cash and cash equivalents",)),
        (("operating_cash_flow", "经营现金流"), ("net cash provided by operating activities", "cash flows")),
        (("total_assets", "资产"), ("total assets",)),
        (("total_liabilities", "liabilities", "负债"), ("total liabilities",)),
        (("segment", "地区", "分部"), ("reportable segment", "operating segment", "americas", "europe", "greater china")),
        (("product", "产品"), ("iphone", "mac", "ipad", "wearables", "services")),
    )
    for triggers, expansions in field_aliases:
        if any(trigger in joined for trigger in triggers):
            aliases.update(expansion.casefold() for expansion in expansions)
    return aliases


def _slice_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _first_present(record: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _first_non_empty(values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    parsed = _coerce_float(value)
    return parsed


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / max(denominator, 1), 4)


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _chunk_score_row(
    query: str,
    chunk: DocumentChunk,
    index: int,
    *,
    term_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    term_weights = term_weights or _query_term_weights(query)
    lowered = chunk.text.casefold()
    matched_terms = {
        term: lowered.count(term)
        for term in term_weights
        if lowered.count(term) > 0
    }
    lexical_score = sum(count * term_weights[term] for term, count in matched_terms.items())
    bonus = _financial_chunk_bonus(query, lowered)
    table_bonus = 1.0 if chunk.kind == "table" else 0.0
    statement_bonus = _statement_type_bonus(query, str(chunk.metadata.get("statement_type") or ""))
    score = lexical_score + bonus + table_bonus + statement_bonus
    return {
        "chunk_id": chunk.chunk_id,
        "kind": chunk.kind,
        "table_index": chunk.table_index,
        "score": round(float(score), 4),
        "lexical_score": round(float(lexical_score), 4),
        "financial_bonus": bonus,
        "table_bonus": table_bonus,
        "statement_bonus": statement_bonus,
        "matched_terms": matched_terms,
        "caption": chunk.metadata.get("caption"),
        "context": chunk.metadata.get("context"),
        "statement_type": chunk.metadata.get("statement_type"),
        "unit_context": chunk.metadata.get("unit_context"),
        "rank_source_index": index,
    }


def _statement_type_bonus(query: str, statement_type: str) -> int:
    query_lower = query.casefold()
    if not statement_type:
        return 0
    if statement_type == "income_statement" and any(
        term in query_lower
        for term in ("revenue", "sales", "income", "margin", "营收", "收入", "利润")
    ):
        return 6
    if statement_type == "balance_sheet" and any(
        term in query_lower
        for term in ("asset", "liabilit", "cash", "资产", "负债", "现金")
    ):
        return 6
    if statement_type == "cash_flow" and any(
        term in query_lower
        for term in ("cash flow", "operating activities", "经营现金流")
    ):
        return 6
    return 0


def _nearby_table_context(table: Any) -> str:
    labels: list[str] = []
    for previous in table.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"], limit=8):
        text = previous.get_text(" ", strip=True)
        if not text or len(text) > 180:
            continue
        lowered = text.casefold()
        if any(
            term in lowered
            for term in (
                "consolidated",
                "statements",
                "statement",
                "operations",
                "balance sheets",
                "cash flows",
                "segment",
                "table",
                "in millions",
                "in thousands",
            )
        ):
            labels.append(text)
        if len(labels) >= 3:
            break
    return " / ".join(reversed(labels))


def _infer_statement_type(text: str) -> str | None:
    lowered = text.casefold()
    if "cash flow" in lowered or "cash flows" in lowered:
        return "cash_flow"
    if "balance sheet" in lowered or "balance sheets" in lowered:
        return "balance_sheet"
    if (
        "statement of operations" in lowered
        or "statements of operations" in lowered
        or "income statement" in lowered
        or "net sales" in lowered
        or "operating income" in lowered
    ):
        return "income_statement"
    if "segment" in lowered or "geographic" in lowered:
        return "segment"
    if "product" in lowered or "services" in lowered:
        return "product"
    return None


def infer_table_unit_context(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    context: dict[str, Any] = {}
    currency = _detect_currency(text)
    if currency:
        context["currency"] = currency
    elif re.search(r"\$|dollars?", lowered):
        context["currency"] = "USD"
    scale = _detect_scale(text)
    if scale:
        context["scale"] = scale
    if "basis point" in lowered or re.search(r"\bbps\b", lowered):
        context["percent_scale"] = "bps"
    elif "%" in text or "percent" in lowered:
        context["percent_scale"] = "percent"
    if "per share" in lowered or "per-share" in lowered:
        context["per_share"] = True
    if re.search(r"\bshares?\b", lowered):
        context["shares"] = True
    if "except per share" in lowered:
        context["except_per_share"] = True
    return context


def _validate_llm_extraction(graph: AttributeValueGraph, query: str) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    query_lower = query.casefold()
    fields = set(graph.fields)
    years = set(graph.years)
    canonical_ids = {
        token.canonical_concept_id
        for token in graph.tokens
        if token.canonical_concept_id
    }

    if _looks_like_growth_query(query_lower) and len(years) < 2:
        errors.append("missing_records_for_growth: expected at least two years of the same metric")
    if _looks_like_ratio_query(query_lower) and len(fields) < 2:
        errors.append("missing_ratio_components: expected raw numerator and denominator records")
    if _looks_like_candidate_query(query_lower) and len(graph.tokens) < 2:
        errors.append("candidate_set_too_small: count/ranking/average needs peer records")

    for token in graph.tokens:
        excerpt = (token.source.text_excerpt if token.source else "") or ""
        source_text = " ".join(
            str(item or "")
            for item in (
                token.source.table if token.source else "",
                token.source.column if token.source else "",
                excerpt,
            )
        ).casefold()
        if token.value == 0.0 and not re.search(r"(^|[^0-9])0(?:\.0+)?([^0-9]|$)|zero", excerpt.casefold()):
            warnings.append(f"zero_value_without_evidence:{token.field_name}:{token.year}")
        if (
            ("change" in source_text or "%" in source_text)
            and "%" not in query
            and "percent" not in query_lower
            and "change" not in query_lower
            and token.unit not in {"percent", "bps"}
        ):
            warnings.append(f"change_or_percentage_column_used:{token.field_name}:{token.year}")
        if ("per share" in source_text or "per-share" in source_text) and "per_share" not in token.field_name:
            warnings.append(f"per_share_context_on_non_per_share_field:{token.field_name}:{token.year}")
        if not token.canonical_concept_id:
            errors.append(f"missing_canonical_concept_id:{token.field_name}:{token.year}")
            continue
        concept = get_canonical_concept(token.canonical_concept_id)
        if concept is None:
            errors.append(f"unsupported_canonical_concept_id:{token.canonical_concept_id}")
            continue
        concept_validation = validate_concept_record(
            concept,
            unit=token.unit,
            dimensions=token.dimensions,
            raw_label=token.raw_label or token.field_label,
        )
        warnings.extend(concept_validation.warnings)
        errors.extend(concept_validation.errors)

    total_detail_warnings = _total_detail_warnings(graph)
    warnings.extend(total_detail_warnings)

    return {
        "warnings": list(dict.fromkeys(warnings)),
        "validation_errors": list(dict.fromkeys(errors)),
        "field_coverage": _field_coverage(query, fields),
        "canonical_coverage": _canonical_coverage(query, canonical_ids),
        "candidate_completeness": {
            "candidate_query": _looks_like_candidate_query(query_lower),
            "record_count": len(graph.tokens),
            "entity_count": len({token.entity_id for token in graph.tokens}),
            "field_count": len(fields),
            "year_count": len(years),
            "complete": not _looks_like_candidate_query(query_lower) or len(graph.tokens) >= 2,
        },
    }


def _looks_like_growth_query(query_lower: str) -> bool:
    return bool(re.search(r"growth|increase|decrease|percent change|增长|下降|同比|环比|增长率|变化率", query_lower))


def _looks_like_ratio_query(query_lower: str) -> bool:
    if _looks_like_growth_query(query_lower):
        return False
    return bool(re.search(r"ratio|margin|share|proportion|fraction|占|比例|比率|利润率|净利率|毛利率|除以", query_lower))


def _looks_like_candidate_query(query_lower: str) -> bool:
    return bool(
        re.search(
            r"how many|count|number of|top\s*\d|highest|largest|average|mean|"
            r"多少[家个项条]|几[家个项条]|数量|个数|前\s*\d|最高|最大|平均",
            query_lower,
        )
    )


def _field_coverage(query: str, fields: set[str]) -> dict[str, Any]:
    query_norm = normalize_identifier(query)
    mentioned: list[str] = []
    missing: list[str] = []
    for field_name in sorted(fields):
        aliases = {field_name, field_name.replace("_", " ")}
        if field_name == "revenue":
            aliases.update({"sales", "net sales", "收入", "营收"})
        if any(normalize_identifier(alias) in query_norm for alias in aliases):
            mentioned.append(field_name)
    return {
        "mentioned_fields_present": mentioned,
        "missing_mentioned_fields": missing,
        "graph_fields": sorted(fields),
    }


def _canonical_coverage(query: str, concept_ids: set[str | None]) -> dict[str, Any]:
    query_norm = normalize_identifier(query)
    present = sorted(concept_id for concept_id in concept_ids if concept_id)
    mentioned: list[str] = []
    for concept_id in present:
        concept = get_canonical_concept(concept_id)
        if concept is None:
            continue
        aliases = {concept.concept_id, concept.display_name, concept.default_field_name, *concept.aliases}
        if any(normalize_identifier(alias) in query_norm for alias in aliases):
            mentioned.append(concept_id)
    return {
        "present_concepts": present,
        "mentioned_concepts_present": sorted(set(mentioned)),
        "missing_mentioned_concepts": [],
    }


def _total_detail_warnings(graph: AttributeValueGraph) -> list[str]:
    warnings: list[str] = []
    by_field_year: dict[tuple[str, int | None], list[AttributeValueToken]] = {}
    for token in graph.tokens:
        by_field_year.setdefault((token.field_name, token.year), []).append(token)
    for (field_name, year), tokens in by_field_year.items():
        if len(tokens) < 2:
            continue
        source_texts = [
            ((token.source.text_excerpt if token.source else "") or "").casefold()
            for token in tokens
        ]
        has_total = any("total" in text or "合计" in text for text in source_texts)
        has_detail = any("total" not in text and "合计" not in text for text in source_texts)
        if has_total and has_detail:
            warnings.append(f"total_detail_mixed:{field_name}:{year}")
    return warnings


def _should_retry_extraction(validation: dict[str, Any]) -> bool:
    errors = list(validation.get("validation_errors", []))
    retryable_prefixes = (
        "missing_records_for_growth",
        "missing_ratio_components",
        "candidate_set_too_small",
    )
    return any(any(error.startswith(prefix) for prefix in retryable_prefixes) for error in errors)


def _retryable_error_messages(attempts: list[dict[str, Any]]) -> list[str]:
    if not attempts:
        return []
    messages: list[str] = []
    for attempt in attempts[-2:]:
        messages.extend(str(item) for item in attempt.get("validation_errors", []))
        if attempt.get("error"):
            messages.append(str(attempt["error"]))
    return messages
