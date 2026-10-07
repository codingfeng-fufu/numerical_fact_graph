from __future__ import annotations

from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken
from graph_numeric.pipeline.pipeline import PipelineResult


def dameng_schema_summary() -> dict[str, Any]:
    """Return the first-stage Dameng storage contract for NEG rows."""
    return {
        "adapter": "dameng_numerical_evidence_graph_v1",
        "graph_type": "numerical_evidence_graph",
        "tables": {
            "documents": {
                "primary_key": "document_id",
                "columns": [
                    "document_id",
                    "source_name",
                    "token_count",
                ],
            },
            "entities": {
                "primary_key": "entity_id",
                "columns": [
                    "entity_id",
                    "document_id",
                    "entity_name",
                    "entity_type",
                    "metadata",
                ],
            },
            "fields": {
                "primary_key": "field_id",
                "columns": [
                    "field_id",
                    "field_name",
                    "field_label",
                    "canonical_concept_id",
                    "raw_labels",
                    "default_unit",
                ],
            },
            "numeric_facts": {
                "primary_key": "fact_id",
                "columns": [
                    "fact_id",
                    "document_id",
                    "entity_id",
                    "field_name",
                    "value",
                    "year",
                    "unit",
                    "industry",
                    "dimensions",
                    "raw_label",
                    "evidence_id",
                ],
            },
            "evidence_spans": {
                "primary_key": "evidence_id",
                "columns": [
                    "evidence_id",
                    "document_id",
                    "page",
                    "table_name",
                    "row_index",
                    "column_name",
                    "char_start",
                    "char_end",
                    "text_excerpt",
                ],
            },
            "query_sessions": {
                "primary_key": "query_id",
                "columns": [
                    "query_id",
                    "document_id",
                    "query_text",
                    "selected_strategy",
                    "status",
                    "selected_operator",
                ],
            },
            "operator_plans": {
                "primary_key": "plan_id",
                "columns": [
                    "plan_id",
                    "query_id",
                    "operator",
                    "plan_json",
                    "trace_json",
                ],
            },
            "execution_traces": {
                "primary_key": "trace_id",
                "columns": [
                    "trace_id",
                    "query_id",
                    "plan_id",
                    "answer",
                    "calculation",
                    "selected_fact_ids",
                    "records_used",
                    "output_unit",
                    "execution_metadata_json",
                ],
            },
            "verification_reports": {
                "primary_key": "report_id",
                "columns": [
                    "report_id",
                    "query_id",
                    "trace_id",
                    "passed",
                    "checks",
                    "warnings",
                    "failed_checks",
                    "failure_categories",
                    "grouped_checks",
                ],
            },
        },
        "indexes": {
            "documents": ["source_name"],
            "entities": ["document_id", "entity_name"],
            "fields": ["field_name", "canonical_concept_id"],
            "numeric_facts": [
                "document_id",
                "entity_id",
                "field_name",
                "year",
                "unit",
                "industry",
                "value",
            ],
            "evidence_spans": ["document_id", "table_name", "row_index", "column_name"],
            "query_sessions": ["document_id", "selected_strategy", "status"],
            "operator_plans": ["query_id", "operator"],
            "execution_traces": ["query_id", "plan_id"],
            "verification_reports": ["query_id", "passed", "failure_categories"],
        },
        "vector_columns": {
            "fields": ["embedding"],
            "entities": ["embedding"],
            "evidence_spans": ["embedding"],
            "query_sessions": ["query_embedding"],
        },
    }


def export_dameng_payload(
    graph: AttributeValueGraph,
    *,
    query: str | None = None,
    pipeline: PipelineResult | None = None,
    document_id: str | None = None,
    query_id: str = "query-1",
) -> dict[str, Any]:
    """Export graph and optional query trace rows with the Dameng schema contract."""
    payload: dict[str, Any] = {
        "schema_summary": dameng_schema_summary(),
        "graph_rows": export_storage_rows(graph, document_id=document_id),
        "trace_rows": {},
    }
    if query is not None and pipeline is not None:
        payload["trace_rows"] = export_trace_rows(
            query=query,
            pipeline=pipeline,
            query_id=query_id,
            document_id=document_id,
        )
    return payload


def export_storage_rows(
    graph: AttributeValueGraph,
    *,
    document_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Export graph facts into logical rows for a database adapter."""
    resolved_document_id = document_id or graph.source_name or "document"
    return {
        "documents": [_document_row(graph, resolved_document_id)],
        "entities": _entity_rows(graph, resolved_document_id),
        "fields": _field_rows(graph),
        "numeric_facts": [
            _numeric_fact_row(token, resolved_document_id)
            for token in graph.tokens
        ],
        "evidence_spans": [
            _evidence_row(token, resolved_document_id)
            for token in graph.tokens
            if token.source is not None
        ],
    }


def reconstruct_fact_index(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Reconstruct a fact-index view from exported logical rows.

    This helper is intentionally small: it proves the exported rows preserve the
    fact/evidence relationship needed by a Dameng adapter without pretending to
    be a full database import layer.
    """
    evidence_by_id = {
        row["evidence_id"]: row
        for row in rows.get("evidence_spans", [])
    }
    facts: dict[str, dict[str, Any]] = {}
    for fact in rows.get("numeric_facts", []):
        row = dict(fact)
        evidence_id = row.get("evidence_id")
        row["evidence"] = evidence_by_id.get(evidence_id)
        facts[str(row["fact_id"])] = row
    return facts


def export_trace_rows(
    *,
    query: str,
    pipeline: PipelineResult,
    query_id: str,
    document_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Export one pipeline run into logical audit rows."""
    plan_id = f"{query_id}:plan"
    trace_id = f"{query_id}:trace"
    report_id = f"{query_id}:verification"
    plan_payload = pipeline.plan.to_dict() if pipeline.plan is not None else None
    result = pipeline.result
    hybrid_trace = pipeline.hybrid_query_trace()
    execution_metadata = pipeline.execution_metadata()
    verification_report = pipeline.verification_report()
    return {
        "query_sessions": [
            {
                "query_id": query_id,
                "document_id": document_id,
                "query_text": query,
                "selected_strategy": hybrid_trace["strategy"],
                "status": pipeline.status,
                "selected_operator": pipeline.selected_operator,
            }
        ],
        "operator_plans": [
            {
                "plan_id": plan_id,
                "query_id": query_id,
                "operator": pipeline.selected_operator,
                "plan_json": plan_payload,
                "trace_json": hybrid_trace,
            }
        ],
        "execution_traces": [
            {
                "trace_id": trace_id,
                "query_id": query_id,
                "plan_id": plan_id,
                "answer": pipeline.answer,
                "calculation": result.calculation if result is not None else None,
                "selected_fact_ids": [
                    token.token_id
                    for token in (result.selected_tokens if result is not None else ())
                ],
                "records_used": (
                    result.metadata.get("records_used", [])
                    if result is not None
                    else []
                ),
                "output_unit": (
                    result.metadata.get("output_unit")
                    if result is not None
                    else None
                ),
                "execution_metadata_json": execution_metadata,
            }
        ],
        "verification_reports": [
            {
                "report_id": report_id,
                "query_id": query_id,
                "trace_id": trace_id,
                "passed": verification_report["passed"],
                "checks": verification_report["checks"],
                "warnings": verification_report["warnings"],
                "failed_checks": verification_report["failed_checks"],
                "failure_categories": verification_report["failure_categories"],
                "grouped_checks": verification_report["grouped_checks"],
            }
        ],
    }


def _document_row(graph: AttributeValueGraph, document_id: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "source_name": graph.source_name,
        "token_count": len(graph.tokens),
    }


def _entity_rows(
    graph: AttributeValueGraph,
    document_id: str,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for token in graph.tokens:
        rows.setdefault(
            token.entity_id,
            {
                "entity_id": token.entity_id,
                "document_id": document_id,
                "entity_name": token.company_name,
                "entity_type": "company",
                "metadata": {
                    "industry": token.industry,
                },
            },
        )
    return [rows[key] for key in sorted(rows)]


def _field_rows(graph: AttributeValueGraph) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for token in graph.tokens:
        rows.setdefault(
            token.field_name,
            {
                "field_id": token.field_name,
                "field_name": token.field_name,
                "field_label": token.field_label,
                "canonical_concept_id": token.canonical_concept_id,
                "raw_labels": [],
                "default_unit": token.unit,
            },
        )
        if token.raw_label and token.raw_label not in rows[token.field_name]["raw_labels"]:
            rows[token.field_name]["raw_labels"].append(token.raw_label)
    return [rows[key] for key in sorted(rows)]


def _numeric_fact_row(
    token: AttributeValueToken,
    document_id: str,
) -> dict[str, Any]:
    return {
        "fact_id": token.token_id,
        "document_id": document_id,
        "entity_id": token.entity_id,
        "field_name": token.field_name,
        "value": token.value,
        "year": token.year,
        "unit": token.unit,
        "industry": token.industry,
        "dimensions": dict(token.dimensions or {}),
        "raw_label": token.raw_label,
        "evidence_id": _evidence_id(token),
    }


def _evidence_row(
    token: AttributeValueToken,
    document_id: str,
) -> dict[str, Any]:
    source = token.source
    if source is None:
        raise ValueError("Cannot export evidence for token without source")
    return {
        "evidence_id": _evidence_id(token),
        "document_id": source.document_id or document_id,
        "page": source.page,
        "table_name": source.table,
        "row_index": source.row,
        "column_name": source.column,
        "char_start": source.char_start,
        "char_end": source.char_end,
        "text_excerpt": source.text_excerpt,
    }


def _evidence_id(token: AttributeValueToken) -> str | None:
    if token.source is None:
        return None
    return f"{token.token_id}:evidence"
