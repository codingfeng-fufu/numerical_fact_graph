from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource, field_aliases, graph_from_document_file
from graph_numeric.core.attribute_graph import graph_from_csv_text, graph_from_json_text, graph_from_markdown_table, graph_from_text_table
from graph_numeric.extraction.llm_extraction import LLMGraphExtractor, LLMGraphExtractorConfig
from graph_numeric.extraction.llm_table_structure import LLMTableStructureParser
from graph_numeric.learning.evidence_faithfulness import TokenFaithfulnessChecker, evaluate_token_faithfulness


@dataclass(frozen=True)
class DocumentGraphExtraction:
    """Document parser output used by the end-to-end demo.

    The graph itself is kept for execution; ``to_dict`` exposes the extraction
    metadata that should be visible in a demo or report.
    """

    document_path: Path
    parser: str
    graph: AttributeValueGraph
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, max_tokens: int = 12) -> dict[str, Any]:
        payload = {
            "document": str(self.document_path),
            "parser": self.parser,
            "graph_summary": graph_summary(self.graph),
            "tokens_preview": [
                token_to_dict(token)
                for token in self.graph.tokens[:max_tokens]
            ],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
            if self.metadata.get("source") == "llm":
                payload["llm"] = self.metadata
        return payload


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str | None
    table: str | None
    row: int | None
    token_ids: tuple[str, ...]
    entities: tuple[str, ...]
    industries: tuple[str, ...]
    fields: tuple[str, ...]
    years: tuple[int, ...]
    units: tuple[str, ...]
    text: str
    char_start: int | None = None
    char_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "table": self.table,
            "row": self.row,
            "token_ids": list(self.token_ids),
            "entities": list(self.entities),
            "industries": list(self.industries),
            "fields": list(self.fields),
            "years": list(self.years),
            "units": list(self.units),
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def extract_document_graph(path: str | Path) -> DocumentGraphExtraction:
    """Parse a table-like document into an AttributeValueGraph.

    Supported formats match ``graph_from_document_file``: CSV, JSON,
    Markdown tables and whitespace/text tables.
    """
    document_path = Path(path)
    graph = graph_from_document_file(document_path)
    return DocumentGraphExtraction(
        document_path=document_path,
        parser=_parser_name(document_path),
        graph=graph,
    )


def extract_document_graph_from_text(
    text: str,
    *,
    source_name: str = "uploaded.csv",
) -> DocumentGraphExtraction:
    """Parse uploaded document text without requiring a file on disk."""
    source = Path(source_name)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        graph = graph_from_csv_text(text, source_name=source_name)
    elif suffix == ".json":
        graph = graph_from_json_text(text, source_name=source_name)
    elif suffix in {".md", ".markdown"}:
        graph = graph_from_markdown_table(text, source_name=source_name)
    else:
        graph = graph_from_text_table(text, source_name=source_name)
    return DocumentGraphExtraction(
        document_path=source,
        parser=_parser_name(source),
        graph=graph,
    )


def extract_document_graph_with_llm(
    path_or_text: str | Path,
    query: str,
    *,
    source_name: str | None = None,
    llm_config: LLMGraphExtractorConfig | None = None,
    fallback_to_structured: bool = True,
    faithfulness_checker: TokenFaithfulnessChecker | None = None,
    max_numeric_candidates: int | None = None,
    intent_conditioned_numeric_budget: bool = False,
) -> DocumentGraphExtraction:
    """Parse a document into graph tokens using query-conditioned LLM extraction.

    The LLM only produces grounded records. If it is unavailable or returns an
    invalid payload, callers may opt into structured fallback so demos keep a
    deterministic path for CSV/JSON/Markdown inputs.
    """
    source, text = _resolve_text_source(path_or_text, source_name=source_name)
    try:
        extractor_kwargs: dict[str, Any] = {}
        if max_numeric_candidates is not None:
            extractor_kwargs["max_numeric_candidates"] = max_numeric_candidates
        extractor_kwargs["intent_conditioned_numeric_budget"] = (
            intent_conditioned_numeric_budget
        )
        extraction = LLMGraphExtractor(config=llm_config, **extractor_kwargs).extract(
            text,
            query,
            source_name=source.name,
        )
        return DocumentGraphExtraction(
            document_path=source,
            parser="llm_query_conditioned",
            graph=extraction.graph,
            metadata=_metadata_with_s2_faithfulness(
                extraction.metadata,
                extraction.graph,
                faithfulness_checker,
            ),
        )
    except Exception as exc:
        if not fallback_to_structured:
            raise
        structured = extract_document_graph_from_text(text, source_name=source.name)
        return DocumentGraphExtraction(
            document_path=structured.document_path,
            parser=f"{structured.parser}_fallback_after_llm_error",
            graph=structured.graph,
            metadata={
                "source": "structured_fallback",
                "llm_error": str(exc),
                "requested_parser": "llm_query_conditioned",
            },
        )


def _metadata_with_s2_faithfulness(
    metadata: dict[str, Any],
    graph: AttributeValueGraph,
    checker: TokenFaithfulnessChecker | None,
) -> dict[str, Any]:
    if checker is None:
        return metadata
    return {
        **metadata,
        "s2_faithfulness": evaluate_token_faithfulness(
            tuple(graph.tokens),
            checker,
        ),
    }


def extract_document_graph_with_llm_table_structure(
    path_or_text: str | Path,
    query: str | None = None,
    *,
    source_name: str | None = None,
    llm_config: LLMGraphExtractorConfig | None = None,
    fallback_to_structured: bool = True,
) -> DocumentGraphExtraction:
    """Parse table structure with an LLM, then adapt normalized cells to graph tokens.

    Unlike query-conditioned graph extraction, this parser keeps generic table
    fields from header paths, so it can represent facts such as Fair Value even
    when they are outside the closed canonical concept registry.
    """
    source, text = _resolve_text_source(path_or_text, source_name=source_name)
    try:
        extraction = LLMTableStructureParser(config=llm_config).parse(
            text,
            query=query,
            source_name=source.name,
        )
        return DocumentGraphExtraction(
            document_path=source,
            parser="llm_table_structure",
            graph=extraction.graph,
            metadata=extraction.metadata,
        )
    except Exception as exc:
        if not fallback_to_structured:
            raise
        structured = extract_document_graph_from_text(text, source_name=source.name)
        return DocumentGraphExtraction(
            document_path=structured.document_path,
            parser=f"{structured.parser}_fallback_after_llm_table_structure_error",
            graph=structured.graph,
            metadata={
                "source": "structured_fallback",
                "llm_error": str(exc),
                "requested_parser": "llm_table_structure",
            },
        )


def extract_document_graph_with_llm_table_augmentation(
    path_or_text: str | Path,
    query: str | None = None,
    *,
    source_name: str | None = None,
    llm_config: LLMGraphExtractorConfig | None = None,
    fallback_to_structured: bool = True,
) -> DocumentGraphExtraction:
    """Run the normal parser, then augment table numeric facts with LLM structure.

    This keeps the original deterministic graph as the baseline and uses the LLM
    only to add cell-level facts with multi-level row/column paths.
    """
    source, text = _resolve_text_source(path_or_text, source_name=source_name)
    structured = extract_document_graph_from_text(text, source_name=source.name)
    try:
        llm_table = LLMTableStructureParser(config=llm_config).parse(
            text,
            query=query,
            source_name=source.name,
        )
        graph = _merge_graphs(structured.graph, llm_table.graph, source_name=source.name)
        return DocumentGraphExtraction(
            document_path=source,
            parser=f"{structured.parser}_augmented_with_llm_table_structure",
            graph=graph,
            metadata={
                "source": "structured_plus_llm_table_structure",
                "structured_parser": structured.parser,
                "llm_parser": "llm_table_structure",
                "structured_token_count": len(structured.graph.tokens),
                "llm_table_token_count": len(llm_table.graph.tokens),
                "record_count": len(graph.tokens),
                "llm_table": llm_table.metadata,
            },
        )
    except Exception as exc:
        if not fallback_to_structured:
            raise
        return DocumentGraphExtraction(
            document_path=structured.document_path,
            parser=f"{structured.parser}_fallback_after_llm_table_augmentation_error",
            graph=structured.graph,
            metadata={
                "source": "structured_fallback",
                "llm_error": str(exc),
                "requested_parser": "llm_table_structure_augmentation",
                "structured_parser": structured.parser,
                "structured_token_count": len(structured.graph.tokens),
            },
        )


def extract_document_graph_with_llm_and_table_augmentation(
    path_or_text: str | Path,
    query: str,
    *,
    source_name: str | None = None,
    llm_config: LLMGraphExtractorConfig | None = None,
    fallback_to_query_llm: bool = True,
    fallback_to_structured: bool = False,
    faithfulness_checker: TokenFaithfulnessChecker | None = None,
    max_numeric_candidates: int | None = None,
    intent_conditioned_numeric_budget: bool = False,
) -> DocumentGraphExtraction:
    """Run query-conditioned LLM extraction, then add LLM table cell facts.

    This preserves the original complete LLM path as the primary graph and uses
    the table-structure LLM only to add auditable numeric cell facts.
    """
    source, text = _resolve_text_source(path_or_text, source_name=source_name)
    query_llm: DocumentGraphExtraction | None = None
    query_llm_error: str | None = None
    try:
        query_llm = extract_document_graph_with_llm(
            text,
            query,
            source_name=source.name,
            llm_config=llm_config,
            fallback_to_structured=fallback_to_structured,
            faithfulness_checker=faithfulness_checker,
            max_numeric_candidates=max_numeric_candidates,
            intent_conditioned_numeric_budget=intent_conditioned_numeric_budget,
        )
    except Exception as exc:
        if not fallback_to_query_llm:
            raise
        query_llm_error = str(exc)
    try:
        llm_table = LLMTableStructureParser(config=llm_config).parse(
            text,
            query=query,
            source_name=source.name,
        )
        if query_llm is None:
            return DocumentGraphExtraction(
                document_path=source,
                parser="llm_table_structure_after_query_llm_error",
                graph=llm_table.graph,
                metadata={
                    "source": "llm_table_structure_after_query_llm_error",
                    "query_llm_error": query_llm_error,
                    "llm_parser": "llm_table_structure",
                    "llm_table_token_count": len(llm_table.graph.tokens),
                    "record_count": len(llm_table.graph.tokens),
                    "llm_table": llm_table.metadata,
                },
            )
        graph = _merge_graphs(query_llm.graph, llm_table.graph, source_name=source.name)
        return DocumentGraphExtraction(
            document_path=source,
            parser=f"{query_llm.parser}_augmented_with_llm_table_structure",
            graph=graph,
            metadata={
                "source": "llm_query_conditioned_plus_llm_table_structure",
                "query_llm_parser": query_llm.parser,
                "llm_parser": "llm_table_structure",
                "query_llm_token_count": len(query_llm.graph.tokens),
                "llm_table_token_count": len(llm_table.graph.tokens),
                "record_count": len(graph.tokens),
                "query_llm": query_llm.metadata,
                "llm_table": llm_table.metadata,
            },
        )
    except Exception as exc:
        if query_llm is None:
            if fallback_to_structured:
                structured = extract_document_graph_from_text(text, source_name=source.name)
                return DocumentGraphExtraction(
                    document_path=structured.document_path,
                    parser=f"{structured.parser}_fallback_after_llm_and_table_augmentation_error",
                    graph=structured.graph,
                    metadata={
                        "source": "structured_fallback",
                        "query_llm_error": query_llm_error,
                        "llm_table_error": str(exc),
                        "requested_parser": "llm_query_conditioned_table_augmentation",
                        "structured_parser": structured.parser,
                        "structured_token_count": len(structured.graph.tokens),
                    },
                )
            raise RuntimeError(
                "Both query-conditioned LLM extraction and LLM table augmentation failed: "
                f"query_llm_error={query_llm_error}; llm_table_error={exc}"
            ) from exc
        if not fallback_to_query_llm:
            raise
        return DocumentGraphExtraction(
            document_path=query_llm.document_path,
            parser=f"{query_llm.parser}_fallback_after_llm_table_augmentation_error",
            graph=query_llm.graph,
            metadata={
                "source": "query_llm_fallback",
                "llm_error": str(exc),
                "requested_parser": "llm_query_conditioned_table_augmentation",
                "query_llm_parser": query_llm.parser,
                "query_llm_token_count": len(query_llm.graph.tokens),
                "query_llm": query_llm.metadata,
            },
        )


def extract_document_graph_auto(
    path_or_text: str | Path,
    query: str,
    *,
    source_name: str | None = None,
    llm_config: LLMGraphExtractorConfig | None = None,
) -> DocumentGraphExtraction:
    """Prefer deterministic structured parsing, then use LLM as a fallback."""
    source, text = _resolve_text_source(path_or_text, source_name=source_name)
    structured = extract_document_graph_from_text(text, source_name=source.name)
    if structured.graph.tokens:
        return structured
    return extract_document_graph_with_llm(
        text,
        query,
        source_name=source.name,
        llm_config=llm_config,
        fallback_to_structured=True,
    )


def _merge_graphs(
    primary: AttributeValueGraph,
    secondary: AttributeValueGraph,
    *,
    source_name: str | None,
) -> AttributeValueGraph:
    tokens: list[AttributeValueToken] = []
    seen: set[tuple[object, ...]] = set()
    for token in (*primary.tokens, *secondary.tokens):
        source = token.source
        key = (
            token.company_name,
            token.field_name,
            token.year,
            token.value,
            source.table if source else None,
            source.row if source else None,
            source.column if source else None,
            source.text_excerpt if source else None,
        )
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def build_evidence_chunks(graph: AttributeValueGraph) -> tuple[EvidenceChunk, ...]:
    """Group graph tokens into source-aligned evidence chunks.

    Table-like documents use one chunk per source row. If source row metadata is
    unavailable, the token itself becomes a one-token chunk. These chunks are
    used by offline benchmarks and demos to expose the retrieval unit that sits
    between vector grounding and scalar execution.
    """
    groups: dict[tuple[str | None, str | None, int | None, str | None], list[AttributeValueToken]] = {}
    for token in graph.tokens:
        source = token.source
        key = (
            source.document_id if source is not None else graph.source_name,
            source.table if source is not None else None,
            source.row if source is not None else None,
            None if source is not None and source.row is not None else token.token_id,
        )
        groups.setdefault(key, []).append(token)

    chunks = []
    for index, (key, tokens) in enumerate(groups.items()):
        document_id, table, row, _ = key
        sources = [token.source for token in tokens if token.source is not None]
        char_starts = [source.char_start for source in sources if source.char_start is not None]
        char_ends = [source.char_end for source in sources if source.char_end is not None]
        excerpts = [
            source.text_excerpt
            for source in sources
            if source.text_excerpt
        ]
        chunk_id = ":".join(
            part
            for part in (
                document_id or graph.source_name or "graph",
                table,
                f"row_{row}" if row is not None else f"token_{index}",
            )
            if part
        )
        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                document_id=document_id or graph.source_name,
                table=table,
                row=row,
                token_ids=tuple(token.token_id for token in tokens),
                entities=tuple(sorted({token.company_name for token in tokens})),
                industries=tuple(sorted({token.industry for token in tokens if token.industry})),
                fields=tuple(sorted({token.field_name for token in tokens})),
                years=tuple(sorted({token.year for token in tokens if token.year is not None})),
                units=tuple(sorted({token.unit for token in tokens if token.unit})),
                text=" | ".join(excerpts),
                char_start=min(char_starts) if char_starts else None,
                char_end=max(char_ends) if char_ends else None,
            )
        )
    return tuple(chunks)


def merge_evidence_graphs(
    graphs: list[AttributeValueGraph] | tuple[AttributeValueGraph, ...],
    *,
    source_name: str = "merged_evidence_corpus",
) -> AttributeValueGraph:
    """Merge multiple document graphs while preserving source provenance."""
    merged_tokens: list[AttributeValueToken] = []
    used_token_ids: set[str] = set()
    for graph_index, graph in enumerate(graphs):
        source_prefix = graph.source_name or f"graph_{graph_index}"
        for token in graph.tokens:
            token_id = f"{source_prefix}:{token.token_id}"
            if token_id in used_token_ids:
                token_id = f"{source_prefix}:{graph_index}:{token.token_id}"
            used_token_ids.add(token_id)
            source = _source_with_document_id(token.source, source_prefix)
            merged_tokens.append(
                replace(
                    token,
                    token_id=token_id,
                    entity_id=f"{source_prefix}:{token.entity_id}",
                    source=source,
                )
            )
    return AttributeValueGraph(tuple(merged_tokens), source_name=source_name)


def graph_summary(graph: AttributeValueGraph) -> dict[str, Any]:
    entities = sorted({token.company_name for token in graph.tokens})
    units = sorted({token.unit for token in graph.tokens if token.unit})
    records_with_unit = sum(1 for token in graph.tokens if token.unit)
    records_with_source = sum(1 for token in graph.tokens if token.source is not None)
    records_with_text_span = sum(
        1
        for token in graph.tokens
        if token.source is not None
        and (
            token.source.text_excerpt is not None
            or (token.source.char_start is not None and token.source.char_end is not None)
        )
    )
    by_field = {
        field: len(graph.select(field_name=field))
        for field in graph.fields
    }
    field_profiles = {
        field: _phase2_field_profile(field, graph.get_field_profile(field))
        for field in graph.fields
    }
    unit_profiles = {
        field: sorted({token.unit for token in graph.select(field_name=field) if token.unit})
        for field in graph.fields
    }
    canonical_concepts = sorted(
        {
            token.canonical_concept_id
            for token in graph.tokens
            if token.canonical_concept_id
        }
    )
    return {
        "graph_type": "numerical_evidence_graph",
        "schema_version": "neg_schema_v2",
        "source_name": graph.source_name,
        "token_count": len(graph.tokens),
        "entity_count": len(entities),
        "entities": entities,
        "entity_names": entities,
        "field_count": len(graph.fields),
        "fields": list(graph.fields),
        "field_profiles": field_profiles,
        "canonical_concepts": canonical_concepts,
        "years": list(graph.years),
        "industries": list(graph.industries),
        "units": units,
        "unit_count": len(units),
        "records_with_unit": records_with_unit,
        "unit_coverage": _ratio(records_with_unit, len(graph.tokens)),
        "tokens_by_field": by_field,
        "records_with_source": records_with_source,
        "source_coverage": _ratio(records_with_source, len(graph.tokens)),
        "source_span_coverage": _ratio(records_with_text_span, len(graph.tokens)),
        "unit_profiles": unit_profiles,
        "source_completeness": {
            "tokens_with_source": records_with_source,
            "tokens_with_excerpt": records_with_text_span,
            "source_coverage": _ratio(records_with_source, len(graph.tokens)),
            "excerpt_coverage": _ratio(records_with_text_span, len(graph.tokens)),
        },
        "evidence_coverage": {
            "records_with_source": records_with_source,
            "records_with_text_span": records_with_text_span,
            "source_coverage": _ratio(records_with_source, len(graph.tokens)),
            "source_span_coverage": _ratio(records_with_text_span, len(graph.tokens)),
        },
        "chunk_count": len(build_evidence_chunks(graph)),
    }


def token_to_dict(token: AttributeValueToken) -> dict[str, Any]:
    source = token.source
    return {
        "token_id": token.token_id,
        "entity_id": token.entity_id,
        "company_name": token.company_name,
        "industry": token.industry,
        "year": token.year,
        "field_name": token.field_name,
        "field_label": token.field_label,
        "canonical_concept_id": token.canonical_concept_id,
        "dimensions": dict(token.dimensions or {}),
        "raw_label": token.raw_label,
        "external_concept_ids": list(token.external_concept_ids),
        "value": token.value,
        "unit": token.unit,
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


def _parser_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv_table"
    if suffix == ".json":
        return "json_records"
    if suffix in {".md", ".markdown"}:
        return "markdown_table"
    return "text_table"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / max(denominator, 1), 4)


def _jsonable_profile(profile: dict[str, object]) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in profile.items()
    }


def _phase2_field_profile(field: str, profile: dict[str, object]) -> dict[str, object]:
    payload = _jsonable_profile(profile)
    payload["aliases"] = list(field_aliases(field))
    if "min" in payload and "max" in payload:
        payload["value_range"] = {
            "min": payload["min"],
            "max": payload["max"],
        }
    else:
        payload["value_range"] = None
    return payload


def _source_with_document_id(source: TokenSource | None, document_id: str) -> TokenSource | None:
    if source is None:
        return TokenSource(document_id=document_id)
    if source.document_id:
        return source
    return replace(source, document_id=document_id)


def _resolve_text_source(path_or_text: str | Path, *, source_name: str | None) -> tuple[Path, str]:
    if isinstance(path_or_text, Path):
        path = path_or_text
        return path, path.read_text(encoding="utf-8", errors="ignore")
    value = str(path_or_text)
    if "\n" in value or "\r" in value or len(value) > 240:
        return Path(source_name or "document.txt"), value
    path = Path(value)
    if path.exists() and path.is_file():
        return path, path.read_text(encoding="utf-8", errors="ignore")
    return Path(source_name or "document.txt"), value
