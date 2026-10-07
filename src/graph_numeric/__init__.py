"""Schema-general graph numerical reasoning primitives."""

from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
    TokenSource,
    field_aliases,
    graph_from_csv_text,
    graph_from_document_file,
    graph_from_json_text,
    graph_from_markdown_file,
    graph_from_markdown_table,
    graph_from_text_table,
    graphs_from_markdown_files,
)
from graph_numeric.core.canonical_concepts import (
    CANONICAL_CONCEPTS,
    UNSUPPORTED_CONCEPT_ID,
    CanonicalConcept,
    canonical_concept_ids,
    concept_prompt_catalog,
    get_canonical_concept,
    resolve_canonical_concept,
)
from graph_numeric.extraction.document_extraction import (
    DocumentGraphExtraction,
    extract_document_graph_auto,
    extract_document_graph,
    extract_document_graph_from_text,
    extract_document_graph_with_llm_and_table_augmentation,
    extract_document_graph_with_llm_table_augmentation,
    extract_document_graph_with_llm_table_structure,
    extract_document_graph_with_llm,
    graph_summary,
    token_to_dict,
)
from graph_numeric.extraction.llm_extraction import LLMGraphExtractionResult, LLMGraphExtractor, LLMGraphExtractorConfig
from graph_numeric.extraction.llm_table_structure import LLMTableStructureParser, LLMTableStructureResult
from graph_numeric.learning.forecasting import FORECAST_PROFILES, forecast_profile
from graph_numeric.operators.executor import ExecutionResult, execute, execute_composite, execute_sum
from graph_numeric.learning.field_grounder import FieldGrounder, FieldGroundingResult
from graph_numeric.operators.operator_plan import CompositeOperatorPlan, Filter, OperatorPlan, Slot, SumOperatorPlan
from graph_numeric.pipeline.pipeline import PipelineAttempt, PipelineResult, execute_with_fallback, run_operator_pipeline
from graph_numeric.operators.operator_registry import (
    EXECUTOR_OPERATOR_ALIASES,
    OPERATOR_REGISTRY,
    OPERATOR_SPECS,
    PUBLIC_OPERATORS,
    OperatorRegistry,
    OperatorSpec,
)
from graph_numeric.operators.operator_solvers import OperatorSolver, extract_condition, solve_composite_operator_plan, solve_operator_plan
from graph_numeric.learning.router import OPERATOR_PATTERNS, EmbeddingRouter, HybridRouter, RoutingResult, RuleBasedRouter
from graph_numeric.learning.router_diagnostics import router_diagnostics
from graph_numeric.learning.router_factory import DEFAULT_EMBEDDING_ROUTER_DIR, RuntimeFallbackHybridRouter, build_router
from graph_numeric.adapters.storage_schema import (
    dameng_schema_summary,
    export_dameng_payload,
    export_storage_rows,
    export_trace_rows,
    reconstruct_fact_index,
)
from graph_numeric.extraction.text_span_extractor import TextSpanAnswer, extract_text_span_answer, looks_like_text_fact_question
from graph_numeric.core.unit_resolver import UnitIncompatibleError, UnitInfo, UnitResolver
from graph_numeric.operators.validation import PreconditionReport, validate_preconditions
from graph_numeric.operators.verifier import VerificationReport, Verifier

__all__ = [
    "AttributeValueGraph",
    "AttributeValueToken",
    "CANONICAL_CONCEPTS",
    "CanonicalConcept",
    "EmbeddingRouter",
    "ExecutionResult",
    "FieldGrounder",
    "FieldGroundingResult",
    "FORECAST_PROFILES",
    "Filter",
    "HybridRouter",
    "LLMGraphExtractionResult",
    "LLMGraphExtractor",
    "LLMGraphExtractorConfig",
    "LLMTableStructureParser",
    "LLMTableStructureResult",
    "CompositeOperatorPlan",
    "DEFAULT_EMBEDDING_ROUTER_DIR",
    "DocumentGraphExtraction",
    "EXECUTOR_OPERATOR_ALIASES",
    "OPERATOR_REGISTRY",
    "OPERATOR_PATTERNS",
    "OPERATOR_SPECS",
    "OperatorPlan",
    "OperatorRegistry",
    "OperatorSolver",
    "OperatorSpec",
    "PipelineAttempt",
    "PipelineResult",
    "PreconditionReport",
    "PUBLIC_OPERATORS",
    "RoutingResult",
    "RuleBasedRouter",
    "RuntimeFallbackHybridRouter",
    "Slot",
    "SumOperatorPlan",
    "TokenSource",
    "TextSpanAnswer",
    "UNSUPPORTED_CONCEPT_ID",
    "UnitIncompatibleError",
    "UnitInfo",
    "UnitResolver",
    "VerificationReport",
    "Verifier",
    "build_router",
    "canonical_concept_ids",
    "concept_prompt_catalog",
    "dameng_schema_summary",
    "execute",
    "execute_composite",
    "execute_sum",
    "execute_with_fallback",
    "export_dameng_payload",
    "export_storage_rows",
    "export_trace_rows",
    "extract_document_graph_auto",
    "extract_document_graph",
    "extract_document_graph_from_text",
    "extract_document_graph_with_llm_and_table_augmentation",
    "extract_document_graph_with_llm_table_augmentation",
    "extract_document_graph_with_llm_table_structure",
    "extract_document_graph_with_llm",
    "extract_text_span_answer",
    "extract_condition",
    "field_aliases",
    "graph_from_csv_text",
    "graph_from_document_file",
    "graph_from_json_text",
    "forecast_profile",
    "get_canonical_concept",
    "graph_from_markdown_file",
    "graph_from_markdown_table",
    "graph_from_text_table",
    "graph_summary",
    "graphs_from_markdown_files",
    "looks_like_text_fact_question",
    "reconstruct_fact_index",
    "run_operator_pipeline",
    "router_diagnostics",
    "resolve_canonical_concept",
    "solve_operator_plan",
    "solve_composite_operator_plan",
    "token_to_dict",
    "validate_preconditions",
]
