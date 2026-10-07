from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from graph_numeric.core.attribute_graph import normalize_identifier
from graph_numeric.core.unit_resolver import UnitResolver


UNSUPPORTED_CONCEPT_ID = "unsupported_concept"

CANONICAL_DIMENSION_KEYS: tuple[str, ...] = (
    "entity_scope",
    "statement_type",
    "period_type",
    "fiscal_period",
    "segment",
    "product",
    "region",
    "attributable_to",
)


@dataclass(frozen=True)
class CanonicalConcept:
    concept_id: str
    display_name: str
    category: str
    default_field_name: str
    aliases: tuple[str, ...] = ()
    allowed_statement_types: tuple[str, ...] = ()
    external_mappings: tuple[str, ...] = ()
    guidance: str = ""


@dataclass(frozen=True)
class ConceptValidation:
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


CANONICAL_CONCEPTS: dict[str, CanonicalConcept] = {
    "revenue": CanonicalConcept(
        concept_id="revenue",
        display_name="Revenue",
        category="money",
        default_field_name="revenue",
        aliases=(
            "revenue",
            "revenues",
            "total revenue",
            "total revenues",
            "sales",
            "net sales",
            "sales revenue",
            "营收",
            "收入",
            "营业收入",
        ),
        allowed_statement_types=("income_statement", "segment", "product"),
        external_mappings=(
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:SalesRevenueNet",
        ),
        guidance="Use for company-level total revenue/sales unless the query explicitly names total_net_sales.",
    ),
    "total_net_sales": CanonicalConcept(
        concept_id="total_net_sales",
        display_name="Total net sales",
        category="money",
        default_field_name="total_net_sales",
        aliases=("total net sales", "total_net_sales"),
        allowed_statement_types=("income_statement", "segment", "product"),
        external_mappings=("us-gaap:SalesRevenueNet",),
        guidance="Use only when the query or source explicitly targets total_net_sales as a distinct field.",
    ),
    "gross_margin": CanonicalConcept(
        concept_id="gross_margin",
        display_name="Gross margin",
        category="money",
        default_field_name="gross_margin",
        aliases=("gross margin", "gross profit", "毛利", "毛利润"),
        allowed_statement_types=("income_statement", "segment", "product"),
        external_mappings=("us-gaap:GrossProfit",),
        guidance="This is the gross-margin amount from statements, not a precomputed percentage margin.",
    ),
    "operating_income": CanonicalConcept(
        concept_id="operating_income",
        display_name="Operating income",
        category="money",
        default_field_name="operating_income",
        aliases=("operating income", "operating income loss", "operating profit", "经营利润", "营业利润"),
        allowed_statement_types=("income_statement", "segment"),
        external_mappings=("us-gaap:OperatingIncomeLoss",),
    ),
    "net_income": CanonicalConcept(
        concept_id="net_income",
        display_name="Net income",
        category="money",
        default_field_name="net_income",
        aliases=("net income", "net earnings", "profit loss", "net profit", "净利润", "净收益"),
        allowed_statement_types=("income_statement",),
        external_mappings=("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"),
        guidance="If the row says attributable to a holder class, put that holder in dimensions.attributable_to.",
    ),
    "total_assets": CanonicalConcept(
        concept_id="total_assets",
        display_name="Total assets",
        category="money",
        default_field_name="total_assets",
        aliases=("total assets", "assets", "资产", "总资产", "资产总额"),
        allowed_statement_types=("balance_sheet",),
        external_mappings=("us-gaap:Assets",),
    ),
    "cash_and_cash_equivalents": CanonicalConcept(
        concept_id="cash_and_cash_equivalents",
        display_name="Cash and cash equivalents",
        category="money",
        default_field_name="cash_and_cash_equivalents",
        aliases=("cash and cash equivalents", "cash equivalents", "cash", "现金及现金等价物", "现金"),
        allowed_statement_types=("balance_sheet", "cash_flow"),
        external_mappings=("us-gaap:CashAndCashEquivalentsAtCarryingValue",),
    ),
    "operating_cash_flow": CanonicalConcept(
        concept_id="operating_cash_flow",
        display_name="Operating cash flow",
        category="money",
        default_field_name="operating_cash_flow",
        aliases=(
            "operating cash flow",
            "net cash provided by operating activities",
            "cash flow from operations",
            "经营现金流",
            "经营活动现金流",
        ),
        allowed_statement_types=("cash_flow",),
        external_mappings=(
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    "total_liabilities": CanonicalConcept(
        concept_id="total_liabilities",
        display_name="Total liabilities",
        category="money",
        default_field_name="total_liabilities",
        aliases=("total liabilities", "liabilities", "负债", "总负债"),
        allowed_statement_types=("balance_sheet",),
        external_mappings=("us-gaap:Liabilities",),
    ),
    "research_and_development": CanonicalConcept(
        concept_id="research_and_development",
        display_name="Research and development",
        category="money",
        default_field_name="research_and_development",
        aliases=("research and development", "r&d", "r d", "research expense", "研发", "研发费用"),
        allowed_statement_types=("income_statement",),
        external_mappings=("us-gaap:ResearchAndDevelopmentExpense",),
    ),
    "basic_eps": CanonicalConcept(
        concept_id="basic_eps",
        display_name="Basic earnings per share",
        category="per_share",
        default_field_name="basic_eps",
        aliases=("basic earnings per share", "basic eps", "basic earnings per common share"),
        allowed_statement_types=("income_statement",),
        external_mappings=("us-gaap:EarningsPerShareBasic",),
    ),
    "diluted_eps": CanonicalConcept(
        concept_id="diluted_eps",
        display_name="Diluted earnings per share",
        category="per_share",
        default_field_name="diluted_eps",
        aliases=("diluted earnings per share", "diluted eps", "diluted earnings per common share"),
        allowed_statement_types=("income_statement",),
        external_mappings=("us-gaap:EarningsPerShareDiluted",),
    ),
    "shares_outstanding": CanonicalConcept(
        concept_id="shares_outstanding",
        display_name="Shares outstanding",
        category="shares",
        default_field_name="shares_outstanding",
        aliases=("shares outstanding", "common shares outstanding", "ordinary shares", "股本", "股份数"),
        external_mappings=("dei:EntityCommonStockSharesOutstanding",),
    ),
    "employees": CanonicalConcept(
        concept_id="employees",
        display_name="Employees",
        category="count",
        default_field_name="employees",
        aliases=("employees", "employee", "headcount", "workforce", "员工数", "员工", "人数"),
    ),
    "text_metric": CanonicalConcept(
        concept_id="text_metric",
        display_name="Query-specific text metric",
        category="generic_numeric",
        default_field_name="text_metric",
        aliases=("text metric", "query specific metric", "narrative metric"),
        guidance=(
            "Use for query-relevant numeric facts from narrative text when no more specific "
            "canonical financial concept applies; keep the exact metric name in raw_label."
        ),
    ),
}


def canonical_concept_ids(*, include_unsupported: bool = False) -> tuple[str, ...]:
    ids = tuple(sorted(CANONICAL_CONCEPTS))
    if include_unsupported:
        return (*ids, UNSUPPORTED_CONCEPT_ID)
    return ids


def get_canonical_concept(concept_id: object) -> CanonicalConcept | None:
    if concept_id is None:
        return None
    normalized = _normalize_key(concept_id)
    return CANONICAL_CONCEPTS.get(normalized)


def resolve_canonical_concept(*values: object) -> CanonicalConcept | None:
    """Resolve an internal concept id, external tag, or label into the closed registry."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalized = _normalize_key(text)
        if normalized == UNSUPPORTED_CONCEPT_ID:
            return None
        if normalized in CANONICAL_CONCEPTS:
            return CANONICAL_CONCEPTS[normalized]
        alias_match = _ALIAS_TO_CONCEPT.get(normalized)
        if alias_match is not None:
            return alias_match
        compact_match = _ALIAS_TO_CONCEPT.get(normalized.replace("_", ""))
        if compact_match is not None:
            return compact_match
        external_match = _EXTERNAL_TO_CONCEPT.get(normalized)
        if external_match is not None:
            return external_match
    return None


def concept_prompt_catalog() -> str:
    lines = []
    for concept_id in canonical_concept_ids():
        concept = CANONICAL_CONCEPTS[concept_id]
        aliases = ", ".join(concept.aliases[:5])
        statements = ", ".join(concept.allowed_statement_types) or "any"
        external = ", ".join(concept.external_mappings[:3]) or "none"
        guidance = f" {concept.guidance}" if concept.guidance else ""
        lines.append(
            f"- {concept.concept_id}: {concept.display_name}; category={concept.category}; "
            f"default_field={concept.default_field_name}; statements={statements}; "
            f"aliases={aliases}; external={external}.{guidance}"
        )
    lines.append(
        f"- {UNSUPPORTED_CONCEPT_ID}: use only for irrelevant facts that should not enter the graph; "
        "for query-relevant narrative metrics, use text_metric instead."
    )
    return "\n".join(lines)


def validate_concept_record(
    concept: CanonicalConcept,
    *,
    unit: str | None,
    dimensions: Mapping[str, object] | None,
    raw_label: str | None,
) -> ConceptValidation:
    warnings: list[str] = []
    errors: list[str] = []
    dims = dimensions or {}
    statement_type = str(dims.get("statement_type") or "").strip()
    if concept.allowed_statement_types and statement_type and statement_type not in concept.allowed_statement_types:
        warnings.append(
            f"concept_statement_type_mismatch:{concept.concept_id}:{statement_type}"
        )

    unit_info = UnitResolver().detect(unit or "")
    if unit and unit_info.unit_category != "unknown" and unit_info.unit_category != concept.category:
        warnings.append(
            f"concept_unit_category_mismatch:{concept.concept_id}:"
            f"expected_{concept.category}:got_{unit_info.unit_category}"
        )

    label = (raw_label or "").casefold()
    if concept.concept_id == "net_income" and "attributable to" in label and not dims.get("attributable_to"):
        errors.append("missing_attributable_to_dimension:net_income")
    if concept.concept_id == "total_net_sales" and dims.get("entity_scope") in {"product", "segment", "region"}:
        warnings.append(f"total_net_sales_used_for_detail_scope:{dims.get('entity_scope')}")
    return ConceptValidation(
        warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(dict.fromkeys(errors)),
    )


def canonical_external_mappings(concept_id: object) -> tuple[str, ...]:
    concept = get_canonical_concept(concept_id)
    return concept.external_mappings if concept is not None else ()


def _normalize_key(value: object) -> str:
    return normalize_identifier(str(value)).casefold()


def _build_alias_index() -> dict[str, CanonicalConcept]:
    aliases: dict[str, CanonicalConcept] = {}
    for concept in CANONICAL_CONCEPTS.values():
        for raw_alias in (concept.display_name, concept.default_field_name, *concept.aliases):
            normalized = _normalize_key(raw_alias)
            aliases.setdefault(normalized, concept)
            aliases.setdefault(normalized.replace("_", ""), concept)
    return aliases


def _build_external_index() -> dict[str, CanonicalConcept]:
    external: dict[str, CanonicalConcept] = {}
    for concept in CANONICAL_CONCEPTS.values():
        for mapping in concept.external_mappings:
            normalized = _normalize_key(mapping)
            external[normalized] = concept
            external[normalized.replace("_", "")] = concept
            if ":" in mapping:
                external[_normalize_key(mapping.split(":", 1)[1])] = concept
    return external


_ALIAS_TO_CONCEPT = _build_alias_index()
_EXTERNAL_TO_CONCEPT = _build_external_index()
