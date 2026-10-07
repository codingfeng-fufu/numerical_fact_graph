from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from graph_numeric.core.attribute_graph import AttributeValueGraph, AttributeValueToken, TokenSource, normalize_identifier


MONEY_UNIT = "million USD"


@dataclass(frozen=True)
class AppleTableSpec:
    table_index: int
    table_name: str
    layout: str
    years: tuple[int, ...] = ()
    columns: tuple[str, ...] = ()
    row_metric_prefix: str | None = None
    entity_prefix: str | None = None


APPLE_2025_TABLE_SPECS: tuple[AppleTableSpec, ...] = (
    AppleTableSpec(15, "net_sales_by_product", "rows_by_year_with_change", years=(2025, 2024, 2023), entity_prefix="product"),
    AppleTableSpec(16, "gross_margin_by_category", "rows_by_year", years=(2025, 2024, 2023), entity_prefix="category"),
    AppleTableSpec(22, "statement_of_operations", "rows_by_year", years=(2025, 2024, 2023), entity_prefix="company"),
    AppleTableSpec(24, "balance_sheet", "rows_by_year", years=(2025, 2024), entity_prefix="company"),
    AppleTableSpec(26, "cash_flow", "rows_by_year", years=(2025, 2024, 2023), entity_prefix="company"),
    AppleTableSpec(
        48,
        "segment_2025",
        "year_by_columns",
        years=(2025,),
        columns=("Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific", "Corporate", "Total"),
        entity_prefix="segment",
    ),
    AppleTableSpec(
        49,
        "segment_2024",
        "year_by_columns",
        years=(2024,),
        columns=("Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific", "Corporate", "Total"),
        entity_prefix="segment",
    ),
    AppleTableSpec(
        50,
        "segment_2023",
        "year_by_columns",
        years=(2023,),
        columns=("Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific", "Corporate", "Total"),
        entity_prefix="segment",
    ),
    AppleTableSpec(51, "net_sales_by_country_group", "rows_by_year", years=(2025, 2024, 2023), entity_prefix="country_group"),
)


def graph_from_apple_10k_html(path: str | Path) -> AttributeValueGraph:
    """Extract a focused Apple 10-K numeric graph from the SEC iXBRL HTML.

    This is intentionally a pilot parser: it keeps the original SEC document as
    source and converts selected real tables into the shared AttributeValueGraph
    interface used by the executor.
    """

    file_path = Path(path)
    soup = BeautifulSoup(file_path.read_text(encoding="ascii", errors="ignore"), "lxml")
    tables = soup.find_all("table")
    tokens: list[AttributeValueToken] = []
    for spec in APPLE_2025_TABLE_SPECS:
        if spec.table_index >= len(tables):
            continue
        rows = _table_rows(tables[spec.table_index])
        if spec.layout in {"rows_by_year", "rows_by_year_with_change"}:
            tokens.extend(_tokens_from_rows_by_year(rows, spec, file_path.name))
        elif spec.layout == "year_by_columns":
            tokens.extend(_tokens_from_year_by_columns(rows, spec, file_path.name))
    return AttributeValueGraph(tuple(tokens), source_name=file_path.name)


def write_apple_10k_extraction(path: str | Path, output: str | Path) -> Path:
    graph = graph_from_apple_10k_html(path)
    payload = {
        "source": str(path),
        "token_count": len(graph.tokens),
        "fields": list(graph.fields),
        "years": list(graph.years),
        "tokens": [_token_to_dict(token) for token in graph.tokens],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _table_rows(table: object) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):  # type: ignore[attr-defined]
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        cleaned = [cell for cell in cells if cell not in {"", "$"}]
        if cleaned:
            rows.append(cleaned)
    return rows


def _tokens_from_rows_by_year(
    rows: list[list[str]],
    spec: AppleTableSpec,
    document_id: str,
) -> list[AttributeValueToken]:
    tokens: list[AttributeValueToken] = []
    for row_index, row in enumerate(rows):
        if len(row) < 2:
            continue
        label = _clean_label(row[0])
        if _skip_label(label):
            continue
        if label.lower().startswith("total ") and spec.table_name not in {
            "statement_of_operations",
            "balance_sheet",
            "cash_flow",
        }:
            continue
        values = (
            _values_from_change_row(row[1:], len(spec.years))
            if spec.layout == "rows_by_year_with_change"
            else _numeric_values(row[1:])
        )
        if len(values) < len(spec.years):
            continue
        metric = _metric_name(label, spec.table_name)
        entity = _entity_name(label, spec)
        for col_index, (year, value) in enumerate(zip(spec.years, values, strict=False)):
            tokens.append(_token(
                document_id=document_id,
                table=spec.table_name,
                row_index=row_index,
                column=str(year),
                entity=entity,
                year=year,
                field=metric,
                value=value,
                excerpt=str(value),
                industry=_industry_for_spec(spec),
            ))
    return tokens


def _tokens_from_year_by_columns(
    rows: list[list[str]],
    spec: AppleTableSpec,
    document_id: str,
) -> list[AttributeValueToken]:
    tokens: list[AttributeValueToken] = []
    year = spec.years[0]
    for row_index, row in enumerate(rows):
        if len(row) < 2:
            continue
        label = _clean_label(row[0])
        if _skip_label(label):
            continue
        values = _numeric_values(row[1:])
        if len(values) < len(spec.columns):
            continue
        metric = _metric_name(label, spec.table_name)
        for column, value in zip(spec.columns, values, strict=False):
            if column == "Corporate" and value == 0:
                continue
            if column == "Total":
                continue
            tokens.append(_token(
                document_id=document_id,
                table=spec.table_name,
                row_index=row_index,
                column=column,
                entity=column,
                year=year,
                field=metric,
                value=value,
                excerpt=str(value),
                industry="segment",
            ))
    return tokens


def _numeric_values(cells: Iterable[str]) -> list[float]:
    values: list[float] = []
    for cell in cells:
        parsed = _parse_number(cell)
        if parsed is not None:
            values.append(parsed)
    return values


def _values_from_change_row(cells: list[str], expected: int) -> list[float]:
    """Extract annual values from rows shaped as value/change/%/value/change/%/value."""
    values: list[float] = []
    index = 0
    while index < len(cells) and len(values) < expected:
        parsed = _parse_number(cells[index])
        if parsed is not None:
            values.append(parsed)
            # Skip change and percent cells for 2025/2024 groups. The final
            # 2023 column has no following change group.
            index += 3 if index + 2 < len(cells) and cells[index + 2] == "%" else 1
        else:
            index += 1
    return values


def _parse_number(text: str) -> float | None:
    stripped = text.strip()
    if not stripped or stripped in {"—", "-", "%"}:
        return None
    negative = "(" in stripped and ")" in stripped
    cleaned = re.sub(r"[^0-9.]", "", stripped)
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def _clean_label(label: str) -> str:
    label = re.sub(r"\(\d+\)", "", label)
    label = re.sub(r"\s+", " ", label).strip(" :")
    return label


def _skip_label(label: str) -> bool:
    lowered = label.lower()
    return (
        not label
        or lowered in {"years ended", "assets", "current assets", "non-current assets"}
        or lowered.endswith(":")
        or "earnings per share" in lowered
        or "shares used" in lowered
        or "liabilities and shareholders" in lowered
        or "commitments and contingencies" in lowered
        or "supplemental cash flow disclosure" in lowered
        or "adjustments to reconcile" in lowered
        or "changes in operating assets" in lowered
        or lowered in {"2025", "2024", "2023"}
        or lowered == "percentage of total net sales"
    )


def _metric_name(label: str, table_name: str) -> str:
    normalized = normalize_identifier(label)
    lowered = label.lower()
    if lowered.startswith("total "):
        return normalize_identifier(label)
    if table_name in {"net_sales_by_product", "net_sales_by_country_group"}:
        return "net_sales"
    if table_name == "gross_margin_by_category":
        return "gross_margin"
    if lowered == "net sales" and table_name.startswith("segment"):
        return "net_sales"
    if lowered == "operating income/(loss)":
        return "operating_income"
    if table_name == "cash_flow" and lowered == "net income":
        return "cash_flow_net_income"
    return normalized


def _entity_name(label: str, spec: AppleTableSpec) -> str:
    if spec.table_name == "statement_of_operations":
        return "Apple"
    if spec.table_name == "balance_sheet":
        return "Apple"
    if spec.table_name == "cash_flow":
        return "Apple"
    if spec.table_name == "operating_expenses":
        return "Apple"
    if label.lower().startswith("total "):
        return "Apple"
    return label


def _industry_for_spec(spec: AppleTableSpec) -> str:
    if spec.table_name.startswith("net_sales_by_product"):
        return "product"
    if spec.table_name.startswith("net_sales_by_region") or spec.table_name.startswith("net_sales_by_country"):
        return "region"
    if spec.table_name.startswith("segment"):
        return "segment"
    return "company"


def _token(
    *,
    document_id: str,
    table: str,
    row_index: int,
    column: str,
    entity: str,
    year: int,
    field: str,
    value: float,
    excerpt: str,
    industry: str,
) -> AttributeValueToken:
    entity_id = f"{normalize_identifier(entity)}:{year}:{normalize_identifier(table)}"
    return AttributeValueToken(
        token_id=f"{entity_id}:{field}:{normalize_identifier(column)}",
        entity_id=entity_id,
        company_name=entity,
        field_name=field,
        field_label=field.replace("_", " "),
        value=float(value),
        year=year,
        industry=industry,
        unit=MONEY_UNIT,
        source=TokenSource(
            document_id=document_id,
            table=table,
            row=row_index,
            column=column,
            text_excerpt=excerpt,
        ),
    )


def _token_to_dict(token: AttributeValueToken) -> dict[str, object]:
    source = token.source
    return {
        "token_id": token.token_id,
        "entity_id": token.entity_id,
        "company_name": token.company_name,
        "field_name": token.field_name,
        "canonical_concept_id": token.canonical_concept_id,
        "dimensions": dict(token.dimensions or {}),
        "raw_label": token.raw_label,
        "external_concept_ids": list(token.external_concept_ids),
        "value": token.value,
        "year": token.year,
        "industry": token.industry,
        "unit": token.unit,
        "source": {
            "document_id": source.document_id,
            "table": source.table,
            "row": source.row,
            "column": source.column,
            "text_excerpt": source.text_excerpt,
        } if source else None,
    }
