from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Mapping


IDENTITY_FIELDS = {"company_name", "industry", "year", "document_id"}
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "company_name": ("company", "company name", "company_name", "name", "entity"),
    "industry": ("industry", "sector", "segment", "行业"),
    "year": ("year", "fiscal year", "fy", "年份"),
    "revenue": ("revenue", "sales", "turnover", "收入", "营收", "营业收入"),
    "total_net_sales": ("total net sales", "total_net_sales", "net sales"),
    "net_sales": ("net sales", "sales by product", "sales by segment"),
    "gross_margin": ("gross margin", "gross profit", "毛利", "毛利润"),
    "operating_income": (
        "operating income",
        "operating_income",
        "operating income/(loss)",
    ),
    "operating_profit": (
        "operating profit",
        "operating_profit",
        "operating income",
        "营业利润",
        "经营利润",
    ),
    "net_income": ("net income", "net_income", "net earnings", "profit loss", "净利润", "净收益"),
    "net_profit": ("net profit", "net_profit", "profit after tax", "pat", "净利润", "净收益"),
    "accounts_receivable": (
        "accounts receivable",
        "net accounts receivable",
        "trade receivables",
        "receivables",
        "应收账款",
    ),
    "deferred_revenue": (
        "deferred revenue",
        "unearned revenue",
        "accrued revenue",
        "递延收入",
        "未实现收入",
    ),
    "valuation_allowance": ("valuation allowance", "allowance for valuation", "估值备抵"),
    "unrecognized_tax_benefits": (
        "unrecognized tax benefits",
        "uncertain tax positions",
        "tax positions",
        "未确认税收优惠",
    ),
    "capitalized_interest": ("capitalized interest", "interest capitalized", "资本化利息"),
    "employee_contributions": (
        "employee contributions",
        "company contributions",
        "contributions to the profit sharing and other savings plans",
        "employee stock purchase plan",
    ),
    "restricted_stock": ("restricted stock", "restricted shares", "stock awards", "受限股票"),
    "share_based_compensation": (
        "share based compensation",
        "share-based compensation",
        "stock-based compensation",
        "equity awards",
    ),
    "gross_carrying_value": ("gross carrying value", "gross book value", "gross amount"),
    "non_vested_shares": ("nonvested shares", "non vested shares", "unvested shares"),
    "long_term_debt": ("long-term debt", "long term debt", "debt due", "长期债务"),
    "operating_lease_payments": ("operating lease payments", "operating leases", "租赁付款"),
    "capital_lease_payments": ("capital lease payments", "capital leases", "融资租赁付款"),
    "prepaid_expenses": ("prepaid expenses", "prepaids", "prepaid", "预付费用"),
    "goodwill": ("goodwill", "商誉"),
    "employees": (
        "employees",
        "employee",
        "headcount",
        "staff",
        "workforce",
        "员工数",
        "员工",
        "人数",
        "雇员",
    ),
    "total_assets": ("total assets", "assets", "asset base", "总资产", "资产总额"),
    "total_liabilities": ("total liabilities", "liabilities", "debt obligations", "负债", "总负债"),
    "liabilities": ("liabilities", "total liabilities", "debt obligations", "负债", "总负债"),
    "research_and_development": ("research and development", "r&d", "r d", "research expense", "研发费用"),
    "rd_expense": ("r&d expense", "research expense", "research and development", "研发费用"),
    "marketing_spend": ("marketing spend", "marketing expense", "advertising spend", "营销费用"),
    "store_count": ("store count", "stores", "number of stores", "门店数"),
    "customer_count": ("customer count", "customers", "active customers", "客户数"),
    "cash_flow": ("cash flow", "operating cash flow", "现金流", "经营现金流"),
    "cash_and_cash_equivalents": ("cash and cash equivalents", "cash equivalents", "cash", "现金"),
    "operating_cash_flow": (
        "operating cash flow",
        "net cash provided by operating activities",
        "cash flows from operations",
        "cash flow from operations",
        "经营现金流",
    ),
    "inventory": ("inventory", "stock on hand", "存货", "库存"),
    "capex": ("capex", "capital expenditure", "capital spend", "资本开支"),
    "ebitda": (
        "ebitda",
        "adjusted ebitda",
        "operating ebitda",
        "earnings before interest tax depreciation amortization",
    ),
    "loan_balance": ("loan balance", "outstanding loans", "credit book", "贷款余额"),
    "shares_available_for_future_issuance": (
        "shares available for future issuance",
        "available for future issuance",
        "future issuance",
        "shares of common stock available for future issuance",
    ),
    "new_stapled_securities_issued": (
        "new stapled securities issued",
        "new stapled securities",
        "stapled securities issued",
        "issue of stapled securities",
        "issued stapled securities",
    ),
}


@dataclass(frozen=True)
class TokenSource:
    document_id: str | None = None
    page: int | None = None
    table: str | None = None
    row: int | None = None
    column: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    text_excerpt: str | None = None


@dataclass(frozen=True)
class AttributeValueToken:
    token_id: str
    entity_id: str
    company_name: str
    field_name: str
    field_label: str
    value: float
    year: int | None = None
    industry: str | None = None
    source: TokenSource | None = None
    unit: str | None = None
    canonical_concept_id: str | None = None
    dimensions: Mapping[str, object] | None = None
    table_id: str | None = None
    row_id: int | None = None
    col_id: int | None = None
    is_aggregate: bool = False
    provenance_channel: str | None = None
    raw_label: str | None = None
    external_concept_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttributeValueGraph:
    tokens: tuple[AttributeValueToken, ...]
    source_name: str | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted({token.field_name for token in self.tokens}))

    @property
    def industries(self) -> tuple[str, ...]:
        return tuple(sorted({token.industry for token in self.tokens if token.industry}))

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted({token.year for token in self.tokens if token.year is not None}))

    def select(
        self,
        *,
        field_name: str | None = None,
        year: int | None = None,
        industry: str | None = None,
    ) -> tuple[AttributeValueToken, ...]:
        selected = self.tokens
        if field_name is not None:
            selected = tuple(token for token in selected if token.field_name == field_name)
        if year is not None:
            selected = tuple(token for token in selected if token_matches_year(token, year))
        if industry is not None:
            selected = tuple(token for token in selected if token.industry == industry)
        return selected

    def filter_records(
        self,
        field: str | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> tuple[AttributeValueToken, ...]:
        """Return tokens matching a field and normalized graph filters.

        Supported filter keys intentionally mirror OperatorPlan slots:
        year/time, industry, entity/company/company_name/entity_id and unit.
        Unknown keys are ignored so higher-level solvers can pass broader
        filter maps without coupling to every graph implementation detail.
        """
        selected = self.tokens
        if field is not None:
            selected = tuple(token for token in selected if token.field_name == field)

        for key, value in (filters or {}).items():
            if value is None:
                continue
            key_normalized = normalize_identifier(str(key))
            if key_normalized in {"year", "time", "fiscal_year"}:
                try:
                    selected = tuple(token for token in selected if token_matches_year(token, int(value)))
                except (TypeError, ValueError):
                    selected = ()
            elif key_normalized == "industry":
                selected = tuple(token for token in selected if token.industry == str(value))
            elif key_normalized in {"entity", "company", "company_name", "entity_id"}:
                selected = tuple(token for token in selected if _matches_entity(token, str(value)))
            elif key_normalized == "unit":
                selected = tuple(token for token in selected if token.unit == str(value))
        return selected

    def lookup(
        self,
        field: str,
        filters: Mapping[str, object] | None = None,
    ) -> tuple[AttributeValueToken, ...]:
        return self.filter_records(field=field, filters=filters)

    def get_candidate_fields(self) -> tuple[str, ...]:
        return self.fields

    def get_field_profile(self, field: str) -> dict[str, object]:
        records = self.select(field_name=field)
        values = [record.value for record in records]
        units = sorted({record.unit for record in records if record.unit})
        if not values:
            return {
                "field": field,
                "count": 0,
                "years": (),
                "industries": (),
                "unit": None,
            }
        return {
            "field": field,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "years": tuple(sorted({record.year for record in records if record.year is not None})),
            "industries": tuple(sorted({record.industry for record in records if record.industry})),
            "unit": units[0] if len(units) == 1 else None,
            "units": tuple(units),
        }

    def get_records_by_entity(self, entity: str) -> tuple[AttributeValueToken, ...]:
        return tuple(token for token in self.tokens if _matches_entity(token, entity))


def graph_from_markdown_file(path: str | Path) -> AttributeValueGraph:
    file_path = Path(path)
    return graph_from_markdown_table(file_path.read_text(encoding="utf-8"), source_name=file_path.name)


def graph_from_document_file(path: str | Path) -> AttributeValueGraph:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return graph_from_csv_text(text, source_name=file_path.name)
    if suffix == ".json":
        return graph_from_json_text(text, source_name=file_path.name)
    if suffix in {".md", ".markdown"}:
        return graph_from_markdown_table(text, source_name=file_path.name)
    return graph_from_text_table(text, source_name=file_path.name)


def graph_from_csv_text(text: str, source_name: str | None = None) -> AttributeValueGraph:
    rows = list(csv.DictReader(text.splitlines()))
    return _graph_from_row_dicts(
        rows,
        source_name=source_name,
        table=source_name,
        text=text,
    )


def graph_from_json_text(text: str, source_name: str | None = None) -> AttributeValueGraph:
    payload = json.loads(text)
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            rows = payload["records"]
        elif isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload.get("graph"), dict):
            nodes = payload.get("graph", {}).get("nodes", [])
            rows = [
                {
                    "company_name": node.get("name"),
                    **(node.get("attributes") or {}),
                }
                for node in nodes
            ]
        else:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return _graph_from_row_dicts(
        [row for row in rows if isinstance(row, dict)],
        source_name=source_name,
        table=source_name,
        text=text,
    )


def graph_from_text_table(text: str, source_name: str | None = None) -> AttributeValueGraph:
    markdown = graph_from_markdown_table(text, source_name=source_name)
    if markdown.tokens:
        return markdown

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return AttributeValueGraph((), source_name=source_name)
    delimiter = "," if "," in lines[0] else None
    if delimiter is not None:
        return graph_from_csv_text(text, source_name=source_name)

    header = re.split(r"\s{2,}|\t", lines[0].strip())
    if len(header) < 2:
        return AttributeValueGraph((), source_name=source_name)
    rows: list[dict[str, object]] = []
    for line in lines[1:]:
        cells = re.split(r"\s{2,}|\t", line.strip())
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=True)))
    return _graph_from_row_dicts(rows, source_name=source_name, table=source_name, text=text)


def graph_from_markdown_table(text: str, source_name: str | None = None) -> AttributeValueGraph:
    table_cells = _markdown_table_cells(text)
    respectively_graph = _graph_from_respectively_sentences(
        text,
        source_name=source_name,
        table=source_name,
    )
    future_issuance_graph = _graph_from_future_issuance_sentences(
        text,
        source_name=source_name,
        table=source_name,
    )
    segment_sales_graph = _graph_from_segment_net_sales_sentences(
        text,
        source_name=source_name,
        table=source_name,
    )
    new_stapled_graph = _graph_from_new_stapled_securities_sentences(
        text,
        table_cells=table_cells,
        source_name=source_name,
        table=source_name,
    )
    sentence_graph = _merge_graph_tokens(
        _merge_graph_tokens(
            respectively_graph,
            future_issuance_graph,
            source_name=source_name,
        ),
        _merge_graph_tokens(
            segment_sales_graph,
            new_stapled_graph,
            source_name=source_name,
        ),
        source_name=source_name,
    )
    grouped_year_graph = _graph_from_grouped_year_rows(
        table_cells,
        source_name=source_name,
        table=source_name,
        text=text,
    )
    if grouped_year_graph is not None:
        return _merge_graph_tokens(grouped_year_graph, sentence_graph, source_name=source_name)
    cross_graph = _graph_from_year_column_cross_table_cells(
        table_cells,
        source_name=source_name,
        table=source_name,
        text=text,
    )
    if cross_graph is not None:
        return _merge_graph_tokens(cross_graph, sentence_graph, source_name=source_name)
    multirow_year_graph = _graph_from_multirow_column_year_table(
        table_cells,
        source_name=source_name,
        table=source_name,
        text=text,
    )
    if multirow_year_graph is not None:
        return _merge_graph_tokens(multirow_year_graph, sentence_graph, source_name=source_name)
    embedded_header_graph = _graph_from_embedded_header_table(
        table_cells,
        source_name=source_name,
        table=source_name,
        text=text,
    )
    if embedded_header_graph is not None:
        return _merge_graph_tokens(embedded_header_graph, sentence_graph, source_name=source_name)
    rows = _parse_markdown_rows(text)
    row_graph = _graph_from_row_dicts(rows, source_name=source_name, table=source_name, text=text)
    return _merge_graph_tokens(row_graph, sentence_graph, source_name=source_name)


def normalize_identifier(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def field_aliases(field_name: str) -> tuple[str, ...]:
    return FIELD_ALIASES.get(field_name, (_field_label(field_name), field_name.replace("_", " ")))


def _merge_graph_tokens(
    primary: AttributeValueGraph,
    secondary: AttributeValueGraph,
    *,
    source_name: str | None,
) -> AttributeValueGraph:
    if not secondary.tokens:
        return primary
    if not primary.tokens:
        return secondary
    seen = {token.token_id for token in primary.tokens}
    tokens = list(primary.tokens)
    for token in secondary.tokens:
        token_id = token.token_id
        if token_id in seen:
            token_id = f"sentence:{token_id}"
            token = AttributeValueToken(
                token_id=token_id,
                entity_id=token.entity_id,
                company_name=token.company_name,
                field_name=token.field_name,
                field_label=token.field_label,
                value=token.value,
                year=token.year,
                industry=token.industry,
                source=token.source,
                unit=token.unit,
                canonical_concept_id=token.canonical_concept_id,
                dimensions=token.dimensions,
                raw_label=token.raw_label,
                external_concept_ids=token.external_concept_ids,
            )
        seen.add(token_id)
        tokens.append(token)
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def token_matches_year(token: AttributeValueToken, year: object) -> bool:
    try:
        expected = int(year)
    except (TypeError, ValueError):
        return False
    if token.year == expected:
        return True
    if token.year is not None:
        return False
    return any(
        _text_contains_year(value, expected)
        for value in (
            token.field_name,
            token.field_label,
            token.source.column if token.source is not None else None,
        )
    )


def _text_contains_year(value: object | None, year: int) -> bool:
    if value is None:
        return False
    return re.search(rf"(?<!\d){year}(?!\d)", str(value)) is not None


def _graph_from_respectively_sentences(
    text: str,
    *,
    source_name: str | None,
    table: str | None,
) -> AttributeValueGraph:
    if "respectively" not in text.lower():
        return AttributeValueGraph((), source_name=source_name)
    tokens: list[AttributeValueToken] = []
    patterns = (
        re.compile(
            r"(?P<label>[A-Z][^.]*?)\s+during\s+"
            r"(?P<years>(?:19|20)\d{2}(?:\s*,\s*(?:19|20)\d{2})*(?:\s+and\s+(?:19|20)\d{2})?)"
            r"\s*,?\s+was\s+(?P<values>.*?)\s*,?\s+respectively",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<label>[A-Z][^.]*?)\s+for\s+(?:the\s+)?years?\s+ended\s+[^.]*?"
            r"(?P<years>(?:19|20)\d{2}(?:\s*,\s*(?:19|20)\d{2})*(?:\s+and\s+(?:19|20)\d{2})?)"
            r"\s+(?:was|were)\s+(?P<values>.*?)\s*,?\s+respectively",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<label>[A-Z][^.]*?)\s+amounting\s+to\s+(?P<values>.*?)\s+"
            r"at\s+[^.]*?"
            r"(?P<years>(?:19|20)\d{2}(?:\s*,\s*(?:19|20)\d{2})*(?:\s+and\s+(?:19|20)\d{2})?)"
            r"\s*,?\s+respectively",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<label>[A-Z][^.]*?)\s+(?:was|were)\s+(?P<values>.*?)\s+"
            r"for\s+(?:the\s+)?years?\s+ended\s+[^.]*?"
            r"(?P<years>(?:19|20)\d{2}(?:\s*,\s*(?:19|20)\d{2})*(?:\s+and\s+(?:19|20)\d{2})?)"
            r"\s*,?\s+respectively",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<label>[A-Z][^.]*?)\s+(?:was|were)\s+(?P<values>.*?)\s+"
            r"in\s+(?P<years>(?:19|20)\d{2}(?:\s*,\s*(?:19|20)\d{2})*(?:\s+and\s+(?:19|20)\d{2})?)"
            r"\s*,?\s+respectively",
            flags=re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            tokens.extend(
                _respectively_tokens_from_match(
                    match,
                    source_name=source_name,
                    table=table,
                )
            )
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _respectively_tokens_from_match(
    match: re.Match[str],
    *,
    source_name: str | None,
    table: str | None,
) -> list[AttributeValueToken]:
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", match.group("years"))]
    value_matches = list(
        re.finditer(
            r"[$€£]?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand)?",
            match.group("values"),
            flags=re.IGNORECASE,
        )
    )
    if len(years) != len(value_matches) or not years:
        return []
    raw_label = match.group("label").strip()
    field_name = _canonical_header(_respectively_field_label(raw_label))
    tokens: list[AttributeValueToken] = []
    for year, value_match in zip(years, value_matches, strict=True):
        value = float(value_match.group(1).replace(",", ""))
        unit = value_match.group(2).lower() if value_match.group(2) else None
        entity_id = str(year)
        source = TokenSource(
            document_id=source_name,
            table=table,
            row=None,
            column=str(year),
            char_start=match.start(),
            char_end=match.end(),
            text_excerpt=match.group(0),
        )
        tokens.append(
            AttributeValueToken(
                token_id=f"{entity_id}:{field_name}",
                entity_id=entity_id,
                company_name=entity_id,
                field_name=field_name,
                field_label=_field_label(field_name),
                value=value,
                year=year,
                source=source,
                unit=unit,
                dimensions={"sentence_pattern": "respectively"},
                raw_label=raw_label,
            )
        )
    return tokens


def _graph_from_future_issuance_sentences(
    text: str,
    *,
    source_name: str | None,
    table: str | None,
) -> AttributeValueGraph:
    tokens: list[AttributeValueToken] = []
    current_plan: str | None = None
    fact_index = 0
    for sentence_match in re.finditer(r"[^.]+(?:\.|$)", text):
        sentence = sentence_match.group(0).strip()
        if not sentence:
            continue
        plan = _future_issuance_plan_name(sentence)
        if plan is not None:
            current_plan = plan
        fact = re.search(
            r"\bthere\s+were\s*"
            r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s+"
            r"shares\s+of\s+common\s+stock\s+available\s+for\s+future\s+issuance\s+"
            r"under\s+this\s+plan\s+as\s+of\s+(?P<date>[^.]+)",
            sentence,
            flags=re.IGNORECASE,
        )
        if fact is None or current_plan is None:
            continue
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", fact.group("date"))
        year = int(year_match.group(1)) if year_match else None
        value = float(fact.group("value").replace(",", ""))
        field_name = "shares_available_for_future_issuance"
        entity_id = _entity_id(current_plan, year, fact_index)
        source = TokenSource(
            document_id=source_name,
            table=table,
            row=None,
            column="available for future issuance",
            char_start=sentence_match.start() + fact.start(),
            char_end=sentence_match.start() + fact.end(),
            text_excerpt=fact.group(0),
        )
        tokens.append(
            AttributeValueToken(
                token_id=f"{entity_id}:{field_name}",
                entity_id=entity_id,
                company_name=current_plan,
                field_name=field_name,
                field_label="shares available for future issuance",
                value=value,
                year=year,
                source=source,
                unit="shares",
                dimensions={
                    "plan": current_plan,
                    "row_label": current_plan,
                    "date": fact.group("date").strip(),
                    "sentence_pattern": "future_issuance",
                },
                raw_label=current_plan,
            )
        )
        fact_index += 1
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _future_issuance_plan_name(sentence: str) -> str | None:
    match = re.search(
        r"\b(Amended\s+and\s+Restated\s+\d{4}\s+Stock\s+Incentive\s+Plan)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if match:
        return " ".join(match.group(1).split())
    if re.search(r"\bdeferred\s+compensation\s+plan\b", sentence, flags=re.IGNORECASE):
        return "deferred compensation plan"
    return None


def _graph_from_segment_net_sales_sentences(
    text: str,
    *,
    source_name: str | None,
    table: str | None,
) -> AttributeValueGraph:
    if "net sales" not in text.lower():
        return AttributeValueGraph((), source_name=source_name)
    tokens: list[AttributeValueToken] = []
    header_pattern = re.compile(
        r"(?P<segment>[A-Za-z][A-Za-z0-9&.,' -]{1,80}?)\s+net\s+sales\s+"
        r"for\s+(?P<first_year>19\d{2}|20\d{2})\s+were\s+",
        flags=re.IGNORECASE,
    )
    for header in header_pattern.finditer(text):
        sentence_start = header.start()
        sentence_end = text.find(".", header.end())
        if sentence_end < 0:
            sentence_end = len(text)
        else:
            sentence_end += 1
        sentence = text[sentence_start:sentence_end].strip()
        sentence_offset = sentence_start
        segment = _clean_segment_label(header.group("segment"))
        facts: list[tuple[int, float, str, int, int]] = []
        value_after_header = re.search(
            r"were\s+(?P<value>[$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>million|billion|thousand)?",
            text[header.start():sentence_end],
            flags=re.IGNORECASE,
        )
        if value_after_header is not None:
            facts.append((
                int(header.group("first_year")),
                _number_from_text(value_after_header.group("value")),
                _unit_from_value_and_scale(value_after_header.group("value"), value_after_header.group("unit")),
                header.start() + value_after_header.start(),
                header.start() + value_after_header.end(),
            ))
        for fact in re.finditer(
            r"(?P<value>[$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>million|billion|thousand)?\s+"
            r"(?:in|for)\s+(?P<year>19\d{2}|20\d{2})",
            sentence,
            flags=re.IGNORECASE,
        ):
            facts.append((
                int(fact.group("year")),
                _number_from_text(fact.group("value")),
                _unit_from_value_and_scale(fact.group("value"), fact.group("unit")),
                sentence_offset + fact.start(),
                sentence_offset + fact.end(),
            ))
        seen: set[tuple[int, float]] = set()
        for year, value, unit, char_start, char_end in facts:
            key = (year, value)
            if key in seen:
                continue
            seen.add(key)
            raw_label = f"{segment} net sales"
            entity_id = f"{normalize_identifier(segment)}:{year}"
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:revenue",
                    entity_id=entity_id,
                    company_name=segment,
                    field_name="revenue",
                    field_label="net sales",
                    value=value,
                    year=year,
                    source=TokenSource(
                        document_id=source_name,
                        table=table,
                        row=None,
                        column=str(year),
                        char_start=char_start,
                        char_end=char_end,
                        text_excerpt=sentence,
                    ),
                    unit=unit,
                    dimensions={
                        "segment": segment,
                        "row_label": raw_label,
                        "sentence_pattern": "segment_net_sales",
                    },
                    raw_label=raw_label,
                )
            )
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _clean_segment_label(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,.;")
    cleaned = re.sub(r"^(?:and|the|this|business|u\s*s)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.lower()


def _number_from_text(value: str) -> float:
    return float(re.sub(r"[^0-9.+-]", "", value))


def _unit_from_value_and_scale(value: str, scale: str | None) -> str | None:
    prefix = "$" if "$" in value else ("€" if "€" in value else ("£" if "£" in value else ""))
    if scale:
        return f"{prefix} {scale.lower()}".strip()
    return prefix or None


def _graph_from_new_stapled_securities_sentences(
    text: str,
    *,
    table_cells: list[list[str]],
    source_name: str | None,
    table: str | None,
) -> AttributeValueGraph:
    if "stapled securities" not in text.lower():
        return AttributeValueGraph((), source_name=source_name)
    current_year = _first_year_column_value(table_cells)
    tokens: list[AttributeValueToken] = []
    field_name = "new_stapled_securities_issued"
    for fact_index, match in enumerate(
        re.finditer(
            r"\bresulted\s+in\s+the\s+issue\s+of\s+"
            r"(?P<current_value>[-+]?\d[\d,]*(?:\.\d+)?)\s+"
            r"new\s+stapled\s+securities\s*"
            r"\(\s*(?P<prior_year>19\d{2}|20\d{2})\s*:\s*"
            r"[^)]*?\bissue\s+of\s+"
            r"(?P<prior_value>[-+]?\d[\d,]*(?:\.\d+)?)\s+"
            r"stapled\s+securities",
            text,
            flags=re.IGNORECASE,
        )
    ):
        prior_year = int(match.group("prior_year"))
        resolved_current_year = (
            current_year
            if current_year is not None and current_year != prior_year
            else None
        )
        values: list[tuple[int | None, str, str]] = [
            (resolved_current_year, match.group("current_value"), "current"),
            (prior_year, match.group("prior_value"), "prior"),
        ]
        for year, raw_value, period_role in values:
            value = float(raw_value.replace(",", ""))
            entity_id = str(year) if year is not None else f"new_stapled:{fact_index}:{period_role}"
            source = TokenSource(
                document_id=source_name,
                table=table,
                row=None,
                column=str(year) if year is not None else period_role,
                char_start=match.start(),
                char_end=match.end(),
                text_excerpt=match.group(0),
            )
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=entity_id,
                    field_name=field_name,
                    field_label="new stapled securities issued",
                    value=value,
                    year=year,
                    source=source,
                    unit="securities",
                    dimensions={
                        "sentence_pattern": "new_stapled_securities",
                        "period_role": period_role,
                    },
                    raw_label="new stapled securities issued",
                )
            )
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _first_year_column_value(table_rows: list[list[str]]) -> int | None:
    header = _find_year_column_header(table_rows)
    if header is None:
        return None
    year_columns, _ = header
    return year_columns[0][1] if year_columns else None


def _respectively_field_label(raw_label: str) -> str:
    label = re.sub(r"^\s*(the|a|an)\s+", "", raw_label.strip(), flags=re.IGNORECASE)
    label = re.sub(r"\bthat\b", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def _parse_markdown_rows(text: str) -> list[dict[str, object]]:
    table_rows = _markdown_table_cells(text)
    if len(table_rows) < 2:
        return []
    header = table_rows[0]
    canonical_header = [_canonical_header(cell) for cell in header]
    rows = []
    for cells in table_rows[2:]:
        if len(cells) != len(canonical_header):
            continue
        row = {}
        for key, raw_value in zip(canonical_header, cells, strict=True):
            row[key] = raw_value
        rows.append(row)
    return rows


def _markdown_table_cells(text: str) -> list[list[str]]:
    return [
        _split_markdown_row(line.strip())
        for line in text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]


def _graph_from_grouped_year_rows(
    table_rows: list[list[str]],
    *,
    source_name: str | None,
    table: str | None,
    text: str | None,
) -> AttributeValueGraph | None:
    value_header = _find_grouped_year_value_header(table_rows)
    if value_header is None:
        return None
    header_index, column_labels = value_header
    current_year: int | None = None
    tokens: list[AttributeValueToken] = []
    for row_index, cells in enumerate(table_rows[header_index + 1:], start=1):
        if _is_markdown_separator_row(cells):
            continue
        year = _group_year_from_row(cells)
        if year is not None:
            current_year = year
            continue
        if current_year is None or len(cells) < 2:
            continue
        raw_label = cells[0].strip()
        if not raw_label or raw_label.endswith(":"):
            continue
        for column_index, raw_value in enumerate(cells[1:], start=1):
            parsed = _parse_numeric_and_unit(raw_value)
            if parsed is None:
                continue
            column_label = column_labels.get(column_index) or f"value_{column_index}"
            field_name = _canonical_grouped_year_field(raw_label, column_label)
            value, unit = parsed
            entity_id = str(current_year)
            src = _source_for_cell(
                source_name=source_name,
                table=table,
                row_index=row_index,
                column=column_label,
                raw_value=raw_value,
                text=text,
            )
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=entity_id,
                    industry=None,
                    year=current_year,
                    field_name=field_name,
                    field_label=f"{raw_label} {column_label}",
                    value=float(value),
                    source=src,
                    unit=unit,
                    dimensions={"row_label": raw_label, "column_label": column_label},
                    raw_label=raw_label,
                )
            )
    if not tokens:
        return None
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _find_grouped_year_value_header(
    table_rows: list[list[str]],
) -> tuple[int, dict[int, str]] | None:
    first_group_year_index = next(
        (index for index, row in enumerate(table_rows) if _group_year_from_row(row) is not None),
        None,
    )
    if first_group_year_index is None:
        return None
    for row_index, cells in enumerate(table_rows[:first_group_year_index]):
        if _is_markdown_separator_row(cells) or _group_year_from_row(cells) is not None:
            continue
        labels: dict[int, str] = {}
        for column_index, cell in enumerate(cells[1:], start=1):
            label = cell.strip()
            if not label:
                continue
            labels[column_index] = label
        if len(labels) < 2:
            continue
        return row_index, labels
    return None


def _group_year_from_row(cells: list[str]) -> int | None:
    if not cells:
        return None
    first = cells[0].strip()
    if not first:
        return None
    if any(cell.strip() for cell in cells[1:]):
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", first)
    if match is None:
        return None
    return int(match.group(1))


def _canonical_grouped_year_field(raw_label: str, column_label: str) -> str:
    label = _canonical_header(raw_label)
    column = _canonical_header(column_label)
    if column in {"total", "total_fair_value"}:
        return normalize_identifier(f"{label}_{column}")
    return normalize_identifier(f"{label}_{column}")


def _graph_from_embedded_header_table(
    table_rows: list[list[str]],
    *,
    source_name: str | None,
    table: str | None,
    text: str | None,
) -> AttributeValueGraph | None:
    header_index = _find_embedded_header_index(table_rows)
    if header_index is None:
        return None
    header = table_rows[header_index]
    unit = _embedded_table_unit(table_rows[:header_index])
    current_section: str | None = None
    tokens: list[AttributeValueToken] = []
    for row_index, cells in enumerate(table_rows[header_index + 1:], start=1):
        if _is_markdown_separator_row(cells) or len(cells) < 2:
            continue
        raw_label = cells[0].strip()
        if not raw_label:
            continue
        if _is_named_table_section_row(cells):
            current_section = _clean_named_entity_label(raw_label)
            continue
        company = _clean_named_entity_label(raw_label)
        dimensions = {"row_label": company}
        if _is_named_table_aggregate_row(raw_label):
            dimensions["is_total_row"] = "true"
            dimensions["is_aggregate"] = "true"
        elif current_section:
            dimensions["section"] = current_section
            entity_type = _entity_type_from_section(current_section)
            if entity_type:
                dimensions["entity_type"] = entity_type
        for column_index, raw_value in enumerate(cells[1:], start=1):
            if column_index >= len(header):
                continue
            field_label = header[column_index].strip()
            if not field_label:
                continue
            parsed = _parse_numeric_and_unit(raw_value)
            if parsed is None:
                continue
            value, parsed_unit = parsed
            field_name = _canonical_header(field_label)
            entity_id = _entity_id(company, None, row_index)
            src = _source_for_cell(
                source_name=source_name,
                table=table,
                row_index=row_index,
                column=field_label,
                raw_value=raw_value,
                text=text,
            )
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=company,
                    industry=None,
                    year=None,
                    field_name=field_name,
                    field_label=field_label,
                    value=float(value),
                    source=src,
                    unit=parsed_unit or unit,
                    dimensions=dimensions,
                    raw_label=raw_label,
                )
            )
    if not tokens:
        return None
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _find_embedded_header_index(table_rows: list[list[str]]) -> int | None:
    seen_separator = False
    for row_index, cells in enumerate(table_rows):
        if _is_markdown_separator_row(cells):
            seen_separator = True
            continue
        if not seen_separator or len(cells) < 3:
            continue
        first = cells[0].strip()
        if not re.search(
            r"\b(?:particulars?|description|name|director|remuneration)\b",
            first,
            re.IGNORECASE,
        ):
            continue
        text_headers = [
            cell
            for cell in cells[1:]
            if cell.strip()
            and _parse_numeric_and_unit(cell) is None
            and re.search(r"[A-Za-z\u4e00-\u9fff]", cell)
        ]
        if len(text_headers) >= 2:
            return row_index
    return None


def _embedded_table_unit(rows: list[list[str]]) -> str | None:
    for cells in rows:
        for cell in cells:
            match = re.search(
                r"\b(lakh|crore|million|billion|thousand)\b|[`(]\s*lakh\s*[)]",
                cell,
                re.IGNORECASE,
            )
            if match:
                matched = match.group(0).lower()
                return "lakh" if "lakh" in matched else match.group(0).strip("()` ")
    return None


def _is_named_table_section_row(cells: list[str]) -> bool:
    first = cells[0].strip()
    if not re.match(r"\d+\.\s+\S+", first):
        return False
    return all(_parse_numeric_and_unit(cell) is None for cell in cells[1:] if cell.strip())


def _is_named_table_aggregate_row(label: str) -> bool:
    return bool(re.match(r"\s*(?:total|ceiling)\b", label, re.IGNORECASE))


def _clean_named_entity_label(label: str) -> str:
    cleaned = re.sub(r"^\s*\d+\.\s*", "", label.strip())
    cleaned = re.sub(r"[*@†‡]+$", "", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def _entity_type_from_section(section: str) -> str | None:
    lowered = section.lower()
    if "independent director" in lowered:
        return "independent director"
    if "non-executive director" in lowered:
        return "non-executive director"
    return None


def _graph_from_multirow_column_year_table(
    table_rows: list[list[str]],
    *,
    source_name: str | None,
    table: str | None,
    text: str | None,
) -> AttributeValueGraph | None:
    header = _find_multirow_column_year_header(table_rows)
    if header is None:
        return None
    top_headers, year_columns, data_start, row_label_key = header
    tokens: list[AttributeValueToken] = []
    max_column = max(column_index for column_index, _ in year_columns)
    for row_index, cells in enumerate(table_rows[data_start:], start=1):
        if _is_markdown_separator_row(cells) or len(cells) <= max_column:
            continue
        raw_label = cells[0].strip()
        if not raw_label:
            continue
        company = raw_label
        for column_index, year in year_columns:
            top_header = top_headers[column_index].strip()
            if not top_header:
                continue
            raw_value = cells[column_index]
            parsed = _parse_numeric_and_unit(raw_value)
            if parsed is None:
                continue
            value, unit = parsed
            field_name = _canonical_header(top_header)
            entity_id = _entity_id(company, year, row_index)
            src = _source_for_cell(
                source_name=source_name,
                table=table,
                row_index=row_index,
                column=f"{top_header} {year}",
                raw_value=raw_value,
                text=text,
            )
            dimensions = {
                "row_label": raw_label,
                "column_label": top_header,
                row_label_key: raw_label,
            }
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=company,
                    industry=None,
                    year=year,
                    field_name=field_name,
                    field_label=top_header,
                    value=float(value),
                    source=src,
                    unit=unit,
                    dimensions=dimensions,
                    raw_label=raw_label,
                )
            )
    if not tokens:
        return None
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _find_multirow_column_year_header(
    table_rows: list[list[str]],
) -> tuple[list[str], list[tuple[int, int]], int, str] | None:
    for top_index, top_cells in enumerate(table_rows[:-2]):
        if _is_markdown_separator_row(top_cells):
            continue
        separator_index = top_index + 1
        if separator_index >= len(table_rows) or not _is_markdown_separator_row(table_rows[separator_index]):
            continue
        subheader_index = separator_index + 1
        if subheader_index >= len(table_rows):
            continue
        subheader = table_rows[subheader_index]
        if len(subheader) != len(top_cells) or not subheader:
            continue
        row_label = subheader[0].strip()
        if not row_label or _parse_numeric_and_unit(row_label) is not None:
            continue
        top_headers = _forward_fill_header_cells(top_cells)
        year_columns: list[tuple[int, int]] = []
        for column_index, cell in enumerate(subheader[1:], start=1):
            year = _year_from_header_cell(cell)
            if year is not None and top_headers[column_index].strip():
                year_columns.append((column_index, year))
        if len(year_columns) >= 2:
            return top_headers, year_columns, subheader_index + 1, _canonical_header(row_label)
    return None


def _forward_fill_header_cells(cells: list[str]) -> list[str]:
    filled: list[str] = []
    current = ""
    for cell in cells:
        value = cell.strip()
        if value:
            current = value
        filled.append(current)
    return filled


def _graph_from_year_column_cross_table_cells(
    table_rows: list[list[str]],
    *,
    source_name: str | None,
    table: str | None,
    text: str | None,
) -> AttributeValueGraph | None:
    if len(table_rows) < 3:
        return None
    header = _find_year_column_header(table_rows)
    if header is None:
        return None
    year_columns, data_start = header
    if not year_columns:
        return None

    context = _markdown_context_before_first_table(text)
    table_unit = _context_unit(context)
    tokens: list[AttributeValueToken] = []
    last_segment_label: str | None = None
    for row_index, cells in enumerate(table_rows[data_start:], start=1):
        if _is_markdown_separator_row(cells):
            continue
        if len(cells) <= max(column_index for column_index, _ in year_columns):
            continue
        raw_label = cells[0].strip()
        if not raw_label:
            raw_label = _blank_cross_table_row_label(cells, year_columns, context=context)
            if not raw_label:
                continue
        field_name = _canonical_cross_table_row_label(raw_label, context=context)
        cleaned_label = _clean_cross_table_row_label(raw_label)
        if _is_metric_row_label(cleaned_label):
            segment_label = last_segment_label
        else:
            segment_label = cleaned_label
            last_segment_label = cleaned_label
        for column_index, year in year_columns:
            raw_value = cells[column_index]
            parsed = _parse_numeric_and_unit(raw_value)
            if parsed is None:
                continue
            value, unit = parsed
            unit = _merge_unit_with_context(unit, table_unit)
            entity_id = str(year)
            src = _source_for_cell(
                source_name=source_name,
                table=table,
                row_index=row_index,
                column=str(year),
                raw_value=raw_value,
                text=text,
            )
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=entity_id,
                    industry=None,
                    year=year,
                    field_name=field_name,
                    field_label=raw_label,
                    value=float(value),
                    source=src,
                    unit=unit or table_unit,
                    dimensions={
                        "row_label": raw_label,
                        **({"segment_label": segment_label} if segment_label else {}),
                    },
                    raw_label=raw_label,
                )
            )
    if not tokens:
        return None
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _blank_cross_table_row_label(
    cells: list[str],
    year_columns: list[tuple[int, int]],
    *,
    context: str,
) -> str | None:
    if len(cells) <= max(column_index for column_index, _ in year_columns):
        return None
    numeric_values = [
        _parse_numeric_and_unit(cells[column_index])
        for column_index, _ in year_columns
    ]
    if not any(value is not None for value in numeric_values):
        return None
    context_normalized = normalize_identifier(context)
    if "cost" in context_normalized or "charges" in context_normalized:
        return "Total costs"
    return "Total"


def _find_year_column_header(table_rows: list[list[str]]) -> tuple[list[tuple[int, int]], int] | None:
    for row_index, cells in enumerate(table_rows):
        if _is_markdown_separator_row(cells):
            continue
        year_columns = _year_cells_from_row(cells)
        if year_columns is None:
            continue
        data_start = row_index + 1
        while data_start < len(table_rows) and _is_markdown_separator_row(table_rows[data_start]):
            data_start += 1
        while data_start < len(table_rows) and _is_unit_header_row(table_rows[data_start]):
            data_start += 1
        if data_start < len(table_rows):
            return year_columns, data_start
    return None


def _year_cells_from_row(cells: list[str]) -> list[tuple[int, int]] | None:
    if len(cells) < 2 or not _is_year_header_stub(cells[0]):
        return None
    year_columns: list[tuple[int, int]] = []
    for column_index, cell in enumerate(cells[1:], start=1):
        year = _year_from_header_cell(cell)
        if year is not None:
            year_columns.append((column_index, year))
    return year_columns if year_columns else None


def _year_from_header_cell(cell: str) -> int | None:
    text = str(cell)
    if _looks_like_numeric_data_cell(text):
        return None
    period_match = re.fullmatch(
        r"\s*(19\d{2}|20\d{2})\s*/\s*(19\d{2}|20\d{2})\s*",
        text,
    )
    if period_match is not None:
        return int(period_match.group(2))
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if len(matches) != 1:
        return None
    return int(matches[0])


def _looks_like_numeric_data_cell(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.fullmatch(r"(19\d{2}|20\d{2})", stripped):
        return False
    lowered = stripped.lower()
    if re.search(
        r"\b(?:million|billion|thousand|crore|lakh|usd|eur|cny|rmb|dollars?)\b",
        lowered,
    ):
        return True
    if re.search(r"[$€¥￥₹£%]", stripped):
        return True
    return False


def _is_year_header_stub(cell: str) -> bool:
    stripped = cell.strip()
    if not stripped:
        return True
    normalized = normalize_identifier(stripped)
    if normalized in {"year", "years", "fiscal_year", "fiscal_years"}:
        return True
    if re.search(r"\byears?\s+ended\b|\byear\s+ended\b", stripped, re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"\b(?:in\s+)?(?:millions?|thousands?|billions?)\b",
            stripped,
            re.IGNORECASE,
        )
    )


def _is_markdown_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _is_unit_header_row(cells: list[str]) -> bool:
    if len(cells) < 2 or cells[0].strip():
        return False
    return all(_parse_numeric_and_unit(cell) is None for cell in cells[1:] if cell.strip())


def _markdown_context_before_first_table(text: str | None) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            break
        lines.append(line.strip())
    return " ".join(line for line in lines if line)


def _context_unit(context: str) -> str | None:
    normalized = normalize_identifier(context)
    if "million" in normalized:
        return "$ million"
    if "billion" in normalized:
        return "$ billion"
    if "thousand" in normalized:
        return "$ thousand"
    return None


def _merge_unit_with_context(unit: str | None, context_unit: str | None) -> str | None:
    if context_unit is None:
        return unit
    if unit is None:
        return context_unit
    if unit in {"$", "€", "£", "¥", "￥", "₹"}:
        return context_unit
    return unit


def _canonical_cross_table_row_label(raw_label: str, *, context: str = "") -> str:
    raw_label = _clean_cross_table_row_label(raw_label)
    normalized = normalize_identifier(raw_label)
    if normalized == "total" and context:
        context_normalized = normalize_identifier(context)
        if any(
            term in context_normalized
            for term in ("revenue", "sales", "turnover", "net_sales")
        ):
            return "revenue"
        if "loans_to_subsidiaries" in context_normalized or "loans" in context_normalized and "subsidiar" in context_normalized:
            return "total_loans_due_from_subsidiaries"
    return _canonical_header(raw_label)


def _clean_cross_table_row_label(raw_label: str) -> str:
    return re.sub(
        r"^\s*(?:less|add|plus):\s*",
        "",
        raw_label,
        flags=re.IGNORECASE,
    ).strip()


def _is_metric_row_label(label: str) -> bool:
    terms = set(normalize_identifier(label).split("_"))
    return bool(terms & {"ratio", "margin", "change", "difference", "rate", "percentage", "percent"})


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _canonical_header(header: str) -> str:
    normalized = normalize_identifier(header)
    compact = normalized.replace("_", "")
    for field_name, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            alias_normalized = normalize_identifier(alias)
            if compact == alias_normalized.replace("_", ""):
                return field_name
    return normalized


def _coerce_value(raw: object) -> object:
    if isinstance(raw, (int, float)):
        return raw
    if raw is None:
        return None
    stripped = str(raw).strip()
    numeric = stripped.replace(",", "")
    parsed = _parse_numeric_and_unit(stripped)
    if parsed is not None:
        value, _ = parsed
        return int(value) if value.is_integer() else value
    return stripped


def _parse_numeric_and_unit(raw: object) -> tuple[float, str | None] | None:
    if isinstance(raw, (int, float)):
        return float(raw), None
    text = str(raw).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group(0).replace(",", ""))
    if _numeric_match_is_parenthesized_negative(text, match):
        value = -abs(value)
    unit = (text[:match.start()] + " " + text[match.end():]).strip() or None
    return value, unit


def _numeric_match_is_parenthesized_negative(text: str, match: re.Match[str]) -> bool:
    prefix = text[: match.start()]
    suffix = text[match.end() :]
    return bool(re.search(r"\(\s*[$€¥￥₹£]?\s*$", prefix) and re.match(r"\s*\)", suffix))


def _field_label(field_name: str) -> str:
    return field_name.replace("_", " ")


def _entity_id(
    company_name: str,
    year: int | None,
    row_index: int,
    *,
    document_id: object | None = None,
) -> str:
    suffix = str(year) if year is not None else str(row_index)
    base = f"{normalize_identifier(company_name)}:{suffix}"
    if document_id is not None and str(document_id).strip():
        return f"{normalize_identifier(str(document_id))}:{base}"
    return base


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _matches_entity(token: AttributeValueToken, entity: str) -> bool:
    entity_normalized = normalize_identifier(entity)
    return (
        token.entity_id == entity
        or token.company_name == entity
        or normalize_identifier(token.company_name) == entity_normalized
        or normalize_identifier(token.entity_id).startswith(f"{entity_normalized}_")
        or token.entity_id.startswith(f"{entity}:")
    )


def _graph_from_row_dicts(
    rows: list[dict[str, object]],
    *,
    source_name: str | None,
    table: str | None,
    text: str | None,
) -> AttributeValueGraph:
    tokens: list[AttributeValueToken] = []
    for row_index, raw_row in enumerate(rows):
        row = {
            _canonical_header(str(key)): value
            for key, value in raw_row.items()
            if key is not None
        }
        company = str(
            row.get("company_name")
            or row.get("entity")
            or row.get("name")
            or f"row_{row_index}"
        )
        industry = _optional_str(row.get("industry"))
        year = _optional_int(_coerce_value(row.get("year")))
        row_dimension_fields = _row_dimension_fields(row)
        row_dimensions = _row_dimensions(row, row_dimension_fields=row_dimension_fields)
        entity_id = _entity_id(
            company,
            year,
            row_index,
            document_id=row_dimensions.get("document_id"),
        )
        for field_name, raw_value in row.items():
            if field_name in IDENTITY_FIELDS or field_name in {"entity", "name"}:
                continue
            if field_name in row_dimension_fields:
                continue
            parsed = _parse_numeric_and_unit(raw_value)
            if parsed is None:
                continue
            value, unit = parsed
            src = _source_for_cell(
                source_name=source_name,
                table=table,
                row_index=row_index,
                column=field_name,
                raw_value=raw_value,
                text=text,
            )
            tokens.append(
                AttributeValueToken(
                    token_id=f"{entity_id}:{field_name}",
                    entity_id=entity_id,
                    company_name=company,
                    industry=industry,
                    year=year,
                    field_name=field_name,
                    field_label=_field_label(field_name),
                    value=float(value),
                    source=src,
                    unit=unit,
                    dimensions=row_dimensions,
                )
            )
    return AttributeValueGraph(tuple(tokens), source_name=source_name)


def _row_dimension_fields(row: dict[str, object]) -> set[str]:
    fields: set[str] = set()
    for field_name, raw_value in row.items():
        if field_name in {"company_name", "industry", "year", "entity", "name"}:
            continue
        value = _optional_str(raw_value)
        if value is None:
            continue
        if _is_categorical_row_label_field(field_name, value):
            fields.add(field_name)
    return fields


def _is_categorical_row_label_field(field_name: str, value: str) -> bool:
    normalized = normalize_identifier(field_name)
    hints = (
        "category",
        "segment",
        "region",
        "geography",
        "geographic",
        "product",
        "plan",
        "class",
        "type",
        "description",
    )
    if not any(hint in normalized for hint in hints):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", value))


def _row_dimensions(
    row: dict[str, object],
    *,
    row_dimension_fields: set[str] | None = None,
) -> dict[str, object]:
    row_dimension_fields = row_dimension_fields or set()
    dimensions: dict[str, object] = {}
    for field_name, raw_value in row.items():
        if field_name in {"company_name", "industry", "year", "entity", "name"}:
            continue
        if field_name not in row_dimension_fields and _parse_numeric_and_unit(raw_value) is not None:
            continue
        value = _optional_str(raw_value)
        if value is not None:
            dimensions[field_name] = value
    return dimensions


def _source_for_cell(
    *,
    source_name: str | None,
    table: str | None,
    row_index: int,
    column: str,
    raw_value: object,
    text: str | None,
) -> TokenSource:
    excerpt = str(raw_value)
    char_start = None
    char_end = None
    if text and excerpt:
        pos = text.find(excerpt)
        if pos >= 0:
            char_start = pos
            char_end = pos + len(excerpt)
    return TokenSource(
        document_id=source_name,
        table=table,
        row=row_index,
        column=column,
        char_start=char_start,
        char_end=char_end,
        text_excerpt=excerpt,
    )


def graphs_from_markdown_files(paths: Iterable[str | Path]) -> tuple[AttributeValueGraph, ...]:
    return tuple(graph_from_markdown_file(path) for path in paths)
