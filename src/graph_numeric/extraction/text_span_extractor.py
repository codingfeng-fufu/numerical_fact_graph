from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from graph_numeric.core.attribute_graph import normalize_identifier


@dataclass(frozen=True)
class TextSpanAnswer:
    operator: str
    strategy: str
    answer: str
    evidence_text: str
    confidence: float
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "strategy": self.strategy,
            "answer": self.answer,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
            "trace": self.trace,
        }


def extract_text_span_answer(question: str, document_text: str) -> TextSpanAnswer | None:
    """Extract high-confidence fact spans from source text.

    Numerical questions remain in the graph executor. This branch only handles
    direct span/fact questions and returns the source sentence as evidence.
    """
    if not looks_like_text_fact_question(question):
        return None
    table_answer = (
        _table_metadata_answer(question, document_text)
        or _oldest_person_table_answer(question, document_text)
        or _chronological_year_row_values_answer(question, document_text)
        or _respective_table_values_answer(question, document_text)
        or _components_comprising_table_answer(question, document_text)
        or _components_from_table_answer(question, document_text)
        or _components_before_total_row_answer(question, document_text)
        or _top_k_table_label_answer(question, document_text)
        or _table_row_type_answer(question, document_text)
        or _table_type_list_answer(question, document_text)
        or _table_named_column_values_answer(question, document_text)
        or _table_column_list_answer(question, document_text)
    )
    if table_answer is not None:
        answer, evidence_text = table_answer
        return TextSpanAnswer(
            operator="TEXT_SPAN",
            strategy="text_fact_span",
            answer=answer,
            evidence_text=evidence_text,
            confidence=0.9,
            trace={
                "question_type": "text_fact",
                "extractor": "components_from_table",
                "numeric_path_bypassed": True,
            },
        )
    paragraphs = _non_table_paragraphs(document_text)
    if not paragraphs:
        return None
    for extractor_name, extractor in (
        ("table_description", _table_description_answer),
        ("note_information", _note_information_answer),
        ("include_only", _include_only_answer),
        ("excluded_if", _excluded_if_answer),
        ("applicable_to", _applicable_to_answer),
        ("named_definition", _named_definition_answer),
        ("definition", _definition_answer),
        ("receive_method", _receive_method_answer),
        ("computed_definition", _computed_definition_answer),
        ("consist_or_cover_definition", _consist_or_cover_definition_answer),
        ("traded_under_symbol", _traded_under_symbol_answer),
        ("date_declared", _date_declared_answer),
        ("event_date", _event_date_answer),
        ("result_of", _result_of_answer),
        ("represents", _represents_answer),
        ("largest_customer", _largest_customer_answer),
        ("considered_to_be", _considered_to_be_answer),
        ("used_for", _used_for_answer),
        ("operate_from", _operate_from_answer),
        ("reportable_segments", _reportable_segments_answer),
        ("because_clause", _because_clause_answer),
        ("why_result", _why_result_answer),
        ("certified_by", _certified_by_answer),
        ("included_in", _included_in_answer),
        ("stated_or_valued", _stated_or_valued_answer),
        ("related_to", _related_to_answer),
        ("based_on", _based_on_answer),
        ("recognized_in_accordance", _recognized_in_accordance_answer),
        ("determined_by", _determined_by_answer),
        ("model_used", _model_used_answer),
        ("caused_by", _caused_by_answer),
    ):
        extracted = extractor(question, paragraphs)
        if extracted is None:
            continue
        answer, evidence_text = extracted
        return TextSpanAnswer(
            operator="TEXT_SPAN",
            strategy="text_fact_span",
            answer=answer,
            evidence_text=evidence_text,
            confidence=0.9,
            trace={
                "question_type": "text_fact",
                "extractor": extractor_name,
                "numeric_path_bypassed": True,
            },
        )
    return None


def looks_like_text_fact_question(question: str) -> bool:
    normalized = normalize_identifier(question)
    lowered = question.lower()
    normalized_terms = set(normalized.split("_"))
    numeric_terms = {
        "average",
        "change",
        "decrease",
        "difference",
        "divided",
        "exceed",
        "how_many",
        "increase",
        "percent",
        "percentage",
        "proportion",
        "ratio",
        "sum",
        "total",
    }
    if re.search(r"\bwhy\b", lowered) or re.search(r"\bwhat\s+(?:caused|led)\b", lowered):
        return True
    if re.search(r"\bwhat\s+(?:are|were)\s+the\s+components\b", lowered):
        return True
    if re.search(r"\bwhat\s+are\s+the\s+components\s+factored\s+in\b", lowered):
        return True
    if re.search(r"\bfor\s+each\s+(?:financial\s+)?year\b", lowered) or re.search(r"\bchronological\s+order\b", lowered):
        return True
    if (
        any(term in normalized_terms for term in numeric_terms)
        or any(term.startswith("percent") for term in normalized_terms)
        or "how_many" in normalized
    ):
        return False
    return bool(
        re.search(r"\bwhy\b", lowered)
        or re.search(r"\bwhen\s+did\b", lowered)
        or re.search(r"\bwhen\s+was\b", lowered)
        or re.search(r"\bwho\s+was\b.+\blargest\b", lowered)
        or re.search(r"\bwho\s+are\s+considered\b", lowered)
        or re.search(r"\bwhere\s+is\b.+\btraded\b", lowered)
        or re.search(r"\bwhere\s+do\b.+\boperate\b", lowered)
        or re.search(r"\bwhat\s+does\s+the\s+table\s+show\b", lowered)
        or re.search(r"\bwhat\s+(?:data|information)\s+does\s+the\s+table\s+contain\b", lowered)
        or re.search(r"\bwhat\s+years\s+does\s+the\s+table\s+provide\b", lowered)
        or re.search(r"\bwhat\s+years\s+are\s+compared\s+in\s+the\s+table\b", lowered)
        or re.search(r"\bwhich\s+.+\bdoes\s+the\s+table\s+provide\s+information\s+for\b", lowered)
        or re.search(r"\bwhich\s+financial\s+year", lowered)
        or re.search(r"\bwhich\s+are\s+the\s+top\s+\d+\b", lowered)
        or re.search(r"\bwhat\s+information\s+does\s+note\s+\d+\s+provide\b", lowered)
        or re.search(r"\bwhat\s+is\s+the\s+content\s+of\s+note\s+\d+\b", lowered)
        or re.search(r"\bwhat\s+are\s+[a-z][a-z0-9 -]+\?", lowered)
        or re.search(r"\bwhat\s+are\s+the\s+respective\b", lowered)
        or re.search(r"\bfor\s+each\s+(?:financial\s+)?year\b", lowered)
        or re.search(r"\bchronological\s+order\b", lowered)
        or re.search(r"\bwhat\s+are\s+the\s+components\b", lowered)
        or re.search(r"\bwhat\s+are\s+the\s+\w+\s+reportable\s+segments\b", lowered)
        or re.search(r"\bwhat\s+were\s+the\s+components\b", lowered)
        or re.search(r"\bwhat\s+are\s+the\s+types\s+of\b", lowered)
        or re.search(r"\bwhat\s+are\s+the\s+dates\b", lowered)
        or re.search(r"\bwhere\s+are\s+.+\bin\s+the\s+table\b", lowered)
        or re.search(r"\bwhen\s+is\b.+\bexcluded\b", lowered)
        or re.search(r"\bwhat\s+is\s+.+\?", lowered)
        or re.search(r"\bwhat\s+is\s+.+\bapplicable\s+to\b", lowered)
        or re.search(r"\bwhat\s+(?:body|model)\b", lowered)
        or re.search(r"\bwhat\s+caused\b", lowered)
        or re.search(r"\bwhat\s+led\s+to\b", lowered)
        or re.search(r"\bwhat\s+was\s+the\s+result\s+of\b", lowered)
        or re.search(r"\bwhat\s+does\b.+\brepresent\b", lowered)
        or re.search(r"\bwhat\s+did\b.+\brepresent\b", lowered)
        or re.search(r"\bwhat\s+does\b.+\bconsist\s+of\b", lowered)
        or re.search(r"\bwhat\s+do\b.+\bonly\s+include\b", lowered)
        or re.search(r"\bwhat\s+does\b.+\brelate\s+to\b", lowered)
        or re.search(r"\bwhat\s+was\s+.+\s+related\s+to\b", lowered)
        or re.search(r"\bwhere\s+(?:are|is)\b.+\bincluded\b", lowered)
        or re.search(r"\bhow\s+does\b", lowered)
        or re.search(r"\bhow\s+did\b.+\bdetermine\b", lowered)
        or re.search(r"\bhow\s+are\b", lowered)
        or re.search(r"\bhow\s+(?:are|is|were|was)\b.+\b(?:stated|valued)\b", lowered)
        or re.search(r"\bhow\s+is\b.+\b(?:computed|calculated)\b", lowered)
        or re.search(r"\bhow\s+is\b.+\bdetermined\b", lowered)
        or re.search(r"\bwho\s+is\s+covered\b", lowered)
        or re.search(r"\bwho\s+is\s+.+\b(?:oldest|youngest)\b", lowered)
        or re.search(r"\bwhat\s+can\b.+\bbe\s+used\s+for\b", lowered)
        or re.search(r"\bwhat\s+are\s+the\s+types\s+of\b.+\bin\s+the\s+table\b", lowered)
        or re.search(r"\bwhich\s+model\b", lowered)
        or re.search(r"\bbased\s+on\b", lowered)
        or re.search(r"\brecogni[sz]ed\b", lowered)
    )


def _non_table_paragraphs(document_text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in document_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return [re.sub(r"\s+", " ", paragraph).strip() for paragraph in paragraphs if paragraph.strip()]


def _components_from_table_answer(question: str, document_text: str) -> tuple[str, str] | None:
    if not re.search(r"\bwhat\s+(?:are|were)\s+the\s+components\b", question, re.IGNORECASE):
        return None
    target_match = re.search(
        r"\bcalculating\s+(?:the\s+)?(?P<target>.+?)(?:\?|$)",
        question,
        re.IGNORECASE,
    )
    target_terms = _question_core_terms(target_match.group("target")) if target_match else set()
    rows: list[tuple[str, str]] = []
    for line in document_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        label = _clean_text_answer(cells[0])
        if not label or re.fullmatch(r"[-\s]*", label):
            continue
        rows.append((label, stripped))
    if len(rows) < 2:
        return None
    target_index = None
    for index, (label, _) in enumerate(rows):
        label_terms = _question_core_terms(label)
        if target_terms and target_terms <= label_terms:
            target_index = index
            break
    if target_index is None:
        return None
    component_labels = [
        label
        for label, _ in rows[:target_index]
        if not (_question_core_terms(label) & {"note", "year"})
        and not _looks_like_unit_header(label)
    ]
    if not component_labels:
        return None
    evidence = "\n".join(row for _, row in rows[: target_index + 1])
    return "; ".join(component_labels), evidence


def _respective_table_values_answer(question: str, document_text: str) -> tuple[str, str] | None:
    if "respectively" not in question.lower():
        return None
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    header = _best_year_header_row(rows)
    year_indexes = _requested_year_column_indexes(question, header)
    if not year_indexes:
        return None
    row_hint = _respective_row_hint(question)
    hint_terms = _question_core_terms(row_hint)
    candidates: list[tuple[int, int, list[str]]] = []
    header_index = rows.index(header) if header in rows else 0
    for index, row in enumerate(rows[header_index + 1 :], start=header_index + 1):
        if not row or _looks_like_unit_header(row[0]):
            continue
        label_terms = _question_core_terms(row[0])
        overlap = hint_terms & label_terms
        compact_hint = _compact_identifier(row_hint)
        compact_label = _compact_identifier(row[0])
        if not overlap and compact_hint not in compact_label:
            continue
        score = len(overlap)
        if compact_hint and compact_hint in compact_label:
            score += 4
        candidates.append((score, index, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    row = candidates[0][2]
    values: list[str] = []
    for _, column_index in year_indexes:
        if column_index >= len(row):
            return None
        value = _clean_span_numeric_cell(row[column_index])
        if not value:
            return None
        values.append(value)
    if not values:
        return None
    return "; ".join(values), "\n".join((" | ".join(header), " | ".join(row)))


def _chronological_year_row_values_answer(question: str, document_text: str) -> tuple[str, str] | None:
    lowered = question.lower()
    if not (
        re.search(r"\bfor\s+each\s+(?:financial\s+)?year\b", lowered)
        or re.search(r"\bchronological\s+order\b", lowered)
    ):
        return None
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    header = _best_year_header_row(rows)
    year_indexes = [
        (int(year), index)
        for index, cell in enumerate(header)
        for year in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", cell)
    ]
    if len(year_indexes) < 2:
        return None
    row_hint = _chronological_row_hint(question)
    hint_terms = _question_core_terms(row_hint)
    candidates: list[tuple[int, int, list[str]]] = []
    header_index = rows.index(header) if header in rows else 0
    for index, row in enumerate(rows[header_index + 1 :], start=header_index + 1):
        if not row or _looks_like_unit_header(row[0]):
            continue
        label_terms = _question_core_terms(row[0])
        overlap = hint_terms & label_terms
        compact_hint = _compact_identifier(row_hint)
        compact_label = _compact_identifier(row[0])
        if not overlap and compact_hint not in compact_label:
            continue
        score = len(overlap)
        if compact_hint and compact_hint in compact_label:
            score += 4
        candidates.append((score, index, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    row = candidates[0][2]
    ordered_indexes = sorted(year_indexes, key=lambda item: item[0])
    values: list[str] = []
    unit = _unit_suffix_from_row_label(row[0])
    for _, column_index in ordered_indexes:
        if column_index >= len(row):
            return None
        value = _clean_span_numeric_cell(row[column_index])
        if not value:
            return None
        values.append(f"{value} {unit}".strip())
    if not values:
        return None
    return "; ".join(values), "\n".join((" | ".join(header), " | ".join(row)))


def _chronological_row_hint(question: str) -> str:
    match = re.search(
        r"\bwhat\s+is\s+(?:the\s+)?(?P<hint>.+?)\s+for\s+each\s+(?:financial\s+)?year\b",
        question,
        re.IGNORECASE,
    )
    if match:
        return match.group("hint")
    return question


def _unit_suffix_from_row_label(label: str) -> str:
    match = re.search(r"\bin\s+(?P<unit>millions?|thousands?|billions?)\b", label, re.IGNORECASE)
    if match:
        unit = match.group("unit").lower()
        return "million" if unit.startswith("million") else "thousand" if unit.startswith("thousand") else "billion"
    return ""


def _components_comprising_table_answer(question: str, document_text: str) -> tuple[str, str] | None:
    if not re.search(r"\bcomponents\s+comprising\b", question, re.IGNORECASE):
        return None
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    labels: list[str] = []
    evidence: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        label = _clean_text_answer(row[0])
        if not label or _looks_like_unit_header(label):
            continue
        if not _row_has_numeric_value(row):
            continue
        label_terms = _question_core_terms(label)
        if "thereof" in label_terms or "total" in label_terms:
            continue
        labels.append(label)
        evidence.append(" | ".join(row))
    if not labels:
        return None
    return "; ".join(labels), "\n".join(evidence)


def _components_before_total_row_answer(question: str, document_text: str) -> tuple[str, str] | None:
    if not re.search(r"\bcomponents\s+(?:making\s+up|of|factored\s+in)\b", question, re.IGNORECASE):
        return None
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    labels: list[str] = []
    evidence: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        label = _clean_text_answer(row[0])
        if not label or _looks_like_unit_header(label):
            continue
        if re.search(r"\btotal\b", label, re.IGNORECASE):
            break
        if not _row_has_numeric_value(row):
            continue
        labels.append(label)
        evidence.append(" | ".join(row))
    if not labels:
        return None
    return "; ".join(labels), "\n".join(evidence)


def _table_metadata_answer(question: str, document_text: str) -> tuple[str, str] | None:
    lowered = question.lower()
    if not (
        re.search(r"\bwhich\s+.+\bdoes\s+the\s+table\s+provide\s+information\s+for\b", lowered)
        or re.search(r"\bwhich\s+financial\s+year", lowered)
        or re.search(r"\bwhat\s+years\s+does\s+the\s+table\s+provide\b", lowered)
        or re.search(r"\bwhat\s+years\s+are\s+compared\s+in\s+the\s+table\b", lowered)
        or re.search(r"\bwhere\s+are\s+.+\bin\s+the\s+table\b", lowered)
    ):
        return None
    table_rows = _markdown_table_rows(document_text)
    if not table_rows:
        return None
    header = _best_year_header_row(table_rows)
    if re.search(r"\bwhere\s+are\s+.+\bin\s+the\s+table\b", lowered):
        labels = _table_group_child_labels(question, table_rows)
        if labels:
            return "; ".join(label for label, _ in labels), "\n".join(row for _, row in labels)
    if "years are compared" in lowered:
        compared_pairs: list[str] = []
        evidence_rows: list[str] = []
        for row in table_rows[:5]:
            row_text = " | ".join(row)
            for cell in row:
                match = re.search(
                    r"\b(19\d{2}|20\d{2})\s*(?:and|to|vs\.?|versus|-)\s*(19\d{2}|20\d{2})\b",
                    cell,
                    re.IGNORECASE,
                )
                if match is None:
                    continue
                pair = f"{match.group(1)} and {match.group(2)}"
                if pair not in compared_pairs:
                    compared_pairs.append(pair)
                if row_text not in evidence_rows:
                    evidence_rows.append(row_text)
        if compared_pairs:
            return "; ".join(compared_pairs), "\n".join(evidence_rows)
    if "which years" in lowered or "what years" in lowered or "financial year" in lowered or "years are compared" in lowered:
        years: list[str] = []
        for row in table_rows[:5]:
            for cell in row:
                for year in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", cell):
                    if year not in years:
                        years.append(year)
        if years:
            return "; ".join(years), " | ".join(header)
    if "segment" in lowered:
        labels: list[str] = []
        for row in table_rows[:5]:
            row_labels = [
                _clean_text_answer(cell)
                for cell in row
                if re.fullmatch(r"[A-Z]{2,}", cell.strip())
                and cell.strip().upper() != "TOTAL"
            ]
            if len(row_labels) >= 2:
                labels = row_labels
                break
        if not labels:
            labels = [
                _clean_text_answer(cell)
                for cell in header
                if cell.strip()
                and not re.search(r"\btotal\b", cell, re.IGNORECASE)
                and not re.search(r"\breportable\s+segment\b", cell, re.IGNORECASE)
                and not re.search(r"\bfiscal\s+year\b|\byear\s+ended\b", cell, re.IGNORECASE)
            ]
        if labels:
            return "; ".join(labels), " | ".join(header)
    return None


def _table_group_child_labels(question: str, table_rows: list[list[str]]) -> list[tuple[str, str]]:
    question_terms = _question_core_terms(question)
    group_index = None
    for index, row in enumerate(table_rows):
        label = _clean_text_answer(row[0]) if row else ""
        if not label:
            continue
        label_terms = _question_core_terms(label)
        if label_terms and len(question_terms & label_terms) >= min(2, len(label_terms)):
            group_index = index
            break
    start = group_index + 1 if group_index is not None else 1
    labels: list[tuple[str, str]] = []
    for row in table_rows[start:]:
        if not row:
            continue
        label = _clean_text_answer(row[0])
        if not label or _looks_like_unit_header(label):
            continue
        if re.search(r"\btotal\b", label, re.IGNORECASE) or not _row_has_numeric_value(row):
            if labels:
                break
            continue
        labels.append((label, " | ".join(row)))
    return labels


def _best_year_header_row(table_rows: list[list[str]]) -> list[str]:
    for row in table_rows[:5]:
        if sum(1 for cell in row if re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", cell)) >= 2:
            return row
    return table_rows[0]


def _top_k_table_label_answer(question: str, document_text: str) -> tuple[str, str] | None:
    match = re.search(
        r"\bwhich\s+are\s+the\s+top\s+(?P<k>\d+)\s+(?P<target>.+?)\s+for\s+(?P<year>19\d{2}|20\d{2})\b",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    k = int(match.group("k"))
    year = match.group("year")
    target_terms = _question_core_terms(match.group("target"))
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    header = rows[0]
    year_index = next((index for index, cell in enumerate(header) if year in cell), None)
    if year_index is None:
        return None
    candidates: list[tuple[float, str, str]] = []
    for row in rows[1:]:
        if year_index >= len(row):
            continue
        label = _clean_text_answer(row[0])
        if not label or _looks_like_unit_header(label) or re.search(r"\btotal\b", label, re.IGNORECASE):
            continue
        label_terms = _question_core_terms(label)
        if target_terms and not (target_terms & label_terms):
            continue
        value = _number_from_text(row[year_index])
        if value is None:
            continue
        candidates.append((value, label, " | ".join(row)))
    if len(candidates) < k:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:k]
    evidence = "\n".join([" | ".join(header), *(row for _, _, row in selected)])
    return "; ".join(label for _, label, _ in selected), evidence


def _table_column_list_answer(question: str, document_text: str) -> tuple[str, str] | None:
    if not re.search(r"\bwhat\s+are\s+the\s+respective\b", question, re.IGNORECASE):
        return None
    column_match = re.search(
        r"\bwhat\s+are\s+the\s+respective\s+(?P<column>[a-z][a-z0-9 -]+?)(?:\s+of|\s+for|\?)",
        question,
        re.IGNORECASE,
    )
    if column_match is None:
        return None
    column_terms = _question_core_terms(column_match.group("column"))
    if not column_terms:
        return None
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    header = rows[0]
    column_index = None
    for index, cell in enumerate(header):
        cell_terms = _question_core_terms(cell)
        if cell_terms and (column_terms <= cell_terms or cell_terms <= column_terms):
            column_index = index
            break
    if column_index is None:
        return None
    values: list[str] = []
    evidence_rows: list[str] = [" | ".join(header)]
    for row in rows[1:]:
        if column_index >= len(row):
            continue
        value = _clean_text_answer(row[column_index])
        if not value or _looks_like_unit_header(value):
            continue
        values.append(value)
        evidence_rows.append(" | ".join(row))
    if not values:
        return None
    return "; ".join(values), "\n".join(evidence_rows)


def _table_named_column_values_answer(question: str, document_text: str) -> tuple[str, str] | None:
    lowered = question.lower()
    column_hint = None
    if re.search(r"\bwhat\s+are\s+the\s+dates\b", lowered):
        column_hint = "date"
    if column_hint is None:
        return None
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    header = next((row for row in rows[:5] if any(column_hint in cell.lower() for cell in row)), rows[0])
    header_index = rows.index(header)
    column_index = next((index for index, cell in enumerate(header) if column_hint in cell.lower()), None)
    if column_index is None:
        return None
    row_terms = _question_core_terms(question) - {"date", "dates"}
    values: list[str] = []
    evidence_rows: list[str] = [" | ".join(header)]
    for row in rows[header_index + 1 :]:
        if column_index >= len(row):
            continue
        label_terms = _question_core_terms(" ".join(row[:1]))
        if row_terms and not (row_terms & label_terms):
            continue
        value = _clean_text_answer(row[column_index])
        if not value or _looks_like_unit_header(value):
            continue
        if value not in values:
            values.append(value)
            evidence_rows.append(" | ".join(row))
    if not values:
        return None
    return "; ".join(values), "\n".join(evidence_rows)


def _oldest_person_table_answer(question: str, document_text: str) -> tuple[str, str] | None:
    if not re.search(r"\bwho\s+is\s+.+\b(?:oldest|youngest)\b", question, re.IGNORECASE):
        return None
    wants_oldest = bool(re.search(r"\boldest\b", question, re.IGNORECASE))
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    header = rows[0]
    name_index = next(
        (index for index, cell in enumerate(header) if re.search(r"\bname\b", cell, re.IGNORECASE)),
        0,
    )
    age_index = next(
        (index for index, cell in enumerate(header) if re.search(r"\bage\b", cell, re.IGNORECASE)),
        None,
    )
    if age_index is None:
        return None
    candidates: list[tuple[float, str, str]] = []
    for row in rows[1:]:
        if max(name_index, age_index) >= len(row):
            continue
        name = _clean_text_answer(row[name_index])
        age = _number_from_text(row[age_index])
        if not name or age is None:
            continue
        candidates.append((age, name, " | ".join(row)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0] if wants_oldest else item[0], item[1]))
    _, name, evidence = candidates[0]
    return name, "\n".join([" | ".join(header), evidence])


def _table_type_list_answer(question: str, document_text: str) -> tuple[str, str] | None:
    match = re.search(
        r"\bwhat\s+are\s+the\s+types\s+of\s+(?P<target>.+?)\s+in\s+the\s+table\b",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    target_terms = _question_core_terms(match.group("target"))
    rows = _markdown_table_rows(document_text)
    if len(rows) < 2:
        return None
    labels: list[str] = []
    evidence: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        label = _clean_text_answer(row[0])
        if not label or _looks_like_unit_header(label):
            continue
        label_terms = _question_core_terms(label)
        if "total" in label_terms:
            break
        if not _row_has_numeric_value(row):
            continue
        labels.append(label)
        evidence.append(" | ".join(row))
    if not labels:
        return None
    return "; ".join(labels), "\n".join(evidence)


def _table_row_type_answer(question: str, document_text: str) -> tuple[str, str] | None:
    match = re.search(
        r"\bwhat\s+are\s+the\s+types\s+of\s+(?P<target>.+?)\s+provided\s+in\s+the\s+table\b",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    target_terms = _question_core_terms(match.group("target"))
    if not target_terms:
        return None
    rows = _markdown_table_rows(document_text)
    labels: list[str] = []
    evidence: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        label = _clean_text_answer(row[0])
        if not label:
            continue
        label_terms = _question_core_terms(label)
        if target_terms & label_terms and not _looks_like_unit_header(label):
            if "per" in target_terms and "share" in target_terms and not {"per", "share"} <= label_terms:
                continue
            if not _row_has_numeric_value(row):
                continue
            labels.append(label)
            evidence.append(" | ".join(row))
    if not labels:
        return None
    return "; ".join(labels), "\n".join(evidence)


def _markdown_table_rows(document_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in document_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", stripped):
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def _table_description_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not (
        re.search(r"\bwhat\s+does\s+the\s+table\s+show\b", question, re.IGNORECASE)
        or re.search(r"\bwhat\s+(?:data|information)\s+does\s+the\s+table\s+contain\b", question, re.IGNORECASE)
    ):
        return None
    first = paragraphs[0]
    title_match = re.match(r"(?:item\s+\d+[a-z]?\.\s*)?(.+?\b(?:data|information|schedule|table))\b", first, re.IGNORECASE)
    if title_match:
        return _clean_text_answer(title_match.group(1)), first
    match = re.match(r"(.+?)\s+(?:are|is|were|was)\s+as\s+follows\b", first, re.IGNORECASE)
    if match:
        return _clean_text_answer(match.group(1)), first
    match = re.match(r"(.+?)\s+(?:is|are)\s+(?:shown|provided|presented)\b", first, re.IGNORECASE)
    if match:
        return _clean_text_answer(match.group(1)), first
    return _clean_text_answer(first.rstrip(":")), first


def _because_clause_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhy\b|\bbecause\b", question, re.IGNORECASE):
        return None
    question_tail = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _sentences(paragraphs):
        match = re.search(r"\b(Because\s+of\s+[^,.;]+)", sentence, re.IGNORECASE)
        if not match:
            continue
        score = len(question_tail & _question_core_terms(sentence))
        candidates.append((score, _clean_text_answer(match.group(1)), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _why_result_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhy\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for paragraph in paragraphs:
        if not (question_terms & _question_core_terms(paragraph)):
            continue
        parts: list[str] = []
        for sentence in _sentences([paragraph]):
            relate_match = re.search(r"\brelates?\s+to\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
            if relate_match:
                parts.append(relate_match.group(1))
                continue
            result_match = re.search(
                r"\b(.+?\b(?:was|were|is|are)\s+a\s+result\s+of\s+.+?)(?:\.|;|$)",
                sentence,
                re.IGNORECASE,
            )
            if result_match:
                parts.append(result_match.group(1))
                continue
            cause_match = re.search(
                r"\b(?:due\s+to|driven\s+by|resulted\s+from)\s+(.+?)(?:\.|;|$)",
                sentence,
                re.IGNORECASE,
            )
            if cause_match:
                parts.append(cause_match.group(1))
        if not parts:
            continue
        answer = ". ".join(parts)
        candidates.append((len(question_terms & _question_core_terms(answer)), _clean_text_answer(answer), paragraph))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _used_for_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhat\s+can\b.+\bbe\s+used\s+for\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _ranked_sentences(question, paragraphs):
        if not re.search(r"\b(?:may|can|could|will)\s+be\s+used\s+to\b", sentence, re.IGNORECASE):
            continue
        match = re.search(
            r"\b(?:may|can|could|will)\s+be\s+used\s+to\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match is None:
            continue
        answer = _clean_text_answer(match.group(1))
        candidates.append((len(question_terms & _question_core_terms(sentence)), answer, sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _operate_from_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhere\s+do\b.+\boperate\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(r"\b(?:operate|function)\s+from\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _reportable_segments_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if "reportable_segments" not in normalize_identifier(question):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\breportable\s+segments\s*:\s*(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
        match = re.search(
            r"\bdivided\s+into\s+(?:two|three|four|\d+)\s+reportable\s+segments\s*,?\s*(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _certified_by_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bcertif(?:y|ies|ied|ying)\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\bcertified\s+by\s+([^.;,]+?(?:\([^)]*\))?)\s+(?:function|for|from|as|,|\.|;)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _related_to_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    normalized = normalize_identifier(question)
    if "related_to" not in normalized and "relate_to" not in normalized:
        return None
    for paragraph in paragraphs:
        match = re.search(r"\b(related\s+to\s+.+)$", paragraph, re.IGNORECASE)
        if match:
            return _clean_text_answer(match.group(1)), paragraph
        match = re.search(r"\b(relates?\s+(?:solely\s+)?to\s+.+?)(?:\.|;|$)", paragraph, re.IGNORECASE)
        if match:
            answer = re.sub(r"^relates?\b", "relate", match.group(1), flags=re.IGNORECASE)
            return _clean_text_answer(answer), paragraph
    return None


def _included_in_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhere\s+(?:are|is)\b.+\bincluded\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\bincluded\s+(?P<prep>in|within)\s+(?P<answer>.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            if match.group("prep").lower() == "in":
                return _clean_text_answer(f"in {match.group('answer')}"), sentence
            return _clean_text_answer(match.group("answer")), sentence
    return None


def _stated_or_valued_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bhow\s+(?:are|is|were|was)\b.+\b(?:stated|valued)\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\b(?:are|is|were|was)\s+(?P<verb>stated|valued)\s+(?P<tail>.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(f"{match.group('verb')} {match.group('tail')}"), sentence
    return None


def _note_information_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    content_match = re.search(r"\bwhat\s+is\s+the\s+content\s+of\s+note\s+(?P<note>\d+)\b", question, re.IGNORECASE)
    if content_match is not None:
        note = content_match.group("note")
        for paragraph in paragraphs:
            match = re.match(rf"\s*{re.escape(note)}\.\s+(.+)$", paragraph)
            if match:
                title_text = match.group(1)
                uppercase_prefix = re.match(r"([A-Z][A-Z0-9 &,()/.-]+?)(?=\s+[A-Z][a-z])", title_text)
                title = uppercase_prefix.group(1) if uppercase_prefix else re.split(r"\s{2,}|\. ", title_text, maxsplit=1)[0]
                return _clean_text_answer(title), paragraph
        return None
    if not re.search(r"\bwhat\s+information\s+does\s+note\s+\d+\s+provide\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\brefer\s+to\s+note\s+\d+\s+for\s+(information\s+(?:on|about)\s+[^.;]+)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
        match = re.search(
            r"\bnote\s+\d+\s+provides?\s+information\s+about\s+([^.;]+)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(f"Information about {match.group(1)}"), sentence
        match = re.search(r"\bnote\s+\d+\s+provides?\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _definition_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    match = re.search(r"\bwhat\s+are\s+(?P<subject>[a-z][a-z0-9 -]+)\?", question, re.IGNORECASE)
    if match is None:
        return None
    subject = match.group("subject").strip()
    subject_terms = _question_core_terms(subject)
    if not subject_terms:
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        if not subject_terms <= _question_core_terms(sentence):
            continue
        if re.search(rf"\b{re.escape(subject)}\s+(?:are|is)\b", sentence, re.IGNORECASE):
            return _clean_text_answer(sentence), sentence
    return None


def _include_only_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhat\s+do\b.+\bonly\s+include\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _ranked_sentences(question, paragraphs):
        sentence_terms = _question_core_terms(sentence)
        if not (question_terms & sentence_terms):
            continue
        match = re.search(r"\bonly\s+include\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match:
            candidates.append((len(question_terms & sentence_terms), _clean_text_answer(match.group(1)), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _excluded_if_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhen\s+is\b.+\bexcluded\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _ranked_sentences(question, paragraphs):
        if not re.search(r"\bexcluded\b", sentence, re.IGNORECASE):
            continue
        sentence_terms = _question_core_terms(sentence)
        if not (question_terms & sentence_terms):
            continue
        match = re.search(r"\bexcluded\b.+?\b(if\s+.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match:
            candidates.append((len(question_terms & sentence_terms), _clean_text_answer(match.group(1)), sentence))
            continue
        match = re.search(r"\bexcluded\b.+?\bbecause\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match:
            candidates.append((len(question_terms & sentence_terms), _clean_text_answer(f"Because {match.group(1)}"), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _named_definition_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    match = re.search(r"\bwhat\s+is\s+(?:the\s+)?(?P<subject>.+?)\?", question, re.IGNORECASE)
    if match is None:
        return None
    normalized_question = normalize_identifier(question)
    numeric_definition_blockers = {
        "amount",
        "mean",
        "annual",
        "average",
        "rate",
        "value",
        "balance",
        "profit",
        "expense",
        "revenue",
        "interest",
    }
    question_terms = {term for term in normalized_question.split("_") if term}
    if question_terms & numeric_definition_blockers:
        return None
    subject = match.group("subject").strip(" '\"")
    if not subject:
        return None
    subject_terms = _question_core_terms(subject)
    if not subject_terms:
        return None
    sentences = _ranked_sentences(question, paragraphs)
    candidates: list[tuple[int, int, str, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _question_core_terms(sentence)
        if len(subject_terms & sentence_terms) < max(1, min(2, len(subject_terms))):
            continue
        if re.search(r"\b(?:established|introduced|launched|created|adopted)\b", sentence, re.IGNORECASE):
            following = _definition_following_sentences(sentences[index : index + 6])
            if following:
                answer = " ".join(following)
                candidates.append((len(subject_terms & _question_core_terms(answer)), index, _clean_text_answer(answer), answer))
                continue
        match_def = re.search(r"\b(?:is|are|was|were|means|represents?)\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match_def:
            answer = match_def.group(1)
            candidates.append((len(subject_terms & sentence_terms), index, _clean_text_answer(answer), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    _, _, answer, evidence = candidates[0]
    return answer, evidence


def _definition_following_sentences(sentences: list[str]) -> list[str]:
    selected: list[str] = []
    for sentence in sentences[1:]:
        if re.search(r"\b(?:awarded|entitled|conditions?|fair\s+value|measured|valued|subject\s+to)\b", sentence, re.IGNORECASE):
            selected.append(sentence)
            continue
        if selected:
            break
    return selected


def _receive_method_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bhow\s+does\b.+\breceive\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _ranked_sentences(question, paragraphs):
        if not re.search(r"\breceives?\b", sentence, re.IGNORECASE):
            continue
        sentence_terms = _question_core_terms(sentence)
        if not (question_terms & sentence_terms):
            continue
        match = re.search(
            r"\breceives?\b.+?\b(on|upon|at|by|through|via)\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            answer = f"{match.group(1)} {match.group(2)}"
            candidates.append((len(question_terms & sentence_terms), _clean_text_answer(answer), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _computed_definition_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bhow\s+is\b.+\b(?:computed|calculated)\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _ranked_sentences(question, paragraphs):
        if not re.search(r"\b(?:computed|calculated|based\s+upon|based\s+on)\b", sentence, re.IGNORECASE):
            continue
        if not (question_terms & _question_core_terms(sentence)):
            continue
        match = re.search(
            r"\bbased\s+upon\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            answer = f"Based upon {match.group(1)}"
            candidates.append((len(question_terms & _question_core_terms(sentence)), _clean_text_answer(answer), sentence))
            continue
        match = re.search(
            r"\bbased\s+on\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            answer = f"Based on {match.group(1)}"
            candidates.append((len(question_terms & _question_core_terms(sentence)), _clean_text_answer(answer), sentence))
            continue
        match = re.search(
            r"\b(?:is|are|was|were)\s+(?:computed|calculated)\s+by\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            answer = f"by {match.group(1)}"
            candidates.append((len(question_terms & _question_core_terms(sentence)), _clean_text_answer(answer), sentence))
            continue
        match = re.search(
            r"\b(?:computed|calculated)\s+by\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            answer = f"by {match.group(1)}"
            candidates.append((len(question_terms & _question_core_terms(sentence)), _clean_text_answer(answer), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _consist_or_cover_definition_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    lowered = question.lower()
    if not (
        re.search(r"\bwhat\s+does\b.+\bconsist\s+of\b", lowered)
        or re.search(r"\bwho\s+is\s+covered\b", lowered)
    ):
        return None
    question_terms = _question_core_terms(question)
    candidates: list[tuple[int, str, str]] = []
    for sentence in _ranked_sentences(question, paragraphs):
        sentence_terms = _question_core_terms(sentence)
        if not (question_terms & sentence_terms):
            continue
        consist_match = re.search(r"\bconsists?\s+of\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if consist_match:
            answer = consist_match.group(1)
            candidates.append((len(question_terms & sentence_terms), _clean_text_answer(answer), sentence))
            continue
        covered_match = re.search(r"\bcovering\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if covered_match:
            candidates.append((len(question_terms & sentence_terms), _clean_text_answer(covered_match.group(1)), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, answer, evidence = candidates[0]
    return answer, evidence


def _traded_under_symbol_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhere\s+is\b.+\btraded\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\btraded\s+on\s+the\s+(.+?)\s+under\s+the\s+symbol\b",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _date_declared_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhen\s+was\b.+\bdeclared\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        if not re.search(r"\bdeclared\b", sentence, re.IGNORECASE):
            continue
        match = re.search(r"\b(?:on\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+(?:19|20)\d{2})\b", sentence)
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _event_date_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhen\s+did\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    for sentence in _ranked_sentences(question, paragraphs):
        if not (question_terms & _question_core_terms(sentence)):
            continue
        match = re.search(
            r"\b(?:effective\s+|on\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+(?:19|20)\d{2})\b",
            sentence,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _applicable_to_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bapplicable\s+to\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\b(?:is\s+)?((?:only\s+)?applicable\s+to\s+.+?)(?:\s+The\b|\s+We\b|[.;]|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _result_of_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhat\s+was\s+the\s+result\s+of\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\b(?:has\s+)?resulted\s+in\s+(.+?)(?:(?<!\d)\.(?!\d)|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _represents_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwhat\s+(?:does|did)\b.+\brepresent\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(r"\b((?:represents?|represented)\s+.+)$", sentence, re.IGNORECASE)
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _largest_customer_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwho\s+was\b.+\blargest\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\b([A-Z][A-Za-z0-9&.' -]+?)\s+was\s+our\s+largest\s+customer\b",
            sentence,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _considered_to_be_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bwho\s+are\s+considered\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\b(.+?)\s*,?\s+and\s+are\s+considered\s+to\s+be\s+.+?(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            answer = match.group(1)
            subject_start = re.search(r"\b(?:All|The)\s+[a-z][^.;]*$", answer)
            if subject_start:
                answer = subject_start.group(0)
            return _clean_text_answer(answer), sentence
    return None


def _determined_by_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bhow\s+(?:is|did)\b.+\bdetermine[sd]?\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\bdetermine\s+.+?\s+by\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
        match = re.search(
            r"\bdetermined\s+using\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(f"Using {match.group(1)}"), sentence
        match = re.search(
            r"\bdetermined\s+by\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(f"By {match.group(1)}"), sentence
    return None


def _based_on_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\bbased\s+on\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(
            r"\b(?:is\s+)?based\s+on\s+(.+?)(?:\.|;|$)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _recognized_in_accordance_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\brecogni[sz]ed\b", question, re.IGNORECASE):
        return None
    question_terms = _question_core_terms(question)
    clauses: list[tuple[int, int, int, str]] = []
    for sentence_index, sentence in enumerate(_ranked_sentences(question, paragraphs)):
        for clause_index, clause in enumerate(_split_clauses(sentence)):
            if not re.search(r"\brecogni[sz]ed\b", clause, re.IGNORECASE):
                continue
            score = len(question_terms & _question_core_terms(clause))
            clauses.append((score, sentence_index, clause_index, clause))
    for _, _, _, clause in sorted(clauses, key=lambda item: (-item[0], item[1], item[2])):
        match = re.search(
            r"\brecogni[sz]ed\b\s+in\s+accordance\s+with\s+(.+?)(?:\.|;|$)",
            clause,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(f"In accordance with {match.group(1)}"), clause
        match = re.search(
            r"\brecogni[sz]ed\b\s+(.+?)(?:\.|;|$)",
            clause,
            re.IGNORECASE,
        )
        if match:
            return _clean_text_answer(f"Recognized {match.group(1)}"), clause
    return None


def _model_used_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if "model" not in normalize_identifier(question):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(r"\b([A-Z][A-Za-z-]+(?:-[A-Z][A-Za-z-]+)*)\s+model\b", sentence)
        if match:
            return _clean_text_answer(match.group(1)), sentence
    return None


def _caused_by_answer(question: str, paragraphs: list[str]) -> tuple[str, str] | None:
    if not re.search(r"\b(?:caused|led)\b", question, re.IGNORECASE):
        return None
    for sentence in _ranked_sentences(question, paragraphs):
        match = re.search(r"\b(?:due\s+to|primarily\s+due\s+to|resulted\s+from|driven\s+by)\s+(.+?)(?:\.|;|$)", sentence, re.IGNORECASE)
        if match:
            answer = match.group(1)
            if re.search(r"\bdriven\s+by\b", match.group(0), re.IGNORECASE):
                answer = f"Driven by {answer}"
            return _clean_text_answer(answer), sentence
    return None


def _ranked_sentences(question: str, paragraphs: list[str]) -> list[str]:
    q_terms = _question_core_terms(question)
    scored = [
        (len(q_terms & _question_core_terms(sentence)), index, sentence)
        for index, sentence in enumerate(_sentences(paragraphs))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sentence for _, _, sentence in scored]


def _sentences(paragraphs: list[str]) -> list[str]:
    sentences: list[str] = []
    for paragraph in paragraphs:
        protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", paragraph)
        parts = re.split(r"(?<=[.!?])\s+", protected)
        parts = [part.replace("<DOT>", ".") for part in parts]
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def _split_clauses(sentence: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*;\s*", sentence) if part.strip()]


def _question_core_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "body",
        "by",
        "does",
        "for",
        "from",
        "how",
        "is",
        "it",
        "of",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "why",
        "with",
    }
    return {
        term
        for term in normalize_identifier(text).split("_")
        if term and not term.isdigit() and term not in stopwords
    }


def _requested_year_column_indexes(question: str, header: list[str]) -> list[tuple[int, int]]:
    requested_years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    indexes: list[tuple[int, int]] = []
    for year in requested_years:
        for index, cell in enumerate(header):
            if re.search(rf"\b{year}\b", cell):
                indexes.append((year, index))
                break
    return indexes


def _respective_row_hint(question: str) -> str:
    match = re.search(
        r"\brespective\s+(?P<hint>.+?)\s+in\s+(?:19|20)\d{2}\b",
        question,
        re.IGNORECASE,
    )
    if match:
        return match.group("hint")
    match = re.search(r"\bwhat\s+are\s+the\s+(?P<hint>.+?)\s+respectively\b", question, re.IGNORECASE)
    if match:
        return match.group("hint")
    return question


def _compact_identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_identifier(text))


def _clean_span_numeric_cell(cell: str) -> str:
    text = re.sub(r"[$€£¥￥₹,%]", "", cell)
    text = re.sub(r"\(([^)]+)\)", r"-\1", text)
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    if match is None:
        return ""
    value = match.group(0).replace(",", "")
    if value.endswith(".0"):
        return value[:-2]
    return value


def _number_from_text(text: str) -> float | None:
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _row_has_numeric_value(row: list[str]) -> bool:
    return any(_number_from_text(cell) is not None for cell in row[1:])


def _looks_like_unit_header(text: str) -> bool:
    normalized = normalize_identifier(text)
    terms = {term for term in normalized.split("_") if term}
    return bool(terms) and terms <= {
        "usd",
        "us",
        "dollar",
        "dollars",
        "million",
        "millions",
        "thousand",
        "thousands",
        "m",
        "000",
    } or "$" in text


def _clean_text_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip(" :;.")
    answer = answer.replace(",", "")
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    if answer.lower().startswith("because "):
        return "Because " + answer[8:].strip()
    if answer.lower().startswith("related to "):
        return "Related to " + answer[11:].strip()
    return answer
