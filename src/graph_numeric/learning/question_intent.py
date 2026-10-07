from __future__ import annotations

from dataclasses import dataclass
import re

from graph_numeric.core.attribute_graph import normalize_identifier


@dataclass(frozen=True)
class QuestionIntent:
    answer_type: str
    preferred_operators: tuple[str, ...]
    confidence: float
    reasons: tuple[str, ...]


def classify_question_intent(question: str) -> QuestionIntent:
    """Classify answer/operator intent from question text only.

    This is a transparent gate for external generalization. It never uses
    dataset IDs, gold answers, or document-specific labels.
    """

    lowered = question.lower()
    normalized = normalize_identifier(question)
    reasons: list[str] = []

    if _looks_boolean(lowered):
        reasons.append("boolean_comparison_question")
        return QuestionIntent("boolean", ("BOOLEAN",), 0.86, tuple(reasons))

    if _looks_textual(lowered, normalized):
        reasons.append("textual_answer_question")
        return QuestionIntent("text", ("TEXT_SPAN", "LOOKUP"), 0.84, tuple(reasons))

    if _looks_year_list(lowered, normalized):
        reasons.append("year_list_question")
        return QuestionIntent("list", ("YEAR_LIST", "TEXT_SPAN"), 0.82, tuple(reasons))

    if _looks_multi_value(lowered, normalized):
        reasons.append("multi_value_question")
        return QuestionIntent("multi_value", ("TEXT_SPAN", "LOOKUP"), 0.78, tuple(reasons))

    if _looks_percent_change(lowered, normalized):
        reasons.append("percent_change_question")
        return QuestionIntent("number", ("PERCENT_CHANGE", "DIFFERENCE"), 0.82, tuple(reasons))

    if _looks_share_of_total(lowered, normalized):
        reasons.append("share_of_total_question")
        return QuestionIntent("number", ("SHARE", "RATIO"), 0.80, tuple(reasons))

    if _looks_ratio(lowered, normalized):
        reasons.append("ratio_or_share_question")
        return QuestionIntent("number", ("RATIO", "SHARE"), 0.80, tuple(reasons))

    if _looks_difference(lowered, normalized):
        reasons.append("difference_question")
        return QuestionIntent("number", ("DIFFERENCE", "PERCENT_CHANGE"), 0.76, tuple(reasons))

    if _looks_aggregation(lowered, normalized):
        reasons.append("aggregation_question")
        return QuestionIntent("number", ("SUM", "AVG"), 0.74, tuple(reasons))

    return QuestionIntent("number", ("LOOKUP", "SUM"), 0.55, ("default_numeric_lookup",))


def _looks_boolean(lowered: str) -> bool:
    return bool(
        re.search(r"^(?:was|were|is|are|did|does|do|has|have|had)\b", lowered)
        and re.search(r"\b(?:greater|less|higher|lower|more|fewer|increase|decrease|exceed|above|below|than)\b", lowered)
    )


def _looks_textual(lowered: str, normalized: str) -> bool:
    textual_patterns = (
        r"\bwho\s+is\b",
        r"\bwho\s+was\b",
        r"\bwhere\s+",
        r"\bwhy\b",
        r"\bhow\s+(?:does|did|are|is|should)\b",
        r"\bwhat\s+(?:caused|led|resulted)\b",
        r"\bwhat\s+can\b.+\bbe\s+used\s+for\b",
        r"\bwhat\s+does\b.+\b(?:represent|consist\s+of|include|relate\s+to|mean)\b",
        r"\bwhat\s+do\b.+\b(?:represent|consist\s+of|include|relate\s+to|mean)\b",
        r"\bwhat\s+is\s+included\s+in\b",
        r"\bwhat\s+is\s+.+\b(?:made\s+up\s+of|component\s+of|purpose|policy)\b",
        r"\bwhat\s+are\s+the\s+types\s+of\b",
        r"\bwhat\s+are\s+the\s+components\b",
        r"\bwhat\s+years\s+",
        r"\bwhen\s+",
        r"\bwhich\s+associate",
    )
    if any(re.search(pattern, lowered) for pattern in textual_patterns):
        return True
    return any(
        term in normalized
        for term in (
            "duration",
            "useful_life",
            "expiration_date",
            "no_expiration_date",
            "read_in_conjunction",
        )
    )


def _looks_year_list(lowered: str, normalized: str) -> bool:
    return bool(
        "what_years" in normalized
        or "which_years" in normalized
        or re.search(r"\bwhat\s+(?:fiscal\s+)?years\b", lowered)
        or re.search(r"\bin\s+which\s+(?:fiscal\s+)?years\b", lowered)
    )


def _looks_multi_value(lowered: str, normalized: str) -> bool:
    return bool(
        "respectively" in normalized
        or "chronological_order" in normalized
        or re.search(r"\bfor\s+each\s+(?:financial\s+)?year\b", lowered)
    )


def _looks_percent_change(lowered: str, normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "percent_change",
            "percentage_change",
            "percentage_increase",
            "percent_increase",
            "percentage_decrease",
            "percent_decrease",
            "growth_rate",
        )
    )


def _looks_ratio(lowered: str, normalized: str) -> bool:
    return bool(
        any(term in normalized for term in ("ratio", "portion", "proportion", "percentage_of", "percent_of", "as_percentage_of"))
        or re.search(r"\bwhat\s+percent(?:age)?\s+of\b", lowered)
        or re.search(r"\b(?:as|what)\s+(?:a\s+)?percentage\s+of\b", lowered)
        or re.search(r"\bover\b|\bdivided\s+by\b", lowered)
    )


def _looks_share_of_total(lowered: str, normalized: str) -> bool:
    return bool(
        any(
            term in normalized
            for term in (
                "percentage_of_total",
                "percent_of_total",
                "portion_of_total",
                "proportion_of_total",
                "share_of_total",
            )
        )
        or re.search(r"\bwhat\s+percent(?:age)?\s+of\s+(?:the\s+)?total\b", lowered)
        or re.search(r"\bwhat\s+portion\s+of\s+(?:the\s+)?total\b", lowered)
        or re.search(r"\bas\s+(?:a\s+)?percentage\s+of\s+(?:the\s+)?total\b", lowered)
    )


def _looks_difference(lowered: str, normalized: str) -> bool:
    return bool(
        any(term in normalized for term in ("difference", "change_in", "change_of", "increase_in", "decrease_in"))
        or re.search(r"\bchange\s+(?:in|of)\b", lowered)
        or re.search(r"\bdifference\s+between\b", lowered)
    )


def _looks_aggregation(lowered: str, normalized: str) -> bool:
    return bool(
        any(term in normalized for term in ("total", "sum", "combined", "aggregate", "average"))
        or re.search(r"\btotal\b|\bsum\b|\bcombined\b|\baverage\b", lowered)
    )
