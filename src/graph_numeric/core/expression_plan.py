from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
SHORT_FISCAL_YEAR_RE = re.compile(
    r"\b(?:FY|fiscal\s+year|fiscal)\s*['’]?(?P<year>\d{2})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceQuery:
    """A symbolic request for one numerical value before graph grounding."""

    field_surface: str
    time_surface: str | None = None
    entity_surface: str | None = None
    unit_surface: str | None = None
    role: str = "value"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_surface": self.field_surface,
            "time_surface": self.time_surface,
            "entity_surface": self.entity_surface,
            "unit_surface": self.unit_surface,
            "role": self.role,
        }


@dataclass(frozen=True)
class ExpressionNode:
    """A small expression-tree node for numerical question intent."""

    op: str
    children: tuple["ExpressionNode", ...] = field(default_factory=tuple)
    evidence_query: EvidenceQuery | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "children": [child.to_dict() for child in self.children],
            "evidence_query": (
                self.evidence_query.to_dict()
                if self.evidence_query is not None
                else None
            ),
            "value": self.value,
        }


@dataclass(frozen=True)
class ExpressionPlan:
    """Dataset-agnostic expression skeleton parsed from a question."""

    kind: str
    route_operator: str
    root: ExpressionNode
    confidence: float = 0.85
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "route_operator": self.route_operator,
            "root": self.root.to_dict(),
            "confidence": self.confidence,
            "trace": dict(self.trace),
        }


def parse_expression_plan(query: str) -> ExpressionPlan | None:
    """Parse a high-confidence numerical expression skeleton from a query."""
    normalized = _normalize_space(query)
    lowered = normalized.lower()

    if _is_percent_change(lowered):
        return _parse_percent_change(normalized)
    if _is_growth_rate(lowered):
        plan = _parse_percent_change(normalized)
        if plan is not None:
            return plan
    if _is_percent_increase_decrease(lowered):
        plan = _parse_percent_change(normalized)
        if plan is not None:
            return plan
    if _is_increase_decrease_change(lowered):
        plan = _parse_difference(normalized)
        if plan is not None:
            return plan
    if _is_part_to_whole(lowered):
        return _parse_part_to_whole(normalized)
    if _is_ratio(lowered):
        return _parse_ratio(normalized)
    if _is_average(lowered):
        return _parse_average(normalized)
    if _is_difference(lowered):
        return _parse_difference(normalized)
    if _is_sum(lowered):
        return _parse_sum(normalized)
    if _is_lookup(lowered):
        return _parse_lookup(normalized)
    return None


def expression_route_operator(query: str) -> str | None:
    plan = parse_expression_plan(query)
    return plan.route_operator if plan is not None else None


def _parse_percent_change(query: str) -> ExpressionPlan | None:
    years = YEAR_RE.findall(query)
    if len(years) < 2:
        return None
    from_year, to_year = years[0], years[-1]
    year_prefix = _year_range_prefix()
    field = _field_between(
        query,
        start_patterns=(
            r"percentage\s+change\s+in\s+",
            r"percent\s+change\s+in\s+",
            r"percentage\s+change\s+of\s+",
            r"percent\s+change\s+of\s+",
            r"percentage\s+of\s+the\s+change\s+in\s+",
            r"percent\s+of\s+the\s+change\s+in\s+",
            r"percentage\s+net\s+change\s+in\s+",
            r"percent\s+net\s+change\s+in\s+",
            r"percentage\s+net\s+change\s+of\s+",
            r"percent\s+net\s+change\s+of\s+",
        ),
        end_pattern=rf"\s+(?:from|between)\s+{year_prefix}{re.escape(from_year)}\b",
    )
    if not field:
        # Trailing form: "the change in X from Y to Z ... the percent of the change"
        field = _field_between(
            query,
            start_patterns=(
                r"change\s+in\s+",
                r"change\s+of\s+",
            ),
            end_pattern=rf"\s+(?:from|between)\s+{year_prefix}{re.escape(from_year)}\b",
        )
    if not field:
        field = _field_between(
            query,
            start_patterns=(
                r"growth\s+rate\s+in\s+",
                r"growth\s+rate\s+of\s+",
                r"growth\s+observed\s+in\s+",
            ),
            end_pattern=rf"\s+(?:from|during)\s+{year_prefix}{re.escape(from_year)}\b",
        )
    if not field:
        field = _field_between(
            query,
            start_patterns=(
                r"percentage\s+increase\s+in\s+",
                r"percent\s+increase\s+in\s+",
                r"percentage\s+decrease\s+in\s+",
                r"percent\s+decrease\s+in\s+",
                r"percentage\s+dropped\s+in\s+",
                r"percent\s+dropped\s+in\s+",
            ),
            end_pattern=rf"\s+from\s+{year_prefix}{re.escape(from_year)}\b",
        )
    if not field:
        field = _field_before_year_range(query, years)
    if not field:
        return None
    field = _append_trailing_year_range_qualifier(
        _clean_field_surface(field),
        query,
        years,
    )
    from_lookup = _lookup(field, from_year, role="from")
    to_lookup = _lookup(field, to_year, role="to")
    subtract = ExpressionNode("subtract", children=(to_lookup, from_lookup))
    root = ExpressionNode("divide", children=(subtract, from_lookup))
    return ExpressionPlan(
        kind="percent_change",
        route_operator="PERCENT_CHANGE",
        root=root,
        confidence=0.9,
        trace={"parser": "expression_plan.regex_v1"},
    )


def _parse_part_to_whole(query: str) -> ExpressionPlan | None:
    year = _last_year(query)
    lowered = query.lower()
    split = _parse_x_as_percentage_of_total_y(lowered)
    if split is not None:
        numerator_text, denominator_text = split
        numerator_field, numerator_year = _extract_trailing_year(numerator_text)
        denominator_field, denominator_year = _extract_trailing_year(denominator_text)
        numerator = _lookup(
            _strip_leading_question_context(_clean_field_surface(numerator_field)),
            numerator_year or year,
            role="part",
        )
        denominator = _lookup(
            _clean_field_surface(denominator_field),
            denominator_year or year,
            role="whole",
        )
        return ExpressionPlan(
            kind="part_to_whole",
            route_operator="SHARE",
            root=ExpressionNode("divide", children=(numerator, denominator)),
            confidence=0.84,
            trace={"parser": "expression_plan.regex_v1"},
        )

    split = _parse_percent_of_part_as_total(lowered)
    if split is not None:
        numerator_text, denominator_text = split
        numerator = _lookup(_clean_field_surface(numerator_text), year, role="part")
        denominator = _lookup(_clean_field_surface(denominator_text), year, role="whole")
        return ExpressionPlan(
            kind="part_to_whole",
            route_operator="SHARE",
            root=ExpressionNode("divide", children=(numerator, denominator)),
            confidence=0.76,
            trace={"parser": "expression_plan.regex_v1"},
        )

    split = _parse_percent_of_part_to_whole(lowered)
    if split is not None:
        numerator_text, denominator_text = split
        numerator_field, numerator_year = _extract_trailing_year(numerator_text)
        denominator_field, denominator_year = _extract_trailing_year(denominator_text)
        numerator = _lookup(
            _clean_field_surface(numerator_field),
            numerator_year or year,
            role="part",
        )
        denominator = _lookup(
            _clean_field_surface(denominator_field),
            denominator_year or year,
            role="whole",
        )
        return ExpressionPlan(
            kind="part_to_whole",
            route_operator="SHARE",
            root=ExpressionNode("divide", children=(numerator, denominator)),
            confidence=0.78,
            trace={"parser": "expression_plan.regex_v1"},
        )

    split = _parse_percentage_of_x_are_y(lowered)
    if split is not None:
        denominator_text, numerator_text, split_confidence = split
        denominator_field, denominator_year = _extract_leading_year(denominator_text)
        numerator_field, numerator_year = _extract_leading_year(numerator_text)
        year = numerator_year or denominator_year or year
        numerator = _lookup(_clean_field_surface(numerator_field), year, role="part")
        denominator = _lookup(_clean_field_surface(denominator_field), year, role="whole")
        return ExpressionPlan(
            kind="part_to_whole",
            route_operator="SHARE",
            root=ExpressionNode("divide", children=(numerator, denominator)),
            confidence=split_confidence,
            trace={"parser": "expression_plan.regex_v1"},
        )

    split = _parse_percentage_of_whole_with_text_condition(lowered)
    if split is not None:
        denominator_text, condition_text = split
        denominator_field, denominator_year = _extract_trailing_time_qualifier(
            denominator_text
        )
        denominator_field = _clean_field_surface(denominator_field)
        condition_field = _clean_field_surface(condition_text)
        if denominator_field and condition_field:
            effective_year = denominator_year or year
            numerator = _lookup(
                _normalize_space(f"{denominator_field} {condition_field}").lower(),
                effective_year,
                role="part",
            )
            denominator = _lookup(denominator_field, effective_year, role="whole")
            return ExpressionPlan(
                kind="part_to_whole",
                route_operator="SHARE",
                root=ExpressionNode("divide", children=(numerator, denominator)),
                confidence=0.83,
                trace={"parser": "expression_plan.regex_v1"},
            )

    field = ""
    match = re.search(
        r"\b(?:what\s+)?(?:portion|percentage|percent|proportion|fraction)\s+of\s+"
        r"(?:the\s+)?(?:total\s+)?(?P<field>.+?)\s+"
        r"(?:are|is|was|were)\s+",
        lowered,
        re.IGNORECASE,
    )
    if match:
        field = match.group("field")
    if not field:
        match = re.search(
            r"\b(?P<field>.+?)\s+as\s+a\s+percentage\s+of\s+(?:the\s+)?total\b",
            lowered,
            re.IGNORECASE,
        )
        if match:
            field = match.group("field")
    if not field:
        return None
    field = _clean_field_surface(field)
    numerator = _lookup(field, year, role="part")
    denominator = _lookup(f"total {field}", None, role="whole")
    return ExpressionPlan(
        kind="part_to_whole",
        route_operator="SHARE",
        root=ExpressionNode("divide", children=(numerator, denominator)),
        confidence=0.86,
        trace={"parser": "expression_plan.regex_v1"},
    )


def _parse_difference(query: str) -> ExpressionPlan | None:
    years = YEAR_RE.findall(query)
    if len(years) < 2:
        return None
    from_year, to_year = years[0], years[-1]
    year_prefix = _year_range_prefix()
    field = _field_for_compare_to_change(query, years)
    if field:
        left_year, right_year = from_year, to_year
    else:
        left_year, right_year = to_year, from_year
    if not field:
        field = _field_between(
        query,
        start_patterns=(
            r"difference\s+in\s+",
            r"difference\s+of\s+",
            r"change\s+in\s+",
            r"change\s+of\s+",
        ),
        end_pattern=rf"\s+(?:from|between)\s+{year_prefix}{re.escape(from_year)}\b",
        )
    if not field:
        field = _field_before_year_range(query, years)
    if not field:
        return None
    field = _append_trailing_year_range_qualifier(
        _clean_field_surface(field),
        query,
        years,
    )
    left = _lookup(field, left_year, role="left")
    right = _lookup(field, right_year, role="right")
    return ExpressionPlan(
        kind="difference",
        route_operator="DIFFERENCE",
        root=ExpressionNode("subtract", children=(left, right)),
        confidence=0.84,
        trace={"parser": "expression_plan.regex_v1"},
    )


def _parse_ratio(query: str) -> ExpressionPlan | None:
    year = _last_year(query)
    lowered = query.lower()
    per_match = re.search(
        r"\b(?P<numerator>.+?)\s+per\s+(?P<denominator>[^?,.]+)(?:\?|$)",
        lowered,
        re.IGNORECASE,
    )
    if per_match is not None:
        numerator = _clean_field_surface(per_match.group("numerator"))
        denominator = _clean_field_surface(per_match.group("denominator"))
        numerator = _strip_leading_question_context(numerator)
        denominator = _strip_trailing_question_context(denominator)
        if numerator and denominator:
            return ExpressionPlan(
                kind="ratio",
                route_operator="RATIO",
                root=ExpressionNode(
                    "divide",
                    children=(
                        _lookup(numerator, year, role="numerator"),
                        _lookup(denominator, year, role="denominator"),
                    ),
                ),
                confidence=0.72,
                trace={"parser": "expression_plan.regex_v1"},
            )

    match = re.search(
        r"\bratio\s+of\s+(?:the\s+)?(?P<numerator>.+?)\s+to\s+(?:the\s+)?(?P<denominator>.+?)(?:\?|$)",
        lowered,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"\b(?P<numerator>.+?)\s+to\s+(?:the\s+)?(?P<denominator>.+?)\s+ratio(?:\?|$)",
            lowered,
            re.IGNORECASE,
        )
    if match is None:
        return None
    numerator = _clean_field_surface(match.group("numerator"))
    denominator = _clean_field_surface(match.group("denominator"))
    if not numerator or not denominator:
        return None
    return ExpressionPlan(
        kind="ratio",
        route_operator="RATIO",
        root=ExpressionNode(
            "divide",
            children=(
                _lookup(numerator, year, role="numerator"),
                _lookup(denominator, year, role="denominator"),
            ),
        ),
        confidence=0.86,
        trace={"parser": "expression_plan.regex_v1"},
    )


def _parse_average(query: str) -> ExpressionPlan | None:
    years = YEAR_RE.findall(query)
    lowered = query.lower()
    match = re.search(
        r"\b(?:average|mean)\s+(?P<field>.+?)\s+from\s+"
        r"(?P<from_year>19\d{2}|20\d{2})\s+to\s+(?P<to_year>19\d{2}|20\d{2})",
        lowered,
        re.IGNORECASE,
    )
    if match is not None:
        field = _clean_field_surface(match.group("field"))
        if not field:
            return None
        children = tuple(_lookup(field, year, role="average_part") for year in _year_range(match.group("from_year"), match.group("to_year")))
        if len(children) < 2:
            return None
        return ExpressionPlan(
            kind="average",
            route_operator="AVG",
            root=ExpressionNode("average", children=children),
            confidence=0.84,
            trace={"parser": "expression_plan.regex_v1"},
        )

    if len(years) >= 2:
        match = re.search(
            r"\b(?:average|mean)\s+(?P<field>.+?)\s+"
            r"(?:for|during|in)\s+(?:the\s+)?(?:period\s+)?"
            r"(?:ended\s+)?(?:december\s+31,\s*)?"
            r"(?P<left>19\d{2}|20\d{2})\s+(?:and|,)\s+(?P<right>19\d{2}|20\d{2})",
            lowered,
            re.IGNORECASE,
        )
        if match is not None:
            field = _clean_field_surface(match.group("field"))
            if not field:
                return None
            return ExpressionPlan(
                kind="average",
                route_operator="AVG",
                root=ExpressionNode(
                    "average",
                    children=(
                        _lookup(field, match.group("left"), role="average_part"),
                        _lookup(field, match.group("right"), role="average_part"),
                    ),
                ),
                confidence=0.78,
                trace={"parser": "expression_plan.regex_v1"},
            )

    match = re.search(
        r"\bmidpoint\s+(?:earnings\s+exposure\s+)?between\s+(?:a\s+)?"
        r"(?P<left>[-+]?\d+(?:\.\d+)?\s*bp)\s+and\s+(?:a\s+)?"
        r"(?P<right>[-+]?\d+(?:\.\d+)?\s*bp)",
        lowered,
        re.IGNORECASE,
    )
    if match is not None:
        return ExpressionPlan(
            kind="midpoint",
            route_operator="AVG",
            root=ExpressionNode(
                "average",
                children=(
                    _lookup(match.group("left"), years[-1] if years else None, role="average_part"),
                    _lookup(match.group("right"), years[-1] if years else None, role="average_part"),
                ),
            ),
            confidence=0.82,
            trace={"parser": "expression_plan.regex_v1"},
        )
    return None


def _parse_sum(query: str) -> ExpressionPlan | None:
    year = _last_year(query)
    lowered = query.lower()
    years = YEAR_RE.findall(query)
    year_list_match = re.search(
        r"\b(?:total|sum)\s+(?:amount\s+)?(?:of\s+)?(?P<field>.+?)\s+"
        r"(?:during|from|for)\s+(?P<years>(?:19\d{2}|20\d{2})(?:\s*,\s*|\s+and\s+|,\s+and\s+)+(?:19\d{2}|20\d{2}).*?)(?:\?|$)",
        lowered,
        re.IGNORECASE,
    )
    if year_list_match is not None and len(years) >= 2:
        field = _clean_field_surface(year_list_match.group("field"))
        if not field:
            return None
        return ExpressionPlan(
            kind="sum",
            route_operator="SUM",
            root=ExpressionNode(
                "add",
                children=tuple(_lookup(field, item, role="addend") for item in years),
            ),
            confidence=0.84,
            trace={"parser": "expression_plan.regex_v1"},
        )

    prefix = re.search(
        r"\b(?:total|sum)\s+(?:amount\s+)?(?:of\s+)?(?P<body>.+?)(?:\?|$)",
        lowered,
        re.IGNORECASE,
    )
    if prefix is None:
        return None
    body = prefix.group("body")
    split = _split_sum_body(body)
    if split is None:
        return None
    left = _clean_field_surface(split[0])
    right = _clean_field_surface(_strip_year_surface(split[1]))
    if not left or not right:
        return None
    return ExpressionPlan(
        kind="sum",
        route_operator="SUM",
        root=ExpressionNode(
            "add",
            children=(
                _lookup(left, year, role="addend"),
                _lookup(right, year, role="addend"),
            ),
        ),
        confidence=0.8,
        trace={"parser": "expression_plan.regex_v1"},
    )


def _parse_lookup(query: str) -> ExpressionPlan | None:
    year = _last_year(query)
    if year is None:
        return None
    lowered = query.lower()
    year_surface = _last_year_surface(query) or year
    patterns = (
        rf"\b(?:what|how\s+much)\s+(?:was|is|were|are)\s+(?:the\s+)?(?P<field>.+?)\s+(?:in|for|during|as\s+of)\s+{re.escape(year_surface)}\b",
        rf"\b(?:in|for|during|as\s+of)\s+{re.escape(year_surface)}\s+(?:what|how\s+much)\s+(?:was|is|were|are)\s+(?:the\s+)?(?P<field>.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match is None:
            continue
        field = _clean_field_surface(match.group("field"))
        if not field:
            continue
        if _lookup_field_is_blocked(field):
            continue
        return ExpressionPlan(
            kind="lookup",
            route_operator="LOOKUP",
            root=_lookup(field, year, role="value"),
            confidence=0.72,
            trace={"parser": "expression_plan.regex_v1"},
        )
    return None


def _split_sum_body(body: str) -> tuple[str, str] | None:
    matches = list(re.finditer(r"\s+and\s+", body, re.IGNORECASE))
    if not matches:
        return None
    preferred_terms = ("cost", "costs", "charge", "charges", "expense", "expenses", "liabilit", "asset", "assets")
    for match in reversed(matches):
        right = body[match.end():].strip()
        if any(term in right for term in preferred_terms):
            return body[:match.start()], right
    match = matches[-1]
    return body[:match.start()], body[match.end():]


def _lookup(field_surface: str, time_surface: str | None, *, role: str) -> ExpressionNode:
    return ExpressionNode(
        "lookup",
        evidence_query=EvidenceQuery(
            field_surface=field_surface,
            time_surface=time_surface,
            role=role,
        ),
    )


def _parse_percentage_of_x_are_y(query: str) -> tuple[str, str, float] | None:
    match = re.search(
        r"\b(?:what\s+)?(?:percentage|percent|portion|proportion|fraction)\s+of\s+"
        r"(?P<whole>.+?)\s+(?:are|is|was|were)\s+(?P<part>.+?)\??$",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    part = match.group("part")
    if re.fullmatch(
        r"(?:due|paid|payable)?\s*(?:in|by|through|during|within)?\s*"
        r"(?:fiscal\s+year\s+of\s+)?(?:the\s+)?(?:next\s+)?(?:\d+|19\d{2}|20\d{2}|months?).*",
        part.strip(),
        re.IGNORECASE,
    ):
        return None
    return match.group("whole"), part, 0.88


def _parse_percentage_of_whole_with_text_condition(query: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?:what\s+(?:is|was|are|were)\s+(?:the\s+)?)?"
        r"(?:percentage|percent|portion|proportion|fraction)\s+of\s+"
        r"(?:the\s+)?(?P<body>.+?)(?:\?|$)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    body = match.group("body")
    condition_match = re.search(
        r"\s+(?P<condition>"
        r"(?:held|located|stored|kept|generated|derived|attributable|related|subject|"
        r"due|payable|outside|inside)\b.+)$",
        body,
        re.IGNORECASE,
    )
    if condition_match is None:
        return None
    whole = body[: condition_match.start()].strip()
    condition = condition_match.group("condition").strip()
    if not whole or not condition:
        return None
    return whole, condition


def _parse_percent_of_part_as_total(query: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?:what\s+)?(?:percent|percentage)\s+of\s+(?:the\s+)?"
        r"(?P<part>.+?)\s+(?:as\s+part\s+of|as\s+a\s+part\s+of|to)\s+"
        r"(?:the\s+)?(?P<whole>total.+?)(?:\?|$)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group("part"), match.group("whole")


def _parse_percent_of_part_to_whole(query: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?:in\s+(?:19\d{2}|20\d{2})\s+)?"
        r"(?:what\s+(?:was|is|were|are)\s+(?:the\s+)?)?"
        r"(?:percent|percentage)\s+of\s+(?:the\s+)?"
        r"(?P<part>.+?)\s+to\s+(?:the\s+)?(?P<whole>.+?)(?:\?|$)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    if re.search(r"\b(?:is|are|was|were)\s+due$", match.group("part"), re.IGNORECASE):
        return None
    return match.group("part"), match.group("whole")


def _parse_x_as_percentage_of_total_y(query: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?P<part>.+?)\s+as\s+a\s+percentage\s+of\s+(?:the\s+)?(?P<whole>total.+?)(?:\?|$)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group("part"), match.group("whole")


def _extract_leading_year(value: str) -> tuple[str, str | None]:
    match = re.match(r"\s*(?P<year>19\d{2}|20\d{2})\s+(?P<rest>.+)", value)
    if match is None:
        return value, None
    return match.group("rest"), match.group("year")


def _extract_trailing_year(value: str) -> tuple[str, str | None]:
    match = re.search(
        r"(?P<rest>.+?)\s+(?:in|for|during|as\s+of)\s+(?P<year>19\d{2}|20\d{2})\s*$",
        value,
        re.IGNORECASE,
    )
    if match is None:
        return value, None
    return match.group("rest"), match.group("year")


def _extract_trailing_time_qualifier(value: str) -> tuple[str, str | None]:
    match = re.search(
        r"(?P<rest>.+?)\s+"
        r"(?:at|as\s+of|on)\s+"
        r"(?:(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2},?\s+)?"
        r"(?P<year>19\d{2}|20\d{2})\s*$",
        value,
        re.IGNORECASE,
    )
    if match is not None:
        return match.group("rest"), match.group("year")
    return _extract_trailing_year(value)


def _is_percent_change(lowered: str) -> bool:
    return bool(
        re.search(r"\b(?:percent|percentage)\s+change\b", lowered)
        or re.search(r"\b(?:percent|percentage)\s+of\s+the\s+change\b", lowered)
        or re.search(r"\b(?:percent|percentage)\s+net\s+change\b", lowered)
        or re.search(r"\b(?:net\s+)?(?:percent|percentage)\s+change\b", lowered)
    )


def _is_growth_rate(lowered: str) -> bool:
    return bool(re.search(r"\bgrowth\s+(?:rate|observed)\b", lowered))


def _is_increase_decrease_change(lowered: str) -> bool:
    return bool(
        _is_percent_increase_decrease(lowered)
        or re.search(r"\b(?:increase|decrease|change|variation|fluctuation)\s+(?:in|of)\b", lowered)
    )


def _is_percent_increase_decrease(lowered: str) -> bool:
    return bool(re.search(r"\b(?:percent|percentage)\s+(?:increase|decrease|dropped)\b", lowered))


def _is_part_to_whole(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(?:what\s+)?(?:portion|percentage|percent|proportion|fraction)\s+of\b"
            r"|\bas\s+a\s+percentage\s+of\b",
            lowered,
        )
    )


def _is_difference(lowered: str) -> bool:
    return bool(
        (
            re.search(r"\bdifference\s+in\b|\bdifference\s+of\b", lowered)
            and re.search(r"\bbetween\b", lowered)
        )
        or re.search(r"\b(?:net\s+)?change\s+in\b.*\b(?:compare|compared)\s+to\b", lowered)
    )


def _is_ratio(lowered: str) -> bool:
    return bool(re.search(r"\bratio\s+of\b.*\bto\b|\bto\b.*\bratio\b|\bper\b", lowered))


def _is_average(lowered: str) -> bool:
    return bool(re.search(r"\b(?:average|mean|midpoint)\b", lowered))


def _is_sum(lowered: str) -> bool:
    if re.search(r"\b(?:difference|change|increase|decrease|growth|decline)\b", lowered):
        return False
    return bool(re.search(r"\b(?:total|sum)\b.+\band\b", lowered))


def _is_lookup(lowered: str) -> bool:
    if re.search(
        r"\b(?:difference|change|increase|decrease|growth|decline|ratio|percentage|percent|portion|average|mean|sum)\b",
        lowered,
    ):
        return False
    return bool(re.search(r"\b(?:what|how\s+much)\s+(?:was|is|were|are)\b", lowered))


def _field_between(
    query: str,
    *,
    start_patterns: tuple[str, ...],
    end_pattern: str,
) -> str:
    for start in start_patterns:
        match = re.search(start + r"(?P<field>.+?)" + end_pattern, query, re.IGNORECASE)
        if match:
            return match.group("field")
    return ""


def _field_before_year_range(query: str, years: list[str]) -> str:
    first_year = years[0]
    before_year = re.split(
        rf"\b(?:from|between)\s+{_year_range_prefix()}{re.escape(first_year)}\b",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = _strip_leading_question_context(before_year)
    cleaned = re.sub(
        r"^(?:percent|percentage)\s+change\s+(?:in|of)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:growth\s+(?:rate|observed)\s+(?:in|of)|"
        r"(?:increase|decrease|change|variation|fluctuation)\s+(?:in|of)|"
        r"difference\s+(?:in|of))\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _year_range_prefix() -> str:
    return r"(?:(?:fiscal|fiscal\s+year|fy)\s+)?"


def _append_trailing_year_range_qualifier(
    field: str,
    query: str,
    years: list[str],
) -> str:
    if not field or len(years) < 2:
        return field
    final_year = years[-1]
    match = re.search(
        rf"\b{re.escape(final_year)}\b\s+"
        r"(?P<qualifier>(?:at|on|as\s+of)\s+[^?.;,]+)",
        query,
        re.IGNORECASE,
    )
    if match is None:
        return field
    qualifier = _clean_field_surface(match.group("qualifier"))
    if not qualifier:
        return field
    field_terms = _normalized_term_set(field)
    qualifier_terms = _normalized_term_set(qualifier)
    if qualifier_terms <= field_terms:
        return field
    return _normalize_space(f"{field} {qualifier}").lower()


def _normalized_term_set(value: str) -> set[str]:
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return {term for term in normalized.split("_") if term}


def _field_for_compare_to_change(query: str, years: list[str]) -> str:
    left_year = years[0]
    match = re.search(
        rf"\b(?:net\s+)?change\s+in\s+(?P<field>.+?)\s+in\s+{re.escape(left_year)}\b"
        r".*?\b(?:compare|compared)\s+to\b",
        query,
        re.IGNORECASE,
    )
    return match.group("field") if match else ""


def _clean_field_surface(value: str) -> str:
    cleaned = _normalize_space(value)
    cleaned = re.sub(r"^(?:the|a|an|company'?s)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the\s+)+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the\s+)?amount\s+spent\s+for\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the\s+)?amount\s+of\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the\s+)?value\s+of\s+(?:the\s+)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+as$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+expressed$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:from|between|for|in|of)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" ?.,")
    return cleaned


def _lookup_field_is_blocked(field: str) -> bool:
    normalized = field.lower()
    return bool(
        re.search(
            r"\b(?:difference|change|increase|decrease|growth|decline|ratio|percentage|percent|portion|average|mean|sum)\b",
            normalized,
        )
    )


def _strip_leading_question_context(value: str) -> str:
    cleaned = re.sub(
        r"^(?:what|which|how\s+much)\s+(?:was|is|were|are)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^.*?\bmeasured\s+by\s+", "", cleaned, flags=re.IGNORECASE)
    return _clean_field_surface(cleaned)


def _strip_trailing_question_context(value: str) -> str:
    cleaned = re.sub(
        r"\s+(?:was|is|were|are)\s+(?:what|which|how\s+much)(?:\s+in\s+(?:19\d{2}|20\d{2}))?$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+in\s+(?:19\d{2}|20\d{2})$", "", cleaned, flags=re.IGNORECASE)
    return _clean_field_surface(cleaned)


def _last_year(query: str) -> str | None:
    years = YEAR_RE.findall(query)
    if years:
        return years[-1]
    short_years = _short_fiscal_years(query)
    return short_years[-1] if short_years else None


def _last_year_surface(query: str) -> str | None:
    year_matches = list(YEAR_RE.finditer(query))
    short_matches = list(SHORT_FISCAL_YEAR_RE.finditer(query))
    matches = [*year_matches, *short_matches]
    if not matches:
        return None
    return max(matches, key=lambda match: match.start()).group(0)


def _short_fiscal_years(query: str) -> list[str]:
    return [
        _normalize_short_year(match.group("year"))
        for match in SHORT_FISCAL_YEAR_RE.finditer(query)
    ]


def _normalize_short_year(value: str) -> str:
    year = int(value)
    return str(2000 + year)


def _strip_year_surface(value: str) -> str:
    return SHORT_FISCAL_YEAR_RE.sub(" ", value)


def _year_range(from_year: str, to_year: str) -> list[str]:
    start = int(from_year)
    end = int(to_year)
    step = 1 if end >= start else -1
    return [str(year) for year in range(start, end + step, step)]


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
