from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from graph_numeric.core.attribute_graph import field_aliases, normalize_identifier


@dataclass(frozen=True)
class SemanticProfile:
    value_type: str
    family: str
    confidence: float
    matched_terms: tuple[str, ...]


VALUE_TYPES = ("count", "money", "time", "rate", "area", "capacity", "volume", "unknown")
TYPE_COMPATIBILITY: dict[tuple[str, str], float] = {
    ("unknown", "unknown"): 0.55,
    ("unknown", "count"): 0.55,
    ("unknown", "money"): 0.55,
    ("unknown", "time"): 0.45,
    ("unknown", "rate"): 0.45,
    ("unknown", "area"): 0.55,
    ("unknown", "capacity"): 0.55,
    ("unknown", "volume"): 0.55,
    ("count", "unknown"): 0.60,
    ("money", "unknown"): 0.60,
    ("time", "unknown"): 0.50,
    ("rate", "unknown"): 0.50,
    ("area", "unknown"): 0.60,
    ("capacity", "unknown"): 0.60,
    ("volume", "unknown"): 0.60,
    ("count", "capacity"): 0.72,
    ("capacity", "count"): 0.72,
    ("count", "volume"): 0.70,
    ("volume", "count"): 0.70,
    ("money", "volume"): 0.62,
    ("volume", "money"): 0.62,
}

TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "money": (
        "revenue",
        "sales",
        "turnover",
        "profit",
        "income",
        "expense",
        "spend",
        "cost",
        "assets",
        "liabilities",
        "debt",
        "cash flow",
        "capex",
        "bookings",
        "booking value",
        "loan balance",
        "收入",
        "营收",
        "利润",
        "费用",
        "成本",
        "资产",
        "负债",
        "现金流",
        "贷款余额",
    ),
    "time": (
        "time",
        "duration",
        "latency",
        "resolution time",
        "cycle time",
        "days",
        "hours",
        "时长",
        "时间",
        "天数",
        "小时",
    ),
    "rate": (
        "rate",
        "ratio",
        "margin",
        "growth",
        "turnover",
        "utilization",
        "intensity",
        "churn",
        "率",
        "比例",
        "增长",
        "利用率",
    ),
    "area": (
        "area",
        "square footage",
        "square feet",
        "warehouse area",
        "storage area",
        "面积",
        "平方",
    ),
    "capacity": (
        "capacity",
        "beds",
        "available beds",
        "licensed beds",
        "bed capacity",
        "容量",
        "床位",
    ),
    "count": (
        "count",
        "number",
        "tickets",
        "ticket count",
        "cases",
        "requests",
        "employees",
        "headcount",
        "workforce",
        "stores",
        "sites",
        "locations",
        "fleet",
        "vehicles",
        "beds",
        "courses",
        "enrollments",
        "learners",
        "credits",
        "policies",
        "customers",
        "数量",
        "个数",
        "人数",
        "员工",
        "门店",
        "车辆",
        "保单",
        "客户",
    ),
    "volume": (
        "volume",
        "output",
        "generated",
        "generation",
        "energy output",
        "bookings",
        "credits issued",
        "产量",
        "发电量",
        "输出",
    ),
}

FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "support": ("support", "ticket", "tickets", "case", "cases", "request", "requests", "客服", "工单"),
    "store_site": ("store", "stores", "site", "sites", "location", "locations", "门店", "站点"),
    "fleet": ("fleet", "vehicle", "vehicles", "车辆", "车队"),
    "policy": ("policy", "policies", "insurance", "保单", "保险"),
    "energy": ("energy", "power", "generated", "generation", "发电", "能源"),
    "booking": ("booking", "bookings", "预订"),
    "warehouse": ("warehouse", "storage", "area", "square", "仓库", "面积"),
    "bed": ("bed", "beds", "capacity", "床位"),
    "course": ("course", "courses", "enrollment", "enrollments", "learners", "student", "学生", "课程"),
    "carbon": ("carbon", "credits", "offset", "碳"),
    "finance": ("revenue", "profit", "assets", "liabilities", "cash", "loan", "收入", "利润", "资产", "负债"),
    "people": ("employee", "employees", "headcount", "workforce", "人数", "员工"),
}

FIELD_TYPE_OVERRIDES: dict[str, str] = {
    "support_tickets": "count",
    "support_cost": "money",
    "ticket_resolution_time": "time",
    "fleet_size": "count",
    "active_sites": "count",
    "store_count": "count",
    "policy_count": "count",
    "warehouse_area": "area",
    "bed_capacity": "capacity",
    "course_enrollments": "count",
    "carbon_credits": "volume",
    "energy_output": "volume",
    "gross_bookings": "money",
    "booking_growth": "rate",
    "revenue_growth": "rate",
    "employee_growth": "rate",
    "asset_growth": "rate",
    "profit_margin": "rate",
    "asset_turnover": "rate",
    "site_utilization": "rate",
    "energy_intensity": "rate",
    "carbon_intensity": "rate",
    "churn_rate": "rate",
}

QUERY_TYPE_OVERRIDES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("support cost", "cost", "费用", "成本"), "money"),
    (("resolution time", "duration", "latency", "time", "时长", "时间"), "time"),
    (("square footage", "warehouse area", "storage area", "area", "面积"), "area"),
    (("bed capacity", "available beds", "licensed beds", "capacity", "床位"), "capacity"),
    (("growth", "margin", "ratio", "rate", "utilization", "intensity", "增长", "比例", "率"), "rate"),
    (("ticket count", "support tickets", "service requests", "support cases"), "count"),
    (("fleet size", "vehicles deployed", "active fleet", "vehicle count"), "count"),
    (("active sites", "site count", "operating sites", "live locations"), "count"),
    (("policy count", "policies in force", "active policies", "insurance policies"), "count"),
)


@lru_cache(maxsize=512)
def field_semantic_profile(field_name: str) -> SemanticProfile:
    text = _field_text(field_name)
    override_type = FIELD_TYPE_OVERRIDES.get(field_name)
    value_type, type_matches, type_score = _infer_type(text, preferred=override_type)
    family, family_matches, family_score = _infer_family(text)
    confidence = min(1.0, max(type_score, family_score))
    return SemanticProfile(
        value_type=value_type,
        family=family,
        confidence=confidence,
        matched_terms=tuple(dict.fromkeys([*type_matches, *family_matches])),
    )


@lru_cache(maxsize=65536)
def query_semantic_profile(query: str) -> SemanticProfile:
    text = query
    preferred = None
    normalized = _normalized_text(text)
    for terms, value_type in QUERY_TYPE_OVERRIDES:
        if any(_normalized_text(term) in normalized for term in terms):
            preferred = value_type
            break
    value_type, type_matches, type_score = _infer_type(text, preferred=preferred)
    family, family_matches, family_score = _infer_family(text)
    confidence = min(1.0, max(type_score, family_score))
    return SemanticProfile(
        value_type=value_type,
        family=family,
        confidence=confidence,
        matched_terms=tuple(dict.fromkeys([*type_matches, *family_matches])),
    )


@lru_cache(maxsize=1048576)
def semantic_compatibility_score(query: str, field_name: str) -> float:
    query_profile = query_semantic_profile(query)
    field_profile = field_semantic_profile(field_name)
    type_score = type_compatibility(query_profile.value_type, field_profile.value_type)
    family_score = family_compatibility(query_profile.family, field_profile.family)
    confidence = max(query_profile.confidence, field_profile.confidence)
    return float((0.72 * type_score) + (0.28 * family_score * confidence))


def type_compatibility(query_type: str, field_type: str) -> float:
    if query_type == field_type and query_type != "unknown":
        return 1.0
    return TYPE_COMPATIBILITY.get((query_type, field_type), 0.12 if query_type != "unknown" and field_type != "unknown" else 0.55)


def family_compatibility(query_family: str, field_family: str) -> float:
    if query_family == "unknown" or field_family == "unknown":
        return 0.55
    return 1.0 if query_family == field_family else 0.18


def _field_text(field_name: str) -> str:
    return " ".join([field_name, field_name.replace("_", " "), *field_aliases(field_name)])


def _infer_type(text: str, preferred: str | None = None) -> tuple[str, list[str], float]:
    if preferred is not None:
        matches = _matches_for_type(text, preferred)
        return preferred, matches or [preferred], 1.0
    scores = {
        value_type: _matches_for_type(text, value_type)
        for value_type in VALUE_TYPES
        if value_type != "unknown"
    }
    best_type = "unknown"
    best_matches: list[str] = []
    for value_type, matches in scores.items():
        if len(matches) > len(best_matches):
            best_type = value_type
            best_matches = matches
    if not best_matches:
        return "unknown", [], 0.0
    return best_type, best_matches, min(1.0, 0.45 + 0.15 * len(best_matches))


def _infer_family(text: str) -> tuple[str, list[str], float]:
    scores = {
        family: _matches_for_terms(text, terms)
        for family, terms in FAMILY_KEYWORDS.items()
    }
    best_family = "unknown"
    best_matches: list[str] = []
    for family, matches in scores.items():
        if len(matches) > len(best_matches):
            best_family = family
            best_matches = matches
    if not best_matches:
        return "unknown", [], 0.0
    return best_family, best_matches, min(1.0, 0.45 + 0.15 * len(best_matches))


def _matches_for_type(text: str, value_type: str) -> list[str]:
    return _matches_for_terms(text, TYPE_KEYWORDS[value_type])


def _matches_for_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    normalized = _normalized_text(text)
    matches = []
    for term in terms:
        normalized_term = _normalized_text(term)
        if normalized_term and normalized_term in normalized:
            matches.append(term)
    return matches


def _normalized_text(text: str) -> str:
    return normalize_identifier(text).replace("_", " ")
