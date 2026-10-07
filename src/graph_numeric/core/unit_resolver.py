from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


UNIT_CATEGORIES = (
    "money",
    "count",
    "percentage",
    "per_share",
    "shares",
    "area",
    "volume",
    "time",
    "capacity",
    "rate",
    "ratio_dimensionless",
    "unknown",
)

UNIT_SCALE_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(crore|cr)\b", re.IGNORECASE), 1e7, "crore"),
    (re.compile(r"\b(lakh|lac)\b", re.IGNORECASE), 1e5, "lakh"),
    (re.compile(r"\b(billion|bn|b)\b", re.IGNORECASE), 1e9, "billion"),
    (re.compile(r"\b(million|mn|m)\b", re.IGNORECASE), 1e6, "million"),
    (re.compile(r"\b(thousand|k)\b", re.IGNORECASE), 1e3, "thousand"),
]

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "money": (
        "usd", "eur", "cny", "rmb", "inr", "gbp", "jpy",
        "$", "€", "¥", "￥", "₹", "£",
        "dollar", "yuan", "rupee", "yen", "元", "美元", "人民币", "欧元", "卢比", "日元",
    ),
    "count": ("count", "数量", "个数", "人数", "家", "个"),
    "percentage": ("%", "percent", "percentage", "bps", "basis point", "百分", "基点"),
    "per_share": ("per share", "per-share", "每股"),
    "shares": ("shares", "share count", "ordinary shares", "common shares", "股数", "股份"),
    "area": ("sqft", "sqm", "square", "acre", "面积", "平方"),
    "volume": ("kwh", "mwh", "barrel", "产量", "发电"),
    "time": ("days", "hours", "minutes", "years", "天", "小时", "年"),
    "capacity": ("beds", "seats", "mw", "capacity", "床位"),
    "rate": ("rate", "growth", "margin", "率", "增长"),
}

OPERATION_COMPATIBILITY: dict[str, set[str]] = {
    "SUM": {"money", "count", "area", "volume", "time", "capacity"},
    "COUNT": set(),
    "AVG": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "MAX": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "MIN": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "ARGMAX": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "ARGMIN": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "TOP_K": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "PRODUCT": {
        "money",
        "count",
        "percentage",
        "per_share",
        "shares",
        "area",
        "volume",
        "time",
        "capacity",
        "rate",
        "ratio_dimensionless",
    },
    "RATIO": set(),  # any two categories allowed via DIVIDE
    "GROWTH": set(),  # two same-category values → ratio_dimensionless
    "DIFFERENCE": {"money", "count", "shares", "area", "volume", "time", "capacity"},
    "LOOKUP": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "YEAR_LIST": set(),
    "TREND": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
    "PREDICT": {"money", "count", "shares", "area", "volume", "time", "capacity", "percentage", "rate", "per_share"},
}


@dataclass(frozen=True)
class UnitInfo:
    unit_string: str
    unit_category: str
    unit_scale: float
    base_unit: str
    source: str = "inferred"
    currency_code: str | None = None


class UnitResolver:
    """Detect and normalize units for numerical values in document graphs.

    Extracted from _parse_number in document_graph.py, generalized to support
    all unit categories defined in the design document Section 8.
    """

    def detect(self, raw_text: str) -> UnitInfo:
        lowered = raw_text.casefold().strip()
        scale = 1.0
        scale_label = ""
        for pattern, factor, label in UNIT_SCALE_PATTERNS:
            if pattern.search(lowered):
                scale = factor
                scale_label = label
                break
        category = self._infer_category(lowered)
        currency = self._detect_currency(lowered)
        if self._is_per_share(lowered):
            category = "per_share"
        elif self._is_shares(lowered):
            category = "shares"
        if category == "percentage" and self._is_bps(lowered):
            scale = 0.01
            scale_label = "bps"
        base = self._extract_base(lowered, scale_label, category, currency)
        return UnitInfo(
            unit_string=raw_text,
            unit_category=category,
            unit_scale=scale,
            base_unit=base or raw_text,
            source="keyword_detection",
            currency_code=currency,
        )

    def _infer_category(self, lowered: str) -> str:
        scores: dict[str, int] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits:
                scores[category] = hits
        if not scores:
            return "unknown"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _detect_currency(self, lowered: str) -> str | None:
        patterns: tuple[tuple[str, str], ...] = (
            (r"\busd\b|us\$|\$|dollar|美元", "USD"),
            (r"\beur\b|€|euro|欧元", "EUR"),
            (r"\bcny\b|\brmb\b|¥|￥|yuan|人民币|元", "CNY"),
            (r"\binr\b|₹|rupee|卢比", "INR"),
            (r"\bgbp\b|£|pound", "GBP"),
            (r"\bjpy\b|yen|日元", "JPY"),
        )
        for pattern, code in patterns:
            if re.search(pattern, lowered):
                return code
        return None

    @staticmethod
    def _is_bps(lowered: str) -> bool:
        return bool(re.search(r"\bbps\b|basis\s+points?|基点", lowered))

    @staticmethod
    def _is_per_share(lowered: str) -> bool:
        return bool(re.search(r"per[-\s]?share|每股", lowered))

    @staticmethod
    def _is_shares(lowered: str) -> bool:
        return bool(re.search(r"\bshares?\b|股数|股份", lowered)) and not UnitResolver._is_per_share(lowered)

    def _extract_base(
        self,
        lowered: str,
        scale_label: str,
        category: str = "unknown",
        currency: str | None = None,
    ) -> str:
        if category == "percentage":
            return "%"
        if category == "per_share":
            return f"{currency or ''} per share".strip()
        if category == "shares":
            return "shares"
        if category == "money" and currency is not None:
            return currency
        cleaned = lowered
        for _, _, label in UNIT_SCALE_PATTERNS:
            cleaned = re.sub(rf"\b{label}\b", "", cleaned, flags=re.IGNORECASE)
        if scale_label == "bps":
            cleaned = re.sub(r"\bbps\b|basis\s+points?|基点", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or "unit"

    def normalize(self, value: float, from_unit: UnitInfo, to_unit: UnitInfo) -> float:
        if from_unit.unit_category != to_unit.unit_category:
            raise UnitIncompatibleError(from_unit, to_unit)
        if (
            from_unit.unit_category == "money"
            and from_unit.currency_code
            and to_unit.currency_code
            and from_unit.currency_code != to_unit.currency_code
        ):
            raise UnitIncompatibleError(from_unit, to_unit)
        return value * (from_unit.unit_scale / to_unit.unit_scale)

    def check_compatibility(self, operator: str, units: list[UnitInfo]) -> list[str]:
        warnings: list[str] = []
        allowed = OPERATION_COMPATIBILITY.get(operator, set())
        for unit in units:
            if unit.unit_category == "unknown":
                continue
            if allowed and unit.unit_category not in allowed:
                warnings.append(
                    f"Operator {operator} may be incompatible with unit category '{unit.unit_category}'"
                )
        if operator in ("SUM", "DIFFERENCE", "GROWTH") and len(units) >= 2:
            categories = {u.unit_category for u in units if u.unit_category != "unknown"}
            if len(categories) > 1:
                warnings.append(
                    f"Mismatched unit categories in {operator}: {sorted(categories)}. "
                    "Values will be used without normalization."
                )
        money_currencies = {
            unit.currency_code
            for unit in units
            if unit.unit_category == "money" and unit.currency_code
        }
        if len(money_currencies) > 1:
            warnings.append(
                f"Mismatched money currencies: {sorted(money_currencies)}. "
                "Cross-currency normalization is not supported."
            )
        return warnings

    @staticmethod
    def from_field_name(field_name: str) -> UnitInfo:
        """Infer unit from a field name using keyword heuristics.

        Delegates to the same detection logic but starts from a clean field name.
        """
        resolver = UnitResolver()
        return resolver.detect(field_name)


class UnitIncompatibleError(ValueError):
    def __init__(self, from_unit: UnitInfo, to_unit: UnitInfo) -> None:
        super().__init__(
            f"Cannot normalize {from_unit.unit_string} to {to_unit.unit_string}: "
            f"incompatible unit categories or currencies."
        )
        self.from_unit = from_unit
        self.to_unit = to_unit
