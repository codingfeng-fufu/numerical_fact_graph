from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graph_numeric.core.query_cleaning import strip_negative_distractors


@dataclass(frozen=True)
class RoutingResult:
    operator: str
    confidence: float
    intent: str
    route_type: str = "hard"
    fallback_operators: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator": self.operator,
            "confidence": self.confidence,
            "intent": self.intent,
            "route_type": self.route_type,
            "fallback_operators": self.fallback_operators,
            "trace": self.trace,
        }


# (operator, intent, priority) — higher priority matches first in tie-breaking
OPERATOR_PATTERNS: list[tuple[str, str, int, re.Pattern]] = [
    (
        "YEAR_LIST",
        "lookup",
        15,
        re.compile(
            r"\bwhich\s+fiscal\s+years\b"
            r"|\bin\s+which\s+(?:fiscal\s+)?years\b"
            r"|\bwhat\s+(?:fiscal\s+)?years\b.*\b(?:provided|presented|shown|included|available)\b",
            re.IGNORECASE,
        ),
    ),
    # Counting patterns
    (
        "COUNT",
        "condition-counting",
        13,
        re.compile(r"\b(?:companies|firms|entities|issuers)\b\s+with\b.*\b(?:>=|<=|>|<|at\s+or\s+above|at\s+or\s+below|at\s+least|at\s+most|no\s+less\s+than|no\s+more\s+than)\b", re.IGNORECASE),
    ),
    (
        "COUNT",
        "condition-counting",
        10,
        re.compile(r"(多少[家个项条]|几[家个项条]|(?:公司|企业|记录|年份|项目|条目).*数量是多少|[Hh]ow many|number of\s+(?:companies|firms|entities|records|years|items)|count of|count the|count\s+.*\b(?:entities|companies|firms|issuers|records|years|items|where|whose|with)\b|统计.*数量)", re.IGNORECASE),
    ),
    (
        "COUNT",
        "condition-counting",
        9,
        re.compile(r"([Cc]ount|[Nn]umber)\b.*(?:>=|<=|==|>|<|超过|不低于|不高于|大于|小于|等于|exceed|no\s+less\s+than|no\s+more\s+than|at\s+least|at\s+most)|满足.*(?:>=|<=|==|>|<|超过|不低于|不高于|大于|小于|等于).*数量", re.IGNORECASE),
    ),
    # Ranking patterns
    # TOP_K: explicit digit count — priority 11 beats COUNT (几家) and ARGMAX (which...highest)
    (
        "TOP_K",
        "ranking",
        11,
        re.compile(r"(前\s*\d+\s*[名位家个]|[Tt]op\s*\d+|which\s+\d+\b|\d+\s*(?:家公司|家企业|companies|firms|entities)\b)"),
    ),
    (
        "RANK",
        "ranking",
        12,
        re.compile(r"(排名前\s*\d+|排名|排行榜|[Rr]ank(?:ing)?\b|leaderboard)"),
    ),
    (
        "ARGMIN",
        "ranking",
        13,
        re.compile(r"\b(?:laggard|weakest|lowest performer|bottom performer)\b", re.IGNORECASE),
    ),
    (
        "ARGMIN",
        "ranking",
        11,
        re.compile(r"(排名最后\b|垫底的是哪[家个]|(?:最低|最小|最少).*?(?:公司|企业|实体).*?(?:谁|哪[家个]?)|[哪谁][家个]\S*.*(?:最低|最小|最少)|which\b.*\b(?:lowest|least|smallest|min(?:imum)?|leaner)\b)", re.IGNORECASE),
    ),
    (
        "ARGMIN",
        "ranking",
        12,
        re.compile(r"\b(?:in\s+)?which\s+year\b.*\b(?:smaller|lower|lowest|least|smallest|min(?:imum)?)\b|\bwhat\s+year\b.*\b(?:smaller|lower|lowest|least|smallest|min(?:imum)?)\b", re.IGNORECASE),
    ),
    # ARGMAX: single-entity asking — priority 11 for unambiguous entity cues
    (
        "ARGMAX",
        "ranking",
        13,
        re.compile(r"\b(?:leader|leading|strongest|top performer)\b(?!\s*\d)", re.IGNORECASE),
    ),
    (
        "ARGMAX",
        "ranking",
        12,
        re.compile(r"\b(?:in\s+)?which\s+year\b.*\b(?:longer|longest|larger|higher|highest|most|largest|max(?:imum)?)\b|\bwhat\s+year\b.*\b(?:longer|longest|larger|higher|highest|most|largest|max(?:imum)?)\b", re.IGNORECASE),
    ),
    (
        "ARGMAX",
        "ranking",
        11,
        re.compile(r"(领先的是哪[家个]|排名第一\b|[Tt]op(?!\s*\d).*\bcompan|which\b.*\b(?:lead|led|leads)\b|[哪谁][家个]\S*.*(?:最高|最大|最多))", re.IGNORECASE),
    ),
    (
        "ARGMAX",
        "ranking",
        10,
        re.compile(r"(最高|最大|最多).*是[谁哪]|which\b.*\b(?:highest|most|largest|max(?:imum)?|leads?|higher)\b|identify\b.*\b(?:strongest|highest|largest)\b", re.IGNORECASE),
    ),
    (
        "TOP_K",
        "ranking",
        9,
        re.compile(r"(前\s*[三五四3-9]|[Tt]op\s*\d)"),
    ),
    # Growth / change patterns
    (
        "PERCENT_CHANGE",
        "change-over-time",
        13,
        re.compile(r"(percent change\b|percentage change|百分比变化|变化百分比)", re.IGNORECASE),
    ),
    (
        "PERCENT_CHANGE",
        "change-over-time",
        13,
        re.compile(
            r"\bpercent(?:age)?\s+of\s+(?:the\s+)?(?:change|decline|increase|decrease)\b.*\bfrom\b.*\bto\b"
            r"|\b(?:change|decline|increase|decrease)\b.*\bfrom\b.*\bto\b.*\bpercent(?:age)?\s+of\s+(?:the\s+)?(?:change|decline|increase|decrease)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "GROWTH",
        "change-over-time",
        10,
        re.compile(r"(增长|下降|变化|增速|growth|change|increase|decrease).*(?:率|rate|percent|%)"),
    ),
    (
        "GROWTH",
        "change-over-time",
        9,
        re.compile(r"(增长了|下降了|变化了|同比|环比|year.over.year|YoY|QoQ|compared to|\bgr(?:ew|owth|ow)\b|\bincrease\b.*\bfrom\b|[Pp]ercent change\b)", re.IGNORECASE),
    ),
    (
        "TREND",
        "change-over-time",
        13,
        re.compile(r"\b(?:path|trajectory|pattern)\b\s+(?:across|over|from)\b", re.IGNORECASE),
    ),
    (
        "TREND",
        "change-over-time",
        10,
        re.compile(r"(趋势|走势|变化趋势|[Tt]rend|[Tt]rajectory|[Pp]attern.*over\s+time)"),
    ),
    (
        "TREND",
        "change-over-time",
        9,
        re.compile(r"(上升还是下降|[Ii]ncreasing\s+or\s+decreasing|[Ii]ncreasing.*[Dd]ecreasing)"),
    ),
    # DIFFERENCE: explicit gap/difference signals at priority 10 beat MAX's "largest"
    # Use "gap/difference + between" to avoid stealing GROWTH's "grow between X and Y"
    (
        "COMPARE",
        "comparison",
        11,
        re.compile(r"(比较|对比|[Cc]ompare\b|which\b.*\b(?:larger|higher|smaller|lower)\b|哪个更(?:高|低|大|小|多|少))"),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        12,
        re.compile(r"\bdifference\s+in\b.*\b(?:return|percent|percentage|rate)\b", re.IGNORECASE),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        12,
        re.compile(r"\bdifference\s+between\b.*\b(?:basic|diluted)\b.*\bshares?\b", re.IGNORECASE),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        10,
        re.compile(r"([Gg]ap\b.*[Bb]etween\b|[Dd]ifference\b.*[Bb]etween\b|[Bb]etween\b.*\band\b.*(?:差|gap|difference))"),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        10,
        re.compile(r"\bchange\s+(?:in|of)\b.*\b(?:from\b.*20\d{2}\s+to\s+20\d{2}|between\s+20\d{2}\s+and\s+20\d{2})\b", re.IGNORECASE),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        11,
        re.compile(r"\bincrease\s*/\s*\(?decrease\)?\s+in\b.*\bfrom\s+20\d{2}\s+to\s+20\d{2}\b", re.IGNORECASE),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        12,
        re.compile(r"\bchange\s+(?:in|of)\b.*\bin\s+20\d{2}\b.*\bfrom\s+20\d{2}\b", re.IGNORECASE),
    ),
    (
        "DIFFERENCE",
        "change-over-time",
        8,
        re.compile(r"(差多少|多多少|少多少|差值|差距|相差|之差|[Mm]inus\b|how much more)"),
    ),
    # Ratio patterns
    (
        "MARGIN",
        "ratio",
        13,
        re.compile(r"(profitability\s+percentage|profitability\s+ratio)", re.IGNORECASE),
    ),
    (
        "MARGIN",
        "ratio",
        12,
        re.compile(r"([Mm]argin\b|利润率|净利率|毛利率|operating margin|profit margin|net margin)"),
    ),
    (
        "SHARE",
        "ratio",
        14,
        re.compile(
            r"(proportion\s+of\s+total|as\s+a\s+share\s+of|as\s+a\s+percentage\s+of|percentage\s+of\s+(?:the\s+)?total)",
            re.IGNORECASE,
        ),
    ),
    (
        "SHARE",
        "ratio",
        11,
        re.compile(
            r"(占比|份额|占总|share\b|proportion\s+of\s+total|as\s+a\s+share\s+of|as\s+a\s+percentage\s+of|percentage\s+of\s+(?:the\s+)?total)",
            re.IGNORECASE,
        ),
    ),
    (
        "RATIO",
        "ratio",
        13,
        re.compile(r"\bdivided\s+by\b|(?<!-)\bover\b(?!-)", re.IGNORECASE),
    ),
    (
        "RATIO",
        "ratio",
        12,
        re.compile(r"\bpercent(?:age)?\s+of\b(?!\s+(?:the\s+)?total\b).*\b(?:as\s+part\s+of|to)\b", re.IGNORECASE),
    ),
    (
        "RATIO",
        "ratio",
        12,
        re.compile(r"\bwhat\s+percent(?:age)?\s+of\b.*\b(?:were|was|are|is|where)\b", re.IGNORECASE),
    ),
    (
        "RATIO",
        "ratio",
        10,
        re.compile(r"([占比例]比[率]?|比例|比率|除以|\bratio\b|\bproportion\b|\bfraction\b)", re.IGNORECASE),
    ),
    (
        "RATIO",
        "ratio",
        9,
        re.compile(r"(\S{2,}[率比])\s*(是|为|是多少)?"),
    ),
    # Aggregation patterns
    (
        "SUM",
        "aggregation",
        14,
        re.compile(r"\bhow\s+much\b.*\b(?:companies|firms|entities|issuers)\b.*\b(?:report|reported|recorded)\b", re.IGNORECASE),
    ),
    (
        "SUM",
        "aggregation",
        13,
        re.compile(r"\b(?:portfolio\s+rollup|rollup|roll-up)\b", re.IGNORECASE),
    ),
    (
        "SUM",
        "aggregation",
        10,
        re.compile(r"(总共|合计|总和|总量|总收入|总营收|总营业利润|总经营利润|总净利润|一共|加起来|[Tt]otal|\b[Ss]um\b|[Aa]ggregate|[Cc]ombined\b|[Aa]dd\s+up)"),
    ),
    (
        "AVG",
        "aggregation",
        11,
        re.compile(r"(平均|均值|平均数|[Aa]verage|[Mm]ean|[Pp]er-company|[Tt]ypical)"),
    ),
    # Lookup patterns — explicit retrieval verbs at priority 10; question forms at 7
    (
        "LOOKUP",
        "lookup",
        13,
        re.compile(r"\b(?:value\s+on\s+record|on\s+record|recorded\s+value)\b", re.IGNORECASE),
    ),
    (
        "LOOKUP",
        "lookup",
        14,
        re.compile(r"\b(?:average\s+)?share\s+price\b|\bprice\s+per\s+share\b", re.IGNORECASE),
    ),
    (
        "LOOKUP",
        "lookup",
        14,
        re.compile(r"\bshares?\b.*\bavailable\s+for\s+future\s+issuance\b|\bavailable\s+for\s+future\s+issuance\b.*\bshares?\b", re.IGNORECASE),
    ),
    (
        "LOOKUP",
        "lookup",
        10,
        re.compile(r"(查询|[Rr]etrieve\b|show\s+source\s+evidence|cite\s+evidence)", re.IGNORECASE),
    ),
    (
        "LOOKUP",
        "lookup",
        7,
        re.compile(r"(是多少|是多少钱|\S+的\S+是|what (?:is|was)(?: the)?|how much\b|[Ff]iscal\s+year\b|\bFY\d{2,4}\b|\b20\d{2}\b.*\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?\b.*\b(?:revenue|sales|net earnings|net profit|top-line)\b)", re.IGNORECASE),
    ),
    # Max/Min value — priority 9/8 to beat LOOKUP
    (
        "MAX",
        "comparison",
        13,
        re.compile(r"\b(?:ceiling|upper\s+bound)\b.*\bvalue\b", re.IGNORECASE),
    ),
    (
        "MAX",
        "comparison",
        9,
        re.compile(r"(最高[的]?(?:收入|利润|值|金额)?是多少|最大值|最高值|峰值|最高纪录|最大的一[家个]|[Mm]aximum|[Hh]ighest|[Pp]eak\b|[Ll]argest)"),
    ),
    (
        "MIN",
        "comparison",
        8,
        re.compile(r"(最低[的]?(?:收入|利润|值|金额)?是多少|最小值|最低值|最低记录|最小的|[Mm]inimum|[Ll]owest|[Ss]mallest|[Ff]loor\b)"),
    ),
    # Prediction — priority 11 so PREDICT verbs beat field-name collisions
    # (e.g. "total assets" → SUM, "historical trend" → TREND at priority 10)
    (
        "FORECAST",
        "prediction",
        13,
        re.compile(r"\b(?:forward\s+view|forward-looking\s+view)\b", re.IGNORECASE),
    ),
    (
        "FORECAST",
        "prediction",
        12,
        re.compile(r"([Ff]orecast\b|未来预测|预测未来|forecasted|forecasting)"),
    ),
    (
        "PREDICT",
        "prediction",
        12,
        re.compile(r"\b(?:outlook|projection)\b", re.IGNORECASE),
    ),
    (
        "PREDICT",
        "prediction",
        11,
        re.compile(r"(预测|[Pp]redict|[Ff]uture|预估|预计|[Pp]roject(?:ed|ion)\b|[Ee]stimated\b.*\b(?:trend|historical|forecast))"),
    ),
]

INTENT_PATTERNS: list[tuple[str, int, re.Pattern]] = [
    ("aggregation", 1, re.compile(r"(总共|合计|总和|total|sum|aggregate|全部)")),
    ("condition-counting", 1, re.compile(r"(多少|几个|count|number of|统计)")),
    ("ranking", 1, re.compile(r"(最高|最大|top|排名|前几|哪家|谁)")),
    ("change-over-time", 1, re.compile(r"(增长|下降|变化|增速|growth|change|同比|环比)")),
    ("ratio", 1, re.compile(r"(率|比|占比|ratio|margin|proportion)")),
    ("comparison", 1, re.compile(r"(差|difference|比较|compare|高低)")),
    ("lookup", 1, re.compile(r"(是多少|what is|how much)")),
    ("prediction", 1, re.compile(r"(预测|predict|forecast|预估)")),
]

_OP_TO_INTENT: dict[str, str] = {
    "SUM":        "aggregation",
    "AVG":        "aggregation",
    "COUNT":      "condition-counting",
    "MAX":        "comparison",
    "MIN":        "comparison",
    "ARGMIN":     "ranking",
    "ARGMAX":     "ranking",
    "TOP_K":      "ranking",
    "RANK":       "ranking",
    "RATIO":      "ratio",
    "SHARE":      "ratio",
    "MARGIN":     "ratio",
    "GROWTH":     "change-over-time",
    "PERCENT_CHANGE": "change-over-time",
    "TREND":      "change-over-time",
    "DIFFERENCE": "change-over-time",
    "COMPARE":    "comparison",
    "LOOKUP":     "lookup",
    "YEAR_LIST":  "lookup",
    "PREDICT":    "prediction",
    "FORECAST":   "prediction",
    "COMPOSITE":  "composite",
}


class RuleBasedRouter:
    """Cold-start operator router using keyword and pattern matching.

    No training data required. Uses OPERATOR_PATTERNS to match query text
    against known operator-indicating phrases.

    Falls back to SUM as the default operator when no pattern matches.
    """

    def __init__(
        self,
        patterns: list[tuple[str, str, int, re.Pattern]] | None = None,
    ) -> None:
        self.patterns = patterns or OPERATOR_PATTERNS

    def route(self, query: str) -> RoutingResult:
        if not query.strip():
            raise ValueError("Query must be non-empty.")

        match_query = strip_negative_distractors(query)
        composite = _detect_composite_route(match_query)
        if composite is not None:
            return composite

        matches: list[tuple[str, str, int, float]] = []
        for operator, intent, priority, pattern in self.patterns:
            match = pattern.search(match_query)
            if match:
                match_len = len(match.group(0))
                confidence = min(1.0, 0.6 + 0.05 * priority + 0.02 * match_len)
                matches.append((operator, intent, priority, confidence))

        if not matches:
            return RoutingResult(
                operator="SUM",
                confidence=0.35,
                intent="aggregation",
                fallback_operators=["LOOKUP", "COUNT"],
                trace={"rule": "default_fallback", "reason": "no pattern matched"},
            )

        matches.sort(key=lambda x: (x[2], x[3]), reverse=True)
        best_operator, best_intent, _, best_confidence = matches[0]

        fallback = list(dict.fromkeys(
            op for op, _, _, _ in matches[1:3] if op != best_operator
        ))

        return RoutingResult(
            operator=best_operator,
            confidence=best_confidence,
            intent=best_intent,
            fallback_operators=fallback,
            trace={
                "rule": f"pattern_matched_{best_operator}",
                "match_query": match_query,
                "matched_patterns": [(op, intent, conf) for op, intent, _, conf in matches[:5]],
            },
        )

    def classify_intent(self, query: str) -> str:
        """Level 1 coarse classification: which problem type?"""
        if _detect_composite_route(strip_negative_distractors(query)) is not None:
            return "composite"
        for intent, _, pattern in sorted(INTENT_PATTERNS, key=lambda x: x[1]):
            if pattern.search(query):
                return intent
        return "aggregation"


class EmbeddingRouter:
    """Operator router backed by a fine-tuned multilingual sentence encoder.

    Loads the checkpoint produced by scripts/graph_numeric/training/train_embedding_router.py.
    Matches the RuleBasedRouter.route() interface exactly.

    Args:
        model_dir: directory containing best_model/ and label_map.json
        device:    'cuda', 'cpu', or None (auto-detect)
        batch_size: number of queries to encode at once (for bulk routing)
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        max_len: int = 128,
    ) -> None:
        import json
        import torch
        from pathlib import Path as _Path
        from transformers import AutoModel, AutoTokenizer

        if not all(hasattr(torch, attr) for attr in ("cuda", "device", "load", "no_grad")):
            raise RuntimeError(
                "EmbeddingRouter requires a full PyTorch installation. "
                "The imported 'torch' module is missing core runtime attributes."
            )

        model_dir = _Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"EmbeddingRouter model directory not found: {model_dir}\n"
                "Run scripts/graph_numeric/training/train_embedding_router.py first."
            )

        with (model_dir / "label_map.json").open(encoding="utf-8") as f:
            label_map: dict[str, int] = json.load(f)

        self._idx_to_op: dict[int, str] = {v: k for k, v in label_map.items()}
        self._num_classes = len(label_map)
        self._max_len = max_len

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        best_dir = model_dir / "best_model"
        self._tokenizer = AutoTokenizer.from_pretrained(best_dir)
        encoder = AutoModel.from_pretrained(best_dir)

        import torch.nn as nn

        class _Clf(nn.Module):
            def __init__(self, enc, hidden: int, n: int) -> None:
                super().__init__()
                self.encoder    = enc
                self.classifier = nn.Linear(hidden, n)
                self.dropout    = nn.Dropout(0.1)

            def forward(self, input_ids, attention_mask):
                out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(self.dropout(pooled))

        self._model = _Clf(encoder, encoder.config.hidden_size, self._num_classes)
        self._model.classifier.load_state_dict(
            torch.load(best_dir / "classifier_head.pt", map_location=self._device)
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def route(self, query: str) -> RoutingResult:
        if not query.strip():
            raise ValueError("Query must be non-empty.")

        import torch
        import torch.nn.functional as F

        enc = self._tokenizer(
            query,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1).squeeze(0)

        top_indices = probs.argsort(descending=True).tolist()
        best_idx    = top_indices[0]
        operator    = self._idx_to_op[best_idx]
        confidence  = float(probs[best_idx])

        fallback = [
            self._idx_to_op[i]
            for i in top_indices[1:3]
            if self._idx_to_op[i] != operator
        ]
        trace: dict[str, Any] = {
            "top3_probs": {self._idx_to_op[i]: round(float(probs[i]), 4) for i in top_indices[:3]}
        }
        route_type = "embedding"
        if operator == "COMPOSITE":
            rule_composite = _detect_composite_route(query)
            if rule_composite is not None:
                trace = {
                    **trace,
                    **(rule_composite.trace or {}),
                    "embedding_top3_probs": trace["top3_probs"],
                }
                fallback = rule_composite.fallback_operators or fallback
                route_type = "embedding_composite"

        return RoutingResult(
            operator=operator,
            confidence=confidence,
            intent=_OP_TO_INTENT.get(operator, "aggregation"),
            route_type=route_type,
            fallback_operators=fallback,
            trace=trace,
        )

    def classify_intent(self, query: str) -> str:
        return _OP_TO_INTENT.get(self.route(query).operator, "aggregation")


def _detect_composite_route(query: str) -> RoutingResult | None:
    if _has_ranking_signal(query) and _has_margin_signal(query):
        order = "ascending" if _has_lowest_signal(query) else "descending"
        first_op = "ARGMIN" if order == "ascending" else "ARGMAX"
        return RoutingResult(
            operator="COMPOSITE",
            confidence=0.95,
            intent="composite",
            route_type="composite",
            fallback_operators=["MARGIN", first_op],
            trace={
                "rule": "composite_rank_then_margin",
                "match_query": query,
                "composite_case": "arg_rank_then_margin",
                "steps": [
                    {"step_id": "s1", "operator": first_op, "order": order},
                    {"step_id": "s2", "operator": "MARGIN", "depends_on": {"entity": "$s1.entity"}},
                ],
            },
        )

    if _has_top_k_signal(query) and _has_sum_signal(query):
        return RoutingResult(
            operator="COMPOSITE",
            confidence=0.92,
            intent="composite",
            route_type="composite",
            fallback_operators=["TOP_K", "SUM"],
            trace={
                "rule": "composite_topk_then_sum",
                "match_query": query,
                "composite_case": "topk_then_sum",
                "steps": [
                    {"step_id": "s1", "operator": "TOP_K"},
                    {"step_id": "s2", "operator": "SUM", "depends_on": {"entities": "$s1.entities"}},
                ],
            },
        )

    if _has_ranking_signal(query) and _has_lookup_signal(query):
        order = "ascending" if _has_lowest_signal(query) else "descending"
        first_op = "ARGMIN" if order == "ascending" else "ARGMAX"
        return RoutingResult(
            operator="COMPOSITE",
            confidence=0.9,
            intent="composite",
            route_type="composite",
            fallback_operators=["LOOKUP", first_op],
            trace={
                "rule": "composite_rank_then_lookup",
                "match_query": query,
                "composite_case": "arg_rank_then_lookup",
                "steps": [
                    {"step_id": "s1", "operator": first_op, "order": order},
                    {"step_id": "s2", "operator": "LOOKUP", "depends_on": {"entity": "$s1.entity"}},
                ],
            },
        )

    return None


def _has_ranking_signal(query: str) -> bool:
    return bool(
        re.search(
            r"(最高|最大|最多|最低|最小|最少|领先|排名第一|\b(?:leader|leading|laggard|strongest|weakest)\b|which\b.*\b(?:highest|lowest|largest|smallest|lead|led|leads|maximum|minimum|higher|leaner)\b|identify\b.*\b(?:strongest|highest|largest)\b|(?:company|firm|entity|issuer|name)\b.*\b(?:highest|lowest|largest|smallest|maximum|minimum|strongest|leaner)\b|\b(?:highest|lowest|largest|smallest|maximum|minimum|strongest|leaner)\b.*\b(?:company|firm|entity|issuer|name)\b|top(?!\s*\d))",
            query,
            re.IGNORECASE,
        )
    )


def _has_top_k_signal(query: str) -> bool:
    return bool(
        re.search(
            r"(前\s*\d+\s*[名位家个]|排名前\s*\d+|[Tt]op\s*\d+|[Ll]eading\s*\d+|\d+\s*(?:家公司|家企业|companies|firms|entities)\b)",
            query,
        )
    )


def _has_lowest_signal(query: str) -> bool:
    return bool(re.search(r"(最低|最小|最少|lowest|least|smallest|minimum|laggard|weakest)", query, re.IGNORECASE))


def _has_margin_signal(query: str) -> bool:
    return bool(
        re.search(
            r"([Mm]argin\b|利润率|净利率|毛利率|operating margin|profit margin|net margin|profitability\s+percentage|profitability\s+ratio)",
            query,
            re.IGNORECASE,
        )
    )


def _has_sum_signal(query: str) -> bool:
    return bool(
        re.search(
            r"(总共|合计|总和|总量|一共|加起来|[Tt]otal|\b[Ss]um\b|[Aa]ggregate|[Cc]ombined\b|[Aa]dd\s+up|portfolio\s+rollup|rollup|roll-up)",
            query,
        )
    )


def _has_lookup_signal(query: str) -> bool:
    return bool(
        re.search(
            r"(是多少|是多少钱|查询|[Rr]etrieve\b|what (?:is|was)|how much\b)",
            query,
            re.IGNORECASE,
        )
    )


class HybridRouter:
    """Routes via RuleBasedRouter when confident, else falls back to EmbeddingRouter.

    Default threshold=0.5 routes rule fallbacks (conf=0.35) to embedding
    while keeping all pattern-matched cases (conf≥0.99) with the rule router.

    Args:
        model_dir:       directory containing best_model/ and label_map.json
        rule_threshold:  confidence cutoff; below this → EmbeddingRouter
        device:          'cuda', 'cpu', or None (auto-detect)
        max_len:         tokenizer max sequence length
    """

    def __init__(
        self,
        model_dir: str | Path,
        rule_threshold: float = 0.5,
        device: str | None = None,
        max_len: int = 128,
    ) -> None:
        import dataclasses as _dc
        self._dc = _dc
        self._rule = RuleBasedRouter()
        self._emb = EmbeddingRouter(model_dir, device=device, max_len=max_len)
        self._threshold = rule_threshold

    def route(self, query: str) -> RoutingResult:
        rule_result = self._rule.route(query)
        if rule_result.confidence >= self._threshold:
            return self._dc.replace(rule_result, route_type="rule")
        emb_result = self._emb.route(query)
        return self._dc.replace(
            emb_result,
            route_type="hybrid_emb",
            trace={
                **emb_result.trace,
                "rule_confidence": rule_result.confidence,
                "rule_operator": rule_result.operator,
            },
        )

    def classify_intent(self, query: str) -> str:
        return _OP_TO_INTENT.get(self.route(query).operator, "aggregation")
