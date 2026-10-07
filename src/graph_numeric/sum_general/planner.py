from __future__ import annotations

import re
from typing import TYPE_CHECKING

from graph_numeric.core.attribute_graph import AttributeValueGraph, normalize_identifier
from graph_numeric.operators.operator_plan import Filter, OperatorPlan, Slot

if TYPE_CHECKING:
    from graph_numeric.learning.field_grounder import FieldGrounder

YEAR_PATTERN = re.compile(r"(19\d{2}|20\d{2})")


class SchemaGeneralSumPlanner:
    """Pointer-style SUM planner over fields available in the input graph.

    The planner does not classify into a fixed field vocabulary. It scores each
    field present in the graph against the query, then emits a plan that the
    executor can evaluate exactly.

    Uses FieldGrounder for field selection, extractable as an independent module.
    Accepts raw FieldScorer objects for backward compatibility.
    """

    def __init__(self, field_grounder: FieldGrounder | None = None) -> None:
        from graph_numeric.learning.field_grounder import FieldGrounder as FG

        if field_grounder is None:
            self.field_grounder = FG.from_lexical()
        elif isinstance(field_grounder, FG):
            self.field_grounder = field_grounder
        else:
            self.field_grounder = _FieldScorerAdapter(field_grounder)

    @property
    def field_scorer(self):
        fg = self.field_grounder
        if isinstance(fg, _FieldScorerAdapter):
            return fg._scorer
        return fg._scorer

    def plan(self, query: str, graph: AttributeValueGraph) -> OperatorPlan:
        year = self._select_year(query, graph)
        industry = self._select_industry(query, graph)
        grounding = self.field_grounder.ground(query, graph.fields)
        selected_tokens = graph.select(
            field_name=grounding.field_name, year=year, industry=industry
        )

        filters: list[Filter] = []
        if year is not None:
            filters.append(
                Filter(type="time", surface=str(year), grounded_key="year", grounded_value=year, confidence=0.99)
            )
        if industry is not None:
            filters.append(
                Filter(
                    type="entity_metadata",
                    surface=industry,
                    grounded_key="industry",
                    grounded_value=industry,
                    confidence=0.95,
                )
            )

        return OperatorPlan(
            operator="SUM",
            slots={
                "target_field": Slot(
                    surface=query,
                    grounded_value=grounding.field_name,
                    confidence=grounding.confidence,
                ),
                "year": Slot(surface=str(year) if year else "", grounded_value=year, confidence=0.99),
                "industry": Slot(
                    surface=industry or "", grounded_value=industry, confidence=0.95
                ),
            },
            filters=filters,
            confidence=grounding.confidence,
            trace=grounding.trace,
        )

    def _select_year(self, query: str, graph: AttributeValueGraph) -> int | None:
        match = YEAR_PATTERN.search(query)
        if match:
            return int(match.group(1))
        return graph.years[-1] if graph.years else None

    def _select_industry(self, query: str, graph: AttributeValueGraph) -> str | None:
        normalized_query = normalize_identifier(query).replace("_", "")
        matches = []
        for industry in graph.industries:
            normalized_industry = normalize_identifier(industry).replace("_", "")
            if normalized_industry and normalized_industry in normalized_query:
                matches.append((len(normalized_industry), industry))
        if not matches:
            return graph.industries[0] if len(graph.industries) == 1 else None
        matches.sort(reverse=True)
        return matches[0][1]


class _FieldScorerAdapter:
    """Minimal adapter: wraps a FieldScorer to satisfy the FieldGrounder interface.

    Used for backward compatibility when tests pass raw FieldScorer objects
    to SchemaGeneralSumPlanner.
    """

    def __init__(self, scorer) -> None:
        if not hasattr(scorer, "score_fields"):
            raise TypeError(f"Expected a FieldScorer, got {type(scorer)}")
        self._scorer = scorer

    def ground(self, query, candidate_fields):
        from graph_numeric.learning.field_grounder import FieldGroundingResult

        scores = self._scorer.score_fields(query, candidate_fields)
        best_field, best_score = max(scores.items(), key=lambda item: item[1])
        trace = getattr(self._scorer, "last_trace", None)
        return FieldGroundingResult(
            field_name=best_field,
            confidence=float(best_score),
            scores={k: float(v) for k, v in scores.items()},
            trace=trace if isinstance(trace, dict) else None,
        )

    def precompute_texts(self, texts):
        precompute = getattr(self._scorer, "precompute_texts", None)
        if precompute is not None:
            precompute(texts)
