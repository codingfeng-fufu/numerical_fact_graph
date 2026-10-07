from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from graph_numeric.sum_general.field_scorer import (
    EmbeddingFieldScorer,
    FieldScorer,
    LearnedFieldScorer,
    LexicalFieldScorer,
    MoEFieldScorer,
    PrototypeLearnableFieldScorer,
    TypeCompatibilityFieldScorer,
    field_matching_text,
)


@dataclass(frozen=True)
class FieldGroundingResult:
    field_name: str
    confidence: float
    scores: dict[str, float]
    trace: dict[str, Any] | None = None


class FieldGrounder:
    """Independent field grounding module, extracted from SchemaGeneralSumPlanner.

    Takes a natural language query and a set of candidate field names from the
    attribute-value graph, and determines which field the query targets.

    Multiple grounding strategies are available:
    - lexical:     alias dictionary + char n-gram (best for registered aliases)
    - embedding:   sentence-transformer semantic similarity (best for open-schema)
    - type:        value-type compatibility (count vs money vs rate)
    - learned:     logistic regression from training data
    - moe:         mixture of the above with heuristic gate

    The moe strategy is recommended for production; lexical alone is a
    strong zero-dependency baseline.
    """

    STRATEGIES = ("lexical", "embedding", "type", "learned", "moe", "learnable_ranker")

    def __init__(
        self,
        strategy: str = "lexical",
        *,
        embedding_model: str = "BAAI/bge-m3",
        embedding_encoder: Any = None,
        embedding_device: str | None = None,
        seen_fields: Sequence[str] | None = None,
        query_terms_by_field: dict[str, Sequence[str]] | None = None,
        query_templates: Sequence[str] | None = None,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown field grounding strategy: {strategy}. Options: {self.STRATEGIES}")

        self.strategy = strategy
        self._scorer = self._build_scorer(
            strategy,
            embedding_model=embedding_model,
            embedding_encoder=embedding_encoder,
            embedding_device=embedding_device,
            seen_fields=seen_fields,
            query_terms_by_field=query_terms_by_field,
            query_templates=query_templates,
        )

    def _build_scorer(
        self,
        strategy: str,
        **kwargs,
    ) -> FieldScorer:
        if strategy == "lexical":
            return LexicalFieldScorer()
        if strategy == "embedding":
            return EmbeddingFieldScorer(
                model_name=kwargs.get("embedding_model", "BAAI/bge-m3"),
                encoder=kwargs.get("embedding_encoder"),
                device=kwargs.get("embedding_device"),
            )
        if strategy == "type":
            return TypeCompatibilityFieldScorer()
        if strategy == "learned":
            seen_fields = kwargs.get("seen_fields")
            query_terms = kwargs.get("query_terms_by_field")
            templates = kwargs.get("query_templates")
            if not seen_fields or not query_terms or not templates:
                raise ValueError("learned strategy requires seen_fields, query_terms_by_field, and query_templates")
            return LearnedFieldScorer.fit(
                seen_fields=seen_fields,
                query_terms_by_field=query_terms,
                query_templates=templates,
            )
        if strategy == "learnable_ranker":
            return PrototypeLearnableFieldScorer()
        # moe: combine lexical + embedding + type
        experts: dict[str, FieldScorer] = {
            "lexical": LexicalFieldScorer(),
        }
        embedding_encoder = kwargs.get("embedding_encoder")
        experts["embedding"] = EmbeddingFieldScorer(
            model_name=kwargs.get("embedding_model", "BAAI/bge-m3"),
            encoder=embedding_encoder,
            device=kwargs.get("embedding_device"),
        )
        experts["type"] = TypeCompatibilityFieldScorer()
        learned_seen = kwargs.get("seen_fields")
        learned_terms = kwargs.get("query_terms_by_field")
        learned_templates = kwargs.get("query_templates")
        if learned_seen and learned_terms and learned_templates:
            experts["learned"] = LearnedFieldScorer.fit(
                seen_fields=learned_seen,
                query_terms_by_field=learned_terms,
                query_templates=learned_templates,
            )
        return MoEFieldScorer(experts)

    def ground(self, query: str, candidate_fields: Sequence[str]) -> FieldGroundingResult:
        if not candidate_fields:
            raise ValueError("At least one candidate field is required.")
        scores = self._scorer.score_fields(query, candidate_fields)
        best_field, best_score = max(scores.items(), key=lambda item: item[1])
        trace = getattr(self._scorer, "last_trace", None)
        return FieldGroundingResult(
            field_name=best_field,
            confidence=float(best_score),
            scores={k: float(v) for k, v in scores.items()},
            trace=trace if isinstance(trace, dict) else None,
        )

    def precompute_texts(self, texts: Sequence[str]) -> None:
        precompute = getattr(self._scorer, "precompute_texts", None)
        if precompute is not None:
            precompute(texts)

    @classmethod
    def from_default_moe(
        cls,
        *,
        embedding_model: str = "BAAI/bge-m3",
        embedding_encoder: Any = None,
        embedding_device: str | None = None,
        seen_fields: Sequence[str] | None = None,
        query_terms_by_field: dict[str, Sequence[str]] | None = None,
        query_templates: Sequence[str] | None = None,
    ) -> FieldGrounder:
        """Convenience factory for the recommended MoE strategy."""
        return cls(
            strategy="moe",
            embedding_model=embedding_model,
            embedding_encoder=embedding_encoder,
            embedding_device=embedding_device,
            seen_fields=seen_fields,
            query_terms_by_field=query_terms_by_field,
            query_templates=query_templates,
        )

    @classmethod
    def from_lexical(cls) -> FieldGrounder:
        """Zero-dependency lexical grounder for cold start."""
        return cls(strategy="lexical")
