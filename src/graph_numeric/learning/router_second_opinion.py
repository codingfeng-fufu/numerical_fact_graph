from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graph_numeric.operators.operator_registry import OPERATOR_REGISTRY


@dataclass(frozen=True)
class RouterSecondOpinionResult:
    operator: str
    confidence: float
    model_name: str | None = None


class RouterSecondOpinion(Protocol):
    def predict_operator(self, query: str) -> RouterSecondOpinionResult:
        ...


@dataclass(frozen=True)
class ClassifierRouterSecondOpinion:
    classifier: Any
    vectorizer: Any | None = None
    model_name: str | None = None

    @classmethod
    def from_joblib(
        cls,
        classifier_path: str | Path,
        *,
        vectorizer_path: str | Path | None = None,
        model_name: str | None = None,
    ) -> "ClassifierRouterSecondOpinion":
        import joblib

        classifier = joblib.load(classifier_path)
        vectorizer = joblib.load(vectorizer_path) if vectorizer_path is not None else None
        return cls(
            classifier=classifier,
            vectorizer=vectorizer,
            model_name=model_name or str(classifier_path),
        )

    def predict_operator(self, query: str) -> RouterSecondOpinionResult:
        rows = [query]
        features = self.vectorizer.transform(rows) if self.vectorizer is not None else rows
        if hasattr(self.classifier, "predict_proba") and hasattr(self.classifier, "classes_"):
            probabilities = self.classifier.predict_proba(features)[0]
            best_index, best_score = max(
                enumerate(probabilities),
                key=lambda item: float(item[1]),
            )
            return RouterSecondOpinionResult(
                operator=str(self.classifier.classes_[best_index]),
                confidence=float(best_score),
                model_name=self.model_name,
            )
        operator = self.classifier.predict(features)[0]
        return RouterSecondOpinionResult(
            operator=str(operator),
            confidence=1.0,
            model_name=self.model_name,
        )


def router_second_opinion_abstain_trace(
    *,
    query: str,
    primary_operator: str,
    second_opinion: RouterSecondOpinion | None,
    min_confidence: float = 0.0,
) -> dict[str, object]:
    if second_opinion is None:
        return {"triggered": False, "reason": None, "detail": None}
    result = second_opinion.predict_operator(query)
    if float(result.confidence) < min_confidence:
        return {"triggered": False, "reason": None, "detail": None}
    primary = OPERATOR_REGISTRY.canonical_executor_operator(primary_operator)
    secondary = OPERATOR_REGISTRY.canonical_executor_operator(result.operator)
    if primary == secondary:
        return {"triggered": False, "reason": None, "detail": None}
    return {
        "triggered": True,
        "reason": "router_disagreement",
        "detail": {
            "primary_operator": primary_operator,
            "second_opinion_operator": result.operator,
            "second_opinion_confidence": float(result.confidence),
            "model": result.model_name,
        },
    }
