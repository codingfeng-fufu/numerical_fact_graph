from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Protocol

from graph_numeric.core.attribute_graph import AttributeValueToken
from graph_numeric.learning.metric_matcher import (
    DEFAULT_TRANSFORMERS_NLI_MODEL,
    NliMetricJudgment,
    TransformersNliMetricProvider,
)


class TokenEntailmentProvider(Protocol):
    def judge_entailment(
        self,
        *,
        premise: str,
        hypothesis: str,
        token: AttributeValueToken,
    ) -> NliMetricJudgment:
        ...


@dataclass(frozen=True)
class TokenFaithfulnessChecker:
    provider: TokenEntailmentProvider
    entailment_threshold: float = 0.5
    score_penalty: float = 0.2

    def check_token(self, token: AttributeValueToken) -> dict[str, Any]:
        premise = token.source.text_excerpt if token.source is not None else None
        hypothesis = render_token_assertion(token)
        if not premise:
            return {
                "token_id": token.token_id,
                "status": "missing_evidence",
                "entailment_score": 0.0,
                "threshold": self.entailment_threshold,
                "score_penalty": self.score_penalty,
                "premise": None,
                "hypothesis": hypothesis,
                "warning": "missing source evidence text",
            }
        try:
            judgment = self.provider.judge_entailment(
                premise=premise,
                hypothesis=hypothesis,
                token=token,
            )
        except Exception as exc:
            return {
                "token_id": token.token_id,
                "status": "unavailable",
                "entailment_score": 0.0,
                "threshold": self.entailment_threshold,
                "score_penalty": 0.0,
                "premise": premise,
                "hypothesis": hypothesis,
                "warning": f"faithfulness NLI unavailable: {exc}",
            }
        score = _entailment_score(judgment)
        status = "supported" if score >= self.entailment_threshold else "low_faithfulness"
        row: dict[str, Any] = {
            "token_id": token.token_id,
            "status": status,
            "entailment_score": score,
            "threshold": self.entailment_threshold,
            "score_penalty": self.score_penalty if status == "low_faithfulness" else 0.0,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": judgment.label,
            "confidence": _bounded_score(judgment.confidence),
            "model": judgment.model_name,
        }
        if judgment.metadata:
            row["metadata"] = dict(judgment.metadata)
        if status == "low_faithfulness":
            row["warning"] = "low_faithfulness"
        return row


@dataclass(frozen=True)
class MetricNliEntailmentProvider:
    metric_provider: Any

    def judge_entailment(
        self,
        *,
        premise: str,
        hypothesis: str,
        token: AttributeValueToken,
    ) -> NliMetricJudgment:
        return self.metric_provider.judge_metric_equivalence(
            premise=premise,
            hypothesis=hypothesis,
            token=token,
            expected_field=token.field_name,
        )


def build_token_faithfulness_checker_from_env(
    *,
    nli_pipeline_factory: Any | None = None,
) -> TokenFaithfulnessChecker | None:
    if not _env_bool("TOKEN_FAITHFULNESS_ENABLE_NLI", default=False):
        return None
    provider = TransformersNliMetricProvider(
        model_name=os.getenv("TOKEN_FAITHFULNESS_NLI_MODEL") or DEFAULT_TRANSFORMERS_NLI_MODEL,
        device=_env_device(os.getenv("TOKEN_FAITHFULNESS_NLI_DEVICE")),
        pipeline_factory=nli_pipeline_factory,
    )
    return TokenFaithfulnessChecker(
        provider=MetricNliEntailmentProvider(provider),
        entailment_threshold=_env_float("TOKEN_FAITHFULNESS_THRESHOLD", 0.5),
        score_penalty=_env_float("TOKEN_FAITHFULNESS_SCORE_PENALTY", 0.2),
    )


def evaluate_token_faithfulness(
    tokens: tuple[AttributeValueToken, ...],
    checker: TokenFaithfulnessChecker | None,
) -> dict[str, Any]:
    if checker is None:
        return {
            "enabled": False,
            "summary": {
                "checked": 0,
                "low_faithfulness": 0,
                "missing_evidence": 0,
            },
            "tokens": [],
        }
    rows = [checker.check_token(token) for token in tokens]
    return {
        "enabled": True,
        "summary": {
            "checked": len(rows),
            "low_faithfulness": sum(row["status"] == "low_faithfulness" for row in rows),
            "missing_evidence": sum(row["status"] == "missing_evidence" for row in rows),
        },
        "tokens": rows,
    }


def render_token_assertion(token: AttributeValueToken) -> str:
    metric = token.field_label or token.raw_label or token.field_name.replace("_", " ")
    subject = token.company_name or token.entity_id
    time_phrase = f" in {token.year}" if token.year is not None else ""
    unit_phrase = f" {token.unit}" if token.unit else ""
    dimension_phrase = _dimension_phrase(token.dimensions)
    return f"{metric} for {subject}{time_phrase}{dimension_phrase} was {token.value}{unit_phrase}."


def _dimension_phrase(dimensions: Mapping[str, object] | None) -> str:
    if not dimensions:
        return ""
    parts = [
        f"{key} {value}"
        for key, value in sorted(dimensions.items())
        if value is not None
    ]
    if not parts:
        return ""
    return " with " + ", ".join(parts)


def _entailment_score(judgment: NliMetricJudgment) -> float:
    label_scores = (judgment.metadata or {}).get("label_scores")
    if isinstance(label_scores, Mapping) and "entailment" in label_scores:
        return _bounded_score(float(label_scores["entailment"]))
    if str(judgment.label).strip().lower() in {"entailment", "same", "supported"}:
        return _bounded_score(judgment.confidence)
    return 0.0


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _env_bool(key: str, *, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_device(value: str | None) -> int | str | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value
