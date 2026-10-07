from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol
from urllib import error, request

from graph_numeric.core.attribute_graph import (
    AttributeValueToken,
    field_aliases,
    normalize_identifier,
)


DEFAULT_TRANSFORMERS_NLI_MODEL = "typeform/distilbert-base-uncased-mnli"


@dataclass(frozen=True)
class NliMetricJudgment:
    label: str
    confidence: float
    rationale: str | None = None
    model_name: str | None = None
    metadata: Mapping[str, Any] | None = None


class MetricNliProvider(Protocol):
    def judge_metric_equivalence(
        self,
        *,
        premise: str,
        hypothesis: str,
        token: AttributeValueToken,
        expected_field: str,
    ) -> NliMetricJudgment:
        ...


@dataclass(frozen=True)
class LlmMetricJudgment:
    verdict: str
    confidence: float = 1.0
    rationale: str | None = None
    model_name: str | None = None
    metadata: Mapping[str, Any] | None = None


class MetricLlmJudge(Protocol):
    def judge_metric_equivalence(
        self,
        *,
        premise: str,
        hypothesis: str,
        token: AttributeValueToken,
        expected_field: str,
        nli_trace: Mapping[str, Any],
    ) -> LlmMetricJudgment:
        ...


@dataclass(frozen=True)
class AliasReviewRecord:
    token_field: str
    expected_field: str
    shared_surface: str
    rationale: str
    source: str
    metadata: Mapping[str, Any] | None = None


class AliasReviewSink(Protocol):
    def record(self, record: AliasReviewRecord) -> dict[str, Any]:
        ...


@dataclass
class InMemoryAliasReviewSink:
    records: list[AliasReviewRecord] = field(default_factory=list)

    def record(self, record: AliasReviewRecord) -> dict[str, Any]:
        self.records.append(record)
        return {
            "status": "queued",
            "index": len(self.records) - 1,
        }


@dataclass(frozen=True)
class MetricMatcherRuntimeConfig:
    enable_nli: bool = False
    nli_model: str = DEFAULT_TRANSFORMERS_NLI_MODEL
    nli_device: int | str | None = None
    nli_accept_threshold: float = 0.8
    nli_reject_threshold: float = 0.3
    enable_llm_judge: bool = False
    llm_model: str | None = None
    alias_review_path: Path | None = None

    @classmethod
    def from_env(cls) -> "MetricMatcherRuntimeConfig":
        return cls(
            enable_nli=_env_bool("METRIC_MATCHER_ENABLE_NLI", default=False),
            nli_model=os.getenv("METRIC_MATCHER_NLI_MODEL") or DEFAULT_TRANSFORMERS_NLI_MODEL,
            nli_device=_env_device(os.getenv("METRIC_MATCHER_NLI_DEVICE")),
            nli_accept_threshold=_env_float("METRIC_MATCHER_NLI_ACCEPT_THRESHOLD", 0.8),
            nli_reject_threshold=_env_float("METRIC_MATCHER_NLI_REJECT_THRESHOLD", 0.3),
            enable_llm_judge=_env_bool("METRIC_MATCHER_ENABLE_LLM_JUDGE", default=False),
            llm_model=os.getenv("METRIC_MATCHER_LLM_MODEL"),
            alias_review_path=(
                Path(os.environ["METRIC_MATCHER_ALIAS_REVIEW_PATH"])
                if os.getenv("METRIC_MATCHER_ALIAS_REVIEW_PATH")
                else None
            ),
        )


@dataclass(frozen=True)
class JsonlAliasReviewSink:
    path: Path

    def record(self, record: AliasReviewRecord) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return {
            "status": "queued",
            "path": str(self.path),
        }


@dataclass
class OpenAICompatibleMetricLlmJudge:
    model_name: str
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 45.0
    chat_completion: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None

    def judge_metric_equivalence(
        self,
        *,
        premise: str,
        hypothesis: str,
        token: AttributeValueToken,
        expected_field: str,
        nli_trace: Mapping[str, Any],
    ) -> LlmMetricJudgment:
        response = self._chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a metric equivalence judge. Return JSON only. "
                        "Do not calculate answers. Decide whether two metric "
                        "descriptions refer to the same financial metric."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "token_metric": {
                                "field_name": token.field_name,
                                "field_label": token.field_label,
                                "premise": premise,
                            },
                            "expected_metric": {
                                "field_name": expected_field,
                                "hypothesis": hypothesis,
                            },
                            "nli_trace": dict(nli_trace),
                            "required_schema": {
                                "verdict": "same | different",
                                "confidence": "number from 0 to 1",
                                "rationale": "one short evidence-based sentence",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        payload = _extract_llm_json_payload(_chat_completion_text(response))
        return LlmMetricJudgment(
            verdict=normalize_identifier(str(payload.get("verdict", "different"))),
            confidence=_bounded_score(float(payload.get("confidence", 1.0))),
            rationale=str(payload.get("rationale") or ""),
            model_name=self.model_name,
        )

    @classmethod
    def from_llm_graph_env(cls) -> "OpenAICompatibleMetricLlmJudge":
        from graph_numeric.extraction.llm_extraction import LLMGraphExtractorConfig

        config = LLMGraphExtractorConfig.from_env()
        return cls(
            model_name=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.chat_completion is not None:
            return self.chat_completion(messages)
        if not self.base_url:
            raise RuntimeError("LLM judge base_url is not configured")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM metric judge failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"LLM metric judge failed: {exc.reason}") from exc


def build_metric_matcher_from_env(
    *,
    nli_pipeline_factory: Callable[..., Any] | None = None,
    llm_chat_completion: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
) -> "MetricMatcher":
    config = MetricMatcherRuntimeConfig.from_env()
    nli_provider = (
        TransformersNliMetricProvider(
            model_name=config.nli_model,
            device=config.nli_device,
            pipeline_factory=nli_pipeline_factory,
        )
        if config.enable_nli
        else None
    )
    llm_judge: OpenAICompatibleMetricLlmJudge | None = None
    if config.enable_llm_judge:
        if llm_chat_completion is not None:
            llm_judge = OpenAICompatibleMetricLlmJudge(
                model_name=config.llm_model or "metric-llm-judge",
                chat_completion=llm_chat_completion,
            )
        else:
            llm_judge = OpenAICompatibleMetricLlmJudge.from_llm_graph_env()
            if config.llm_model:
                llm_judge.model_name = config.llm_model
    alias_review_sink = (
        JsonlAliasReviewSink(config.alias_review_path)
        if config.alias_review_path is not None
        else None
    )
    return MetricMatcher(
        nli_provider=nli_provider,
        nli_accept_threshold=config.nli_accept_threshold,
        nli_reject_threshold=config.nli_reject_threshold,
        llm_judge=llm_judge,
        alias_review_sink=alias_review_sink,
    )


@dataclass
class TransformersNliMetricProvider:
    model_name: str = DEFAULT_TRANSFORMERS_NLI_MODEL
    device: int | str | None = None
    pipeline_factory: Callable[..., Any] | None = None
    label_map: Mapping[str, str] | None = None
    _pipeline: Any = field(default=None, init=False, repr=False)

    def judge_metric_equivalence(
        self,
        *,
        premise: str,
        hypothesis: str,
        token: AttributeValueToken,
        expected_field: str,
    ) -> NliMetricJudgment:
        classifier = self._load_pipeline()
        raw = classifier(
            {"text": premise, "text_pair": hypothesis},
            truncation=True,
        )
        label_scores = _pipeline_label_scores(raw, self.label_map)
        if not label_scores:
            raise RuntimeError("NLI pipeline returned no label scores")
        best_label, best_score = max(
            label_scores.items(),
            key=lambda item: item[1],
        )
        return NliMetricJudgment(
            label=best_label,
            confidence=best_score,
            rationale=f"transformers NLI predicted {best_label} with score {best_score}",
            model_name=self.model_name,
            metadata={"label_scores": label_scores},
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        factory = self.pipeline_factory
        if factory is None:
            try:
                from transformers import pipeline as factory
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required for TransformersNliMetricProvider; "
                    "install the embedding extra first"
                ) from exc
        kwargs: dict[str, Any] = {
            "task": "text-classification",
            "model": self.model_name,
            "top_k": None,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipeline = factory(**kwargs)
        return self._pipeline


@dataclass(frozen=True)
class MetricMatcher:
    nli_provider: MetricNliProvider | None = None
    nli_accept_threshold: float = 0.8
    nli_reject_threshold: float = 0.3
    llm_judge: MetricLlmJudge | None = None
    alias_review_sink: AliasReviewSink | None = None

    def match(self, token: AttributeValueToken, expected_field: str) -> dict[str, Any]:
        if token.field_name == expected_field:
            return {
                "level": "rule_exact",
                "score": 1.0,
                "rationale": f"field_name exact match: {expected_field}",
            }
        alias_trace = self._alias_match_trace(token, expected_field)
        if alias_trace is not None:
            return alias_trace
        if self.nli_provider is None:
            return self._no_match_trace(token, expected_field)
        return self._nli_match_trace(token, expected_field)

    def _alias_match_trace(
        self,
        token: AttributeValueToken,
        expected_field: str,
    ) -> dict[str, Any] | None:
        expected_aliases = {
            normalize_identifier(alias)
            for alias in field_aliases(expected_field)
        }
        for token_alias in field_aliases(token.field_name):
            if normalize_identifier(token_alias) in expected_aliases:
                return {
                    "level": "alias_table",
                    "score": 1.0,
                    "rationale": f"shared alias: {token_alias.replace('_', ' ')}",
                }
        return None

    def _nli_match_trace(
        self,
        token: AttributeValueToken,
        expected_field: str,
    ) -> dict[str, Any]:
        premise = _metric_premise(token)
        hypothesis = _metric_hypothesis(expected_field)
        try:
            judgment = self.nli_provider.judge_metric_equivalence(
                premise=premise,
                hypothesis=hypothesis,
                token=token,
                expected_field=expected_field,
            )
        except Exception as exc:
            return {
                **self._no_match_trace(token, expected_field),
                "rationale": f"nli unavailable: {exc}",
                "nli": {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "status": "error",
                    "error": str(exc),
                },
            }

        score = _nli_entailment_score(judgment)
        nli_trace: dict[str, Any] = {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": judgment.label,
            "confidence": score,
            "model": judgment.model_name,
        }
        if judgment.metadata:
            nli_trace["metadata"] = dict(judgment.metadata)
        if score >= self.nli_accept_threshold:
            nli_trace["decision"] = "accept"
            return {
                "level": "nli_entailment",
                "score": score,
                "rationale": judgment.rationale or f"nli entailment score: {score}",
                "nli": nli_trace,
            }
        if score <= self.nli_reject_threshold:
            nli_trace["decision"] = "reject"
            return {
                "level": "nli_reject",
                "score": 0.0,
                "rationale": (
                    judgment.rationale
                    or f"nli rejected: entailment score {score}"
                ),
                "nli": nli_trace,
            }
        nli_trace["decision"] = "uncertain"
        if self.llm_judge is not None:
            return self._llm_match_trace(
                token,
                expected_field,
                premise=premise,
                hypothesis=hypothesis,
                nli_trace=nli_trace,
            )
        return {
            **self._no_match_trace(token, expected_field),
            "rationale": (
                judgment.rationale
                or f"nli uncertain: entailment score {score}"
            ),
            "nli": nli_trace,
        }

    def _llm_match_trace(
        self,
        token: AttributeValueToken,
        expected_field: str,
        *,
        premise: str,
        hypothesis: str,
        nli_trace: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            judgment = self.llm_judge.judge_metric_equivalence(
                premise=premise,
                hypothesis=hypothesis,
                token=token,
                expected_field=expected_field,
                nli_trace=nli_trace,
            )
        except Exception as exc:
            return {
                **self._no_match_trace(token, expected_field),
                "rationale": f"llm judge unavailable: {exc}",
                "nli": nli_trace,
                "llm": {
                    "status": "error",
                    "error": str(exc),
                },
            }
        confidence = _bounded_score(judgment.confidence)
        llm_trace: dict[str, Any] = {
            "verdict": normalize_identifier(judgment.verdict),
            "confidence": confidence,
            "model": judgment.model_name,
        }
        if judgment.metadata:
            llm_trace["metadata"] = dict(judgment.metadata)
        if _is_same_verdict(judgment.verdict):
            trace = {
                "level": "llm_judge",
                "score": confidence,
                "rationale": judgment.rationale or "llm judged metrics as same",
                "nli": nli_trace,
                "llm": llm_trace,
            }
            alias_review = self._record_alias_review(
                token,
                expected_field,
                rationale=str(trace["rationale"]),
                model_name=judgment.model_name,
            )
            if alias_review is not None:
                trace["alias_review"] = alias_review
            return trace
        return {
            "level": "llm_judge_reject",
            "score": 0.0,
            "rationale": judgment.rationale or "llm judged metrics as different",
            "nli": nli_trace,
            "llm": llm_trace,
        }

    def _record_alias_review(
        self,
        token: AttributeValueToken,
        expected_field: str,
        *,
        rationale: str,
        model_name: str | None,
    ) -> dict[str, Any] | None:
        if self.alias_review_sink is None:
            return None
        return self.alias_review_sink.record(
            AliasReviewRecord(
                token_field=token.field_name,
                expected_field=expected_field,
                shared_surface=token.field_label or token.raw_label or token.field_name.replace("_", " "),
                rationale=rationale,
                source="llm_judge",
                metadata={
                    "token_id": token.token_id,
                    "model": model_name,
                },
            )
        )

    @staticmethod
    def _no_match_trace(
        token: AttributeValueToken,
        expected_field: str,
    ) -> dict[str, Any]:
        return {
            "level": "no_match",
            "score": 0.0,
            "rationale": f"field_name mismatch: expected {expected_field}, got {token.field_name}",
        }


def _metric_premise(token: AttributeValueToken) -> str:
    label = token.field_label or token.raw_label or token.field_name.replace("_", " ")
    return f"Token metric: {label} ({token.field_name})."


def _metric_hypothesis(expected_field: str) -> str:
    label = expected_field.replace("_", " ")
    return f"Expected metric: {label} ({expected_field})."


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _nli_entailment_score(judgment: NliMetricJudgment) -> float:
    label_scores = (judgment.metadata or {}).get("label_scores")
    if isinstance(label_scores, Mapping) and "entailment" in label_scores:
        return _bounded_score(float(label_scores["entailment"]))
    if _is_entailment(judgment.label):
        return _bounded_score(float(judgment.confidence))
    return 0.0


def _pipeline_label_scores(
    raw: Any,
    label_map: Mapping[str, str] | None = None,
) -> dict[str, float]:
    rows = _pipeline_rows(raw)
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = _canonical_nli_label(str(row.get("label") or ""), label_map)
        if not label:
            continue
        scores[label] = _bounded_score(float(row.get("score", 0.0)))
    return scores


def _pipeline_rows(raw: Any) -> list[Any]:
    if isinstance(raw, Mapping):
        return [raw]
    if not isinstance(raw, list):
        return []
    if raw and isinstance(raw[0], list):
        return list(raw[0])
    return list(raw)


def _canonical_nli_label(
    label: str,
    label_map: Mapping[str, str] | None = None,
) -> str:
    explicit_map = {
        normalize_identifier(key): normalize_identifier(value)
        for key, value in (label_map or {}).items()
    }
    normalized = normalize_identifier(label)
    if normalized in explicit_map:
        return explicit_map[normalized]
    default_map = {
        "label_0": "contradiction",
        "label_1": "neutral",
        "label_2": "entailment",
    }
    return default_map.get(normalized, normalized)


def _is_entailment(label: str) -> bool:
    return normalize_identifier(label) in {
        "entailment",
        "same",
        "match",
        "equivalent",
    }


def _is_same_verdict(verdict: str) -> bool:
    return normalize_identifier(verdict) in {"same", "match", "equivalent", "yes"}


def _chat_completion_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM judge response does not contain choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("LLM judge choice is not an object")
    message = first.get("message")
    if isinstance(message, Mapping) and message.get("content") is not None:
        return str(message["content"])
    if first.get("text") is not None:
        return str(first["text"])
    raise ValueError("LLM judge response does not contain text content")


def _extract_llm_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced is not None:
        stripped = fenced.group(1)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("LLM judge JSON payload must be an object")
    return payload


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


DEFAULT_METRIC_MATCHER = MetricMatcher()
