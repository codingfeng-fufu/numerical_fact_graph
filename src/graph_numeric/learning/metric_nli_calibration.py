from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from graph_numeric.core.attribute_graph import AttributeValueToken
from graph_numeric.learning.metric_matcher import MetricMatcher


@dataclass(frozen=True)
class MetricPairExample:
    token_field: str
    token_label: str
    expected_field: str
    same_metric: bool


@dataclass(frozen=True)
class MetricNliCalibrationResult:
    accept_threshold: float
    best_metrics: dict[str, Any]
    metrics_by_threshold: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accept_threshold": self.accept_threshold,
            "best_metrics": dict(self.best_metrics),
            "metrics_by_threshold": [
                dict(row) for row in self.metrics_by_threshold
            ],
        }


def calibrate_metric_nli_thresholds(
    examples: Iterable[MetricPairExample],
    matcher: MetricMatcher,
    *,
    thresholds: Sequence[float] | None = None,
) -> MetricNliCalibrationResult:
    scored = [
        {
            "score": _score_example(example, matcher),
            "same_metric": bool(example.same_metric),
        }
        for example in examples
    ]
    if not scored:
        raise ValueError("At least one calibration example is required")

    rows = tuple(
        _threshold_metrics(scored, threshold)
        for threshold in (thresholds or _default_thresholds())
    )
    best = max(
        rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["threshold"]),
        ),
    )
    return MetricNliCalibrationResult(
        accept_threshold=float(best["threshold"]),
        best_metrics=dict(best),
        metrics_by_threshold=rows,
    )


def load_metric_pair_examples(path: str | Path) -> tuple[MetricPairExample, ...]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return tuple(
                _example_from_mapping(row)
                for row in csv.DictReader(handle)
            )
    with file_path.open("r", encoding="utf-8") as handle:
        return tuple(
            _example_from_mapping(json.loads(line))
            for line in handle
            if line.strip()
        )


def _score_example(example: MetricPairExample, matcher: MetricMatcher) -> float:
    token = AttributeValueToken(
        token_id=example.token_field,
        entity_id="calibration",
        company_name="calibration",
        field_name=example.token_field,
        field_label=example.token_label,
        value=0.0,
    )
    trace = matcher.match(token, example.expected_field)
    nli_trace = trace.get("nli") if isinstance(trace.get("nli"), dict) else {}
    if "confidence" in nli_trace:
        return float(nli_trace["confidence"])
    return float(trace.get("score", 0.0))


def _threshold_metrics(
    scored: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    for row in scored:
        predicted_same = float(row["score"]) >= threshold
        actual_same = bool(row["same_metric"])
        if predicted_same and actual_same:
            true_positive += 1
        elif predicted_same and not actual_same:
            false_positive += 1
        elif not predicted_same and actual_same:
            false_negative += 1
        else:
            true_negative += 1
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _example_from_mapping(row: dict[str, Any]) -> MetricPairExample:
    return MetricPairExample(
        token_field=str(row["token_field"]),
        token_label=str(row.get("token_label") or row["token_field"]),
        expected_field=str(row["expected_field"]),
        same_metric=_parse_bool(row["same_metric"]),
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "same", "y"}


def _default_thresholds() -> tuple[float, ...]:
    return tuple(round(value / 100, 2) for value in range(50, 96, 5))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
