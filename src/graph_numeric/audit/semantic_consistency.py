"""Deterministic statistics for semantic-answer and abstention audits."""

from __future__ import annotations

import bisect
import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ABS_TOL = 1e-4
REL_TOL = 1e-4
_CURRENCY_CLASS = r"[$\u20ac\u00a3\u00a5\uffe5\u20b9]"
_CURRENCY_RE = re.compile(_CURRENCY_CLASS)
_NUMBER_PATTERN = r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"


@dataclass(frozen=True)
class SemanticConsistencyResult:
    normalized_answers: tuple[str, ...]
    cluster_sizes: tuple[int, ...]
    score: float
    entropy: float


@dataclass(frozen=True)
class IsotonicModel:
    upper_bounds: tuple[float, ...]
    values: tuple[float, ...]

    def predict(self, score: float) -> float:
        """Return the fitted rate for ``score`` using its enclosing PAV block."""
        if not self.upper_bounds:
            raise ValueError("isotonic model is empty")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("score must be finite")
        index = bisect.bisect_left(self.upper_bounds, numeric_score)
        if index == len(self.values):
            index -= 1
        return self.values[index]

    def predict_many(self, scores: Sequence[float]) -> tuple[float, ...]:
        return tuple(self.predict(score) for score in scores)


@dataclass(frozen=True)
class PairedRiskBootstrapResult:
    point_estimate: float
    lower_95: float
    upper_95: float


def normalize_answer(answer: Any) -> str:
    """Normalize one answer without importing the benchmark CLI module."""
    text = _normalize_text_answer(str(answer))
    values = _answer_numbers(text)
    if len(values) == 1 and _is_numeric_only_answer(text):
        return _format_number(values[0])
    return text


def answers_equivalent(left: Any, right: Any) -> bool:
    """Compare normalized text or numeric answers with evaluator tolerances."""
    left_normalized = normalize_answer(left)
    right_normalized = normalize_answer(right)
    if left_normalized == right_normalized:
        return True
    left_values = _answer_numbers(left_normalized)
    right_values = _answer_numbers(right_normalized)
    return bool(left_values) and len(left_values) == len(right_values) and all(
        _close(predicted, expected)
        for predicted, expected in zip(left_values, right_values, strict=True)
    )


def semantic_consistency(
    answers: Sequence[Any], *, question: str
) -> SemanticConsistencyResult:
    """Cluster equivalent answers and summarize the largest semantic consensus."""
    if not answers:
        raise ValueError("answers must be nonempty")
    if not isinstance(question, str):
        raise ValueError("question must be a string")

    normalized_answers = tuple(normalize_answer(answer) for answer in answers)
    clusters: list[list[Any]] = []
    for answer in sorted(answers, key=_canonical_answer_key):
        for cluster in clusters:
            if all(answers_equivalent(answer, member) for member in cluster):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])

    total = len(answers)
    cluster_sizes = [len(cluster) for cluster in clusters]
    score = max(cluster_sizes) / total
    entropy = -sum(
        (size / total) * math.log(size / total) for size in cluster_sizes if size
    )
    return SemanticConsistencyResult(
        normalized_answers=normalized_answers,
        cluster_sizes=tuple(sorted(cluster_sizes, reverse=True)),
        score=score,
        entropy=entropy,
    )


def fit_isotonic(scores: Sequence[float], labels: Sequence[int | bool]) -> IsotonicModel:
    """Fit a deterministic nondecreasing PAV calibration model."""
    if not scores or not labels:
        raise ValueError("scores and labels must be nonempty")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal lengths")

    grouped: dict[float, list[int]] = {}
    for score, label in zip(scores, labels, strict=True):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("scores must be finite")
        if label not in (0, 1, False, True):
            raise ValueError("labels must be binary")
        positives, count = grouped.setdefault(numeric_score, [0, 0])
        grouped[numeric_score] = [positives + int(bool(label)), count + 1]

    blocks: list[list[float]] = []
    for score in sorted(grouped):
        positives, count = grouped[score]
        blocks.append([score, float(positives), float(count)])
        while len(blocks) > 1 and _block_mean(blocks[-2]) > _block_mean(blocks[-1]):
            previous = blocks.pop()
            blocks[-1] = [
                previous[0],
                blocks[-1][1] + previous[1],
                blocks[-1][2] + previous[2],
            ]

    return IsotonicModel(
        upper_bounds=tuple(block[0] for block in blocks),
        values=tuple(_block_mean(block) for block in blocks),
    )


def sha256_order(sample_id: str) -> str:
    """Return the stable SHA-256 tie-break key for a sample ID."""
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def exact_coverage_ids(rows: Sequence[Mapping[str, Any]], count: int) -> tuple[str, ...]:
    """Select an exact deterministic coverage subset, highest score first."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("count must be an integer")
    if not 0 <= count <= len(rows):
        raise ValueError("count must be between zero and the number of rows")

    ranked: list[tuple[str, float]] = []
    sample_ids: set[str] = set()
    for row in rows:
        if "sample_id" not in row or "score" not in row:
            raise ValueError("each row requires sample_id and score")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if sample_id in sample_ids:
            raise ValueError("sample IDs must be unique")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError("scores must be finite")
        sample_ids.add(sample_id)
        ranked.append((sample_id, score))

    ranked.sort(key=lambda item: (-item[1], sha256_order(item[0])))
    return tuple(sample_id for sample_id, _ in ranked[:count])


def paired_risk_bootstrap(
    left: Sequence[bool | int],
    right: Sequence[bool | int],
    *,
    draws: int = 10000,
    seed: int = 7,
    left_selected: Sequence[bool | int] | None = None,
    right_selected: Sequence[bool | int] | None = None,
) -> PairedRiskBootstrapResult:
    """Bootstrap paired selective-risk differences on a common item universe.

    ``True``/``1`` denotes an error event; selection masks default to accepting
    every item.  Undefined resamples with no accepted item for either policy
    are discarded and deterministically redrawn, so every percentile input is
    a defined selective-risk comparison.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    left_events = _binary_values(left, name="left")
    right_events = _binary_values(right, name="right")
    item_count = len(left_events)
    left_mask = _binary_values(
        left_selected if left_selected is not None else (True,) * item_count,
        name="left_selected",
    )
    right_mask = _binary_values(
        right_selected if right_selected is not None else (True,) * item_count,
        name="right_selected",
    )
    if not item_count:
        raise ValueError("paired inputs must be nonempty")
    if not all(
        len(values) == item_count
        for values in (right_events, left_mask, right_mask)
    ):
        raise ValueError("events and selection masks must have equal lengths")
    if not any(left_mask) or not any(right_mask):
        raise ValueError("each policy must select at least one item")

    all_indices = tuple(range(item_count))
    point_estimate = (
        _selective_risk(left_events, left_mask, all_indices)
        - _selective_risk(right_events, right_mask, all_indices)
    )
    generator = random.Random(seed)
    sampled: list[float] = []
    while len(sampled) < draws:
        indices = tuple(generator.randrange(item_count) for _ in all_indices)
        left_risk = _selective_risk(left_events, left_mask, indices)
        right_risk = _selective_risk(right_events, right_mask, indices)
        if left_risk is None or right_risk is None:
            continue
        sampled.append(left_risk - right_risk)
    sampled.sort()
    return PairedRiskBootstrapResult(
        point_estimate=point_estimate,
        lower_95=_empirical_percentile(sampled, 0.025),
        upper_95=_empirical_percentile(sampled, 0.975),
    )


def calibration_metrics(
    probabilities: Sequence[float], labels: Sequence[int | bool]
) -> dict[str, float | None]:
    """Return Brier, 10-bin ECE, and correctness/error-detection AUROCs."""
    if not probabilities or not labels:
        raise ValueError("probabilities and labels must be nonempty")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal lengths")

    numeric_probabilities = tuple(float(probability) for probability in probabilities)
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in numeric_probabilities
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if any(label not in (0, 1, False, True) for label in labels):
        raise ValueError("labels must be binary")
    binary_labels = tuple(int(bool(label)) for label in labels)

    return {
        "brier": sum(
            (probability - label) ** 2
            for probability, label in zip(numeric_probabilities, binary_labels, strict=True)
        )
        / len(binary_labels),
        "ece": _ten_bin_ece(numeric_probabilities, binary_labels),
        "correctness_auroc": _binary_auroc(numeric_probabilities, binary_labels),
        "error_detection_auroc": _binary_auroc(
            tuple(1.0 - probability for probability in numeric_probabilities),
            tuple(1 - label for label in binary_labels),
        ),
    }


def _number_tokens(text: str) -> list[tuple[float, bool]]:
    """Mirror the evaluator's parenthesized-negative and currency parsing."""
    positioned_values: list[tuple[int, float, bool]] = []
    parenthesized_spans: list[tuple[int, int]] = []
    for match in re.finditer(
        rf"\(\s*{_CURRENCY_CLASS}?\s*(?P<value>{_NUMBER_PATTERN})\s*(?P<percent>%?)\s*\)", text
    ):
        previous = text[: match.start()].rstrip()
        if previous and (previous[-1].isdigit() or previous[-1] == "%"):
            parenthesized_spans.append(match.span())
            continue
        try:
            value = float(match.group("value").replace(",", ""))
        except ValueError:
            parenthesized_spans.append(match.span())
            continue
        positioned_values.append((match.start(), -abs(value), bool(match.group("percent"))))
        parenthesized_spans.append(match.span())
    for match in re.finditer(_NUMBER_PATTERN, text):
        if any(start <= match.start() < end for start, end in parenthesized_spans):
            continue
        try:
            positioned_values.append(
                (
                    match.start(),
                    float(match.group(0).replace(",", "")),
                    bool(re.match(r"\s*%", text[match.end() :])),
                )
            )
        except ValueError:
            continue
    positioned_values.sort(key=lambda item: item[0])
    values: list[tuple[float, bool]] = []
    for _, value, is_percent in positioned_values:
        if (
            values
            and values[-1][1] == is_percent
            and _close(values[-1][0], value, abs_tol=1e-12)
        ):
            continue
        values.append((value, is_percent))
    return values


def _numbers(text: str) -> list[float]:
    return [value for value, _ in _number_tokens(text)]


def _normalize_text_answer(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = _CURRENCY_RE.sub("", normalized)
    normalized = normalized.strip(" .;:,")
    normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
    normalized = re.sub(r"^(?:by|via|with)\s+", "", normalized)
    return re.sub(
        r"^(?:primarily\s+due\s+to|because\s+of|because|due\s+to|driven\s+by)\s+",
        "",
        normalized,
    )


def _answer_numbers(text: str) -> list[float]:
    return [
        value / 100.0 if is_percent else value
        for value, is_percent in _number_tokens(text)
    ]


def _is_numeric_only_answer(text: str) -> bool:
    return bool(
        re.fullmatch(
            rf"\(?\s*[-+]?\s*(?:{_NUMBER_PATTERN})\s*%?\s*\)?", text
        )
    )


def _format_number(value: float) -> str:
    return format(value, ".15g")


def _close(predicted: float, expected: float, *, abs_tol: float = ABS_TOL) -> bool:
    if abs(predicted - expected) <= abs_tol:
        return True
    return abs(predicted - expected) <= REL_TOL * max(abs(expected), 1.0)


def _block_mean(block: Sequence[float]) -> float:
    return block[1] / block[2]


def _canonical_answer_key(answer: Any) -> tuple[str, str, str]:
    return (normalize_answer(answer), type(answer).__name__, repr(answer))


def _binary_values(values: Sequence[bool | int], *, name: str) -> tuple[bool, ...]:
    result: list[bool] = []
    for value in values:
        if isinstance(value, bool):
            result.append(value)
        elif type(value) is int and value in (0, 1):
            result.append(bool(value))
        else:
            raise ValueError(f"{name} must contain only bool or integer binary values")
    return tuple(result)


def _selective_risk(
    errors: Sequence[bool], selected: Sequence[bool], indices: Sequence[int]
) -> float | None:
    selected_errors = [errors[index] for index in indices if selected[index]]
    if not selected_errors:
        return None
    return sum(selected_errors) / len(selected_errors)


def _empirical_percentile(values: Sequence[float], quantile: float) -> float:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def _ten_bin_ece(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for probability, label in zip(probabilities, labels, strict=True):
        bins[min(int(probability * 10), 9)].append((probability, label))
    return sum(
        len(bucket) / len(probabilities)
        * abs(
            sum(probability for probability, _ in bucket) / len(bucket)
            - sum(label for _, label in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )


def _binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(labels[index] for index, _ in ranked[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
