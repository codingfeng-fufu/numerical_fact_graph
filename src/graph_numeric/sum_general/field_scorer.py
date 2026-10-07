from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from graph_numeric.core.attribute_graph import field_aliases, normalize_identifier
from graph_numeric.sum_general.semantic_profile import semantic_compatibility_score


class FieldScorer(Protocol):
    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        """Return one score per candidate field in the current graph."""


class LexicalFieldScorer:
    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        return {field_name: lexical_field_score(query, field_name) for field_name in field_names}


class TypeCompatibilityFieldScorer:
    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        return {
            field_name: semantic_compatibility_score(query, field_name)
            for field_name in field_names
        }


class EmbeddingFieldScorer:
    """Semantic field scorer backed by a sentence embedding model.

    The production path lazily loads BAAI/bge-m3 through sentence-transformers.
    Tests can inject a lightweight encoder with an ``encode(list[str])`` method
    to keep the scorer deterministic and independent of model downloads.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        encoder: Any | None = None,
        batch_size: int = 64,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._encoder = encoder
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self._embedding_cache: dict[str, np.ndarray] = {}

    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        field_texts = {field_name: field_matching_text(field_name) for field_name in field_names}
        self._ensure_embeddings([query, *field_texts.values()])
        query_vector = self._embedding(query)
        field_vectors = {field_name: self._embedding(text) for field_name, text in field_texts.items()}
        return {
            field_name: _cosine_to_unit_score(query_vector, field_vector)
            for field_name, field_vector in field_vectors.items()
        }

    def _ensure_embeddings(self, texts: Sequence[str]) -> None:
        self._encode_missing(texts)

    def precompute_texts(self, texts: Sequence[str]) -> None:
        self._ensure_embeddings(list(dict.fromkeys(texts)))

    def _embedding(self, text: str) -> np.ndarray:
        if text not in self._embedding_cache:
            self._encode_missing([text])
        return self._embedding_cache[text]

    def _encode_missing(self, texts: Sequence[str]) -> None:
        missing = [text for text in texts if text not in self._embedding_cache]
        if not missing:
            return
        encoder = self._get_encoder()
        try:
            encoded = encoder.encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            encoded = encoder.encode(missing)
        vectors = _as_2d_float_array(encoded)
        vectors = _l2_normalize_rows(vectors)
        for text, vector in zip(missing, vectors, strict=True):
            self._embedding_cache[text] = vector

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on local ML runtime.
            raise RuntimeError(
                "EmbeddingFieldScorer requires sentence-transformers and a valid torch installation. "
                "Install project dependencies or pass a test encoder explicitly."
            ) from exc

        device = self.device
        if device is None:
            cuda = getattr(torch, "cuda", None)
            device = "cuda" if cuda is not None and cuda.is_available() else "cpu"
        self._encoder = SentenceTransformer(
            self.model_name,
            device=device,
            trust_remote_code=self.trust_remote_code,
        )
        return self._encoder


@dataclass
class LearnedFieldScorer:
    model: LogisticRegression

    @classmethod
    def fit(
        cls,
        *,
        seen_fields: Sequence[str],
        query_terms_by_field: dict[str, Sequence[str]],
        query_templates: Sequence[str],
    ) -> LearnedFieldScorer:
        x_rows: list[list[float]] = []
        y_rows: list[int] = []
        for target_field in seen_fields:
            for query_term in query_terms_by_field[target_field]:
                for template in query_templates:
                    query = template.format(field_term=query_term)
                    for candidate_field in seen_fields:
                        x_rows.append(pair_features(query, candidate_field))
                        y_rows.append(1 if candidate_field == target_field else 0)

        model = LogisticRegression(class_weight="balanced", random_state=17)
        model.fit(np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64))
        return cls(model=model)

    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        features = np.asarray([pair_features(query, field_name) for field_name in field_names], dtype=np.float32)
        probabilities = self.model.predict_proba(features)[:, 1]
        return {field_name: float(score) for field_name, score in zip(field_names, probabilities, strict=True)}


class PrototypeLearnableFieldScorer:
    """Dependency-light field ranker prototype for external benchmark probing.

    This is intentionally a small transparent ranker rather than an end-to-end
    neural extractor. It combines alias anchors, open financial concept phrases,
    character/token similarity, and value-type compatibility into one calibrated
    score. A later trained ranker can keep the same ``score_fields`` interface.
    """

    def __init__(self) -> None:
        self.last_trace: dict[str, Any] = {}

    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        traces: dict[str, dict[str, float]] = {}
        for field_name in field_names:
            lexical = lexical_field_score(query, field_name)
            concept = _concept_phrase_score(query, field_name)
            token = _best_token_similarity(query, field_name)
            semantic_type = semantic_compatibility_score(query, field_name)
            score = (
                0.42 * min(lexical, 1.5) / 1.5
                + 0.30 * concept
                + 0.18 * token
                + 0.10 * semantic_type
            )
            score = max(0.0, min(1.0, score))
            scores[field_name] = score
            traces[field_name] = {
                "lexical": lexical,
                "concept": concept,
                "token": token,
                "semantic_type": semantic_type,
                "final": score,
            }
        ranked_fields = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.last_trace = {
            "strategy": "learnable_ranker",
            "model": "prototype_linear_financial_field_ranker_v0",
            "field_traces": traces,
            "ranked_fields": ranked_fields,
        }
        return scores


class MoEFieldScorer:
    """Soft-gated mixture of field grounding experts."""

    def __init__(self, experts: Mapping[str, FieldScorer]) -> None:
        if not experts:
            raise ValueError("MoEFieldScorer requires at least one expert.")
        self.experts = dict(experts)
        self.last_trace: dict[str, Any] = {}

    def score_fields(self, query: str, field_names: Sequence[str]) -> dict[str, float]:
        expert_scores = {
            name: expert.score_fields(query, field_names)
            for name, expert in self.experts.items()
        }
        normalized_scores = {
            name: _normalize_score_dict(scores)
            for name, scores in expert_scores.items()
        }
        gate_weights = self._gate_weights(expert_scores, normalized_scores)
        final_scores = {
            field_name: float(
                sum(
                    gate_weights[expert_name] * normalized_scores[expert_name].get(field_name, 0.0)
                    for expert_name in self.experts
                )
            )
            for field_name in field_names
        }
        ranked_fields = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        self.last_trace = {
            "gate_weights": gate_weights,
            "expert_scores": expert_scores,
            "normalized_expert_scores": normalized_scores,
            "final_scores": final_scores,
            "ranked_fields": ranked_fields,
        }
        return final_scores

    def precompute_texts(self, texts: Sequence[str]) -> None:
        for expert in self.experts.values():
            precompute = getattr(expert, "precompute_texts", None)
            if precompute is not None:
                precompute(texts)

    def _gate_weights(
        self,
        expert_scores: Mapping[str, Mapping[str, float]],
        normalized_scores: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        logits = {name: 0.0 for name in self.experts}
        lexical_name = _find_expert_name(self.experts, "lexical")
        learned_name = _find_expert_name(self.experts, "learned")
        embedding_name = _find_expert_name(self.experts, "embedding")
        type_name = _find_expert_name(self.experts, "type") or _find_expert_name(self.experts, "semantic")

        lexical_exact = False
        if lexical_name is not None:
            lexical_top = _top_score(expert_scores[lexical_name])
            lexical_exact = lexical_top >= 1.0
            if lexical_exact:
                logits[lexical_name] += 1.4
            else:
                logits[lexical_name] -= 0.2

        if learned_name is not None:
            logits[learned_name] -= 0.2
            if _score_margin(normalized_scores[learned_name]) >= 0.25:
                logits[learned_name] += 0.35

        if embedding_name is not None:
            logits[embedding_name] += 0.25
            if not lexical_exact:
                logits[embedding_name] += 0.75
            if _score_margin(normalized_scores[embedding_name]) >= _score_margin(
                normalized_scores.get(lexical_name, {})
            ):
                logits[embedding_name] += 0.35
            elif lexical_exact:
                logits[embedding_name] -= 0.35

        if type_name is not None:
            logits[type_name] += 0.1
            if not lexical_exact:
                logits[type_name] += 0.45
            if _score_margin(normalized_scores[type_name]) >= 0.18:
                logits[type_name] += 0.55
            elif lexical_exact:
                logits[type_name] -= 0.35

        return _softmax(logits)


def lexical_field_score(query: str, field_name: str) -> float:
    normalized_query = normalize_identifier(query).replace("_", "")
    alias_scores = []
    for alias in field_aliases(field_name):
        normalized_alias = normalize_identifier(alias).replace("_", "")
        if not normalized_alias:
            continue
        if normalized_alias in normalized_query:
            alias_scores.append(1.0 + min(len(normalized_alias) / 20.0, 0.5))
        else:
            alias_scores.append(char_ngram_similarity(normalized_query, normalized_alias))
    return max(alias_scores) if alias_scores else 0.0


def pair_features(query: str, field_name: str) -> list[float]:
    normalized_query = normalize_identifier(query).replace("_", "")
    aliases = [normalize_identifier(alias).replace("_", "") for alias in field_aliases(field_name)]
    aliases = [alias for alias in aliases if alias]
    if not aliases:
        return [0.0] * 6

    contains = [1.0 if alias in normalized_query else 0.0 for alias in aliases]
    ngram_scores = [char_ngram_similarity(normalized_query, alias) for alias in aliases]
    token_scores = [token_jaccard_similarity(normalized_query, alias) for alias in aliases]
    best_alias_len = max((len(alias) for alias, score in zip(aliases, ngram_scores, strict=True) if score == max(ngram_scores)), default=0)
    length_ratio = min(best_alias_len, len(normalized_query)) / max(best_alias_len, len(normalized_query), 1)
    return [
        max(contains),
        max(ngram_scores),
        max(token_scores),
        lexical_field_score(query, field_name),
        length_ratio,
        min(best_alias_len / 20.0, 1.0),
    ]


def field_matching_text(field_name: str) -> str:
    terms = [field_name, field_name.replace("_", " "), *field_aliases(field_name)]
    deduped = list(dict.fromkeys(term for term in terms if term))
    return " ; ".join(deduped)


def _concept_phrase_score(query: str, field_name: str) -> float:
    query_compact = normalize_identifier(query).replace("_", "")
    phrases = _financial_concept_phrases(field_name)
    if not phrases:
        return 0.0
    scores: list[float] = []
    for phrase in phrases:
        phrase_compact = normalize_identifier(phrase).replace("_", "")
        if not phrase_compact:
            continue
        if phrase_compact in query_compact:
            scores.append(1.0)
        else:
            scores.append(char_ngram_similarity(query_compact, phrase_compact))
    return max(scores, default=0.0)


def _best_token_similarity(query: str, field_name: str) -> float:
    normalized_query = normalize_identifier(query)
    terms = [field_name, field_name.replace("_", " "), *field_aliases(field_name), *_financial_concept_phrases(field_name)]
    return max(
        (
            token_jaccard_similarity(normalized_query, normalize_identifier(term))
            for term in terms
            if term
        ),
        default=0.0,
    )


def _financial_concept_phrases(field_name: str) -> tuple[str, ...]:
    normalized = normalize_identifier(field_name)
    phrases: list[str] = []
    if "revenue" in normalized or normalized in {"sales", "net_sales", "total_net_sales"}:
        phrases.extend(
            [
                "revenue from contracts with customers",
                "total revenues",
                "total revenue",
                "net revenue",
                "net sales",
                "sales revenue",
                "contract revenue",
                "customer revenue",
                "turnover",
            ]
        )
    if "net_income" in normalized or normalized == "net_profit":
        phrases.extend(
            [
                "net earnings",
                "net income",
                "net profit",
                "profit attributable",
                "income attributable",
                "earnings attributable",
            ]
        )
    if "operating_income" in normalized or "operating_profit" in normalized:
        phrases.extend(
            [
                "operating earnings",
                "operating income",
                "income from operations",
                "operating profit",
            ]
        )
    if "cash" in normalized:
        phrases.extend(
            [
                "cash and cash equivalents",
                "cash equivalents",
                "cash balance",
            ]
        )
    if "asset" in normalized:
        phrases.extend(["total assets", "assets", "asset base"])
    if "liabilit" in normalized:
        phrases.extend(["total liabilities", "liabilities", "obligations"])
    if "employee" in normalized or "headcount" in normalized:
        phrases.extend(["employees", "headcount", "workforce", "staff"])
    if "gross_margin" in normalized:
        phrases.extend(["gross margin", "gross profit"])
    if "research" in normalized or normalized.startswith("rd_"):
        phrases.extend(["research and development", "r&d", "research expense"])
    return tuple(dict.fromkeys(phrases))


def char_ngram_similarity(left: str, right: str, n: int = 3) -> float:
    left_ngrams = _ngrams(left, n)
    right_ngrams = _ngrams(right, n)
    if not left_ngrams or not right_ngrams:
        return 0.0
    intersection = len(left_ngrams & right_ngrams)
    return intersection / math.sqrt(len(left_ngrams) * len(right_ngrams))


def token_jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(filter(None, left.split("_")))
    right_tokens = set(filter(None, right.split("_")))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _as_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Embedding encoder must return a 1D or 2D numeric array.")
    return array


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return values / norms


def _cosine_to_unit_score(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _normalize_score_dict(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = [float(value) for value in scores.values()]
    max_value = max(values)
    min_value = min(values)
    if max_value <= 1.0 and min_value >= 0.0:
        return {name: float(value) for name, value in scores.items()}
    if max_value > 0.0 and min_value >= 0.0:
        return {name: max(0.0, min(1.0, float(value) / max_value)) for name, value in scores.items()}
    span = max(max_value - min_value, 1e-12)
    return {name: (float(value) - min_value) / span for name, value in scores.items()}


def _top_score(scores: Mapping[str, float]) -> float:
    return max((float(value) for value in scores.values()), default=0.0)


def _score_margin(scores: Mapping[str, float]) -> float:
    ordered = sorted((float(value) for value in scores.values()), reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else 0.0
    return ordered[0] - ordered[1]


def _find_expert_name(experts: Mapping[str, FieldScorer], keyword: str) -> str | None:
    for name in experts:
        if keyword in name.lower():
            return name
    return None


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    max_logit = max(logits.values())
    exp_values = {name: math.exp(value - max_logit) for name, value in logits.items()}
    denominator = sum(exp_values.values())
    return {name: value / denominator for name, value in exp_values.items()}
