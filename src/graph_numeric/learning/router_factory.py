from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

from graph_numeric.learning.router import RoutingResult
from graph_numeric.learning.router import EmbeddingRouter, HybridRouter, RuleBasedRouter


DEFAULT_EMBEDDING_ROUTER_DIR = Path("models/graph_numeric/embedding_router_full")


def build_router(
    router_type: str = "rule",
    *,
    model_dir: str | Path | None = None,
    rule_threshold: float = 0.5,
    device: str | None = None,
) -> Any:
    router_type = router_type.lower()
    if router_type == "rule":
        return RuleBasedRouter()
    resolved_model_dir = Path(model_dir) if model_dir is not None else DEFAULT_EMBEDDING_ROUTER_DIR
    if router_type == "embedding":
        return EmbeddingRouter(resolved_model_dir, device=device)
    if router_type == "hybrid":
        try:
            return HybridRouter(resolved_model_dir, rule_threshold=rule_threshold, device=device)
        except (RuntimeError, FileNotFoundError, ImportError) as exc:
            return RuntimeFallbackHybridRouter(str(exc))
    raise ValueError(f"Unsupported router type: {router_type}")


class RuntimeFallbackHybridRouter:
    """Hybrid-compatible router used when embedding runtime is unavailable."""

    def __init__(self, reason: str) -> None:
        self._rule = RuleBasedRouter()
        self.reason = reason

    def route(self, query: str) -> RoutingResult:
        routed = self._rule.route(query)
        return replace(
            routed,
            route_type="hybrid_rule_runtime_fallback",
            trace={
                **(routed.trace or {}),
                "embedding_unavailable": True,
                "embedding_unavailable_reason": self.reason,
            },
        )

    def classify_intent(self, query: str) -> str:
        return self.route(query).intent
