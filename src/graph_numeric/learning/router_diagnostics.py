from __future__ import annotations

from collections import Counter
from typing import Any


def router_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate router trace fields emitted by graph_numeric benchmarks."""
    n = len(rows)
    route_type_counts = Counter(str(row.get("router_route_type") or "unknown") for row in rows)
    fallback_counts: Counter[str] = Counter()
    embedding_confidences: list[float] = []
    embedding_top3_hits = 0
    embedding_rows = 0
    embedding_unavailable = 0

    for row in rows:
        for op in row.get("router_fallback_operators") or []:
            fallback_counts[str(op)] += 1

        trace = row.get("router_trace") or {}
        if isinstance(trace, dict) and trace.get("embedding_unavailable"):
            embedding_unavailable += 1

        top3 = trace.get("top3_probs") if isinstance(trace, dict) else None
        if not isinstance(top3, dict) and isinstance(trace, dict):
            # Hybrid embedding route stores top3 at the top level together with rule context.
            top3 = trace.get("embedding_top3_probs")
        if not isinstance(top3, dict):
            continue

        embedding_rows += 1
        ordered = sorted(
            ((str(op), float(prob)) for op, prob in top3.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        if ordered:
            embedding_confidences.append(ordered[0][1])
        gold_operator = row.get("operator") or row.get("kind")
        if gold_operator is not None and str(gold_operator).upper() in {op.upper() for op, _ in ordered[:3]}:
            embedding_top3_hits += 1

    route_type_total = max(n, 1)
    return {
        "route_type_counts": dict(sorted(route_type_counts.items())),
        "fallback_operator_counts": dict(sorted(fallback_counts.items())),
        "embedding_top1_avg_confidence": round(
            sum(embedding_confidences) / max(len(embedding_confidences), 1),
            4,
        ),
        "embedding_top3_hit_rate": round(embedding_top3_hits / max(embedding_rows, 1), 4),
        "embedding_rows": embedding_rows,
        "embedding_unavailable_count": embedding_unavailable,
        "hybrid_rule_rate": round(
            sum(count for rt, count in route_type_counts.items() if rt in {"rule", "hybrid_rule_runtime_fallback"})
            / route_type_total,
            4,
        ),
        "hybrid_embedding_rate": round(
            sum(
                count
                for rt, count in route_type_counts.items()
                if rt in {"hybrid_emb", "embedding", "embedding_composite"}
            )
            / route_type_total,
            4,
        ),
    }
