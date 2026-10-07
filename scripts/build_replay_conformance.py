from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph_numeric.audit.strict_anchor import sha256_file  # noqa: E402
from graph_numeric.runtime.canonical_derivation import (  # noqa: E402
    from_dict as canonical_from_dict,
    replay as replay_canonical,
)


EXPECTED_METRICS = {
    "total": 100,
    "correct": 61,
    "accepted": 91,
    "pipeline_accepted": 64,
    "fallback_accepted": 27,
    "replay_available": 91,
    "replay_verified": 91,
}
DECISION_KEYS = (
    "status",
    "predicted_answer",
    "answer_match",
    "selected_answer_source",
    "abstain_reason",
    "pipeline_decision",
)


def build_replay_conformance(
    frozen_path: str | Path,
    run1_path: str | Path,
    run2_path: str | Path,
) -> dict[str, Any]:
    frozen_path = Path(frozen_path)
    run1_path = Path(run1_path)
    run2_path = Path(run2_path)
    frozen_rows = _load_rows(frozen_path)
    run1_rows = _load_rows(run1_path)
    run2_rows = _load_rows(run2_path)

    run_diff_count = _decision_diff_count(run1_rows, run2_rows)
    frozen_decision_diff_count = _decision_diff_count(frozen_rows, run1_rows)
    if run_diff_count:
        raise ValueError(f"strict run decision diff count must be 0, got {run_diff_count}")
    if frozen_decision_diff_count:
        raise ValueError(
            f"frozen decision diff count must be 0, got {frozen_decision_diff_count}"
        )

    replay_rows = [_row_replay_summary(row) for row in run1_rows]
    metrics = _metrics(run1_rows, replay_rows)
    for key, expected in EXPECTED_METRICS.items():
        actual = metrics.get(key)
        if actual != expected:
            raise ValueError(f"{key} expected {expected}, got {actual}")

    by_source = _source_breakdown(run1_rows, replay_rows)
    return {
        "schema": "trace_replay_conformance_v2",
        "scope": "FinQA development-100",
        "model": "deepseek-v4-flash",
        "mode": "llm_table_narrow",
        "binder": "s4prime",
        "inputs": {
            "frozen_sha256": sha256_file(frozen_path),
            "run1_sha256": sha256_file(run1_path),
            "run2_sha256": sha256_file(run2_path),
        },
        "strict_protocol": {
            "runs": 2,
            "run_decision_diff_count": run_diff_count,
            "frozen_decision_diff_count": frozen_decision_diff_count,
            "exceptions": sum(1 for row in run1_rows if row.get("status") == "exception"),
            "cache_misses": _cache_miss_count(run1_rows) + _cache_miss_count(run2_rows),
        },
        "decision_diff_count": frozen_decision_diff_count,
        "metrics": metrics,
        "by_source": by_source,
    }


def write_outputs(evidence: Mapping[str, Any], output: str | Path, report: str | Path) -> None:
    output_path = Path(output)
    report_path = Path(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(evidence), encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"expected rows list in {path}")
    return [dict(row) for row in rows]


def _decision_diff_count(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> int:
    left = _rows_by_sample_id(left_rows)
    right = _rows_by_sample_id(right_rows)
    if set(left) != set(right):
        return len(set(left) ^ set(right))
    return sum(
        1
        for sample_id in left
        if _decision_projection(left[sample_id]) != _decision_projection(right[sample_id])
    )


def _rows_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("row missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _decision_projection(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in DECISION_KEYS)


def _row_replay_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("selected_answer_source") == "abstain":
        return {"available": False, "verified": False, "value": None}
    record = row.get("derivation_record")
    if not isinstance(record, Mapping):
        return {"available": False, "verified": False, "value": None}
    plan_payload = record.get("canonical_plan")
    if not isinstance(plan_payload, Mapping):
        return {"available": False, "verified": False, "value": None}
    plan = canonical_from_dict(plan_payload)
    fact_values = {
        str(operand.get("token_id")): operand.get("normalized_value")
        for operand in plan_payload.get("operands", [])
        if isinstance(operand, Mapping)
        and operand.get("source_kind") == "token"
        and operand.get("token_id") is not None
    }
    replay_value = replay_canonical(plan, fact_values=fact_values)
    return {
        "available": True,
        "verified": _answers_equivalent(row.get("predicted_answer"), replay_value),
        "value": replay_value,
    }


def _answers_equivalent(left: Any, right: Any) -> bool:
    left_num = _coerce_number(left)
    right_num = _coerce_number(right)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) <= max(1e-4, abs(right_num) * 1e-4)
    return str(left).strip().lower() == str(right).strip().lower()


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _metrics(rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> dict[str, int]:
    accepted_indexes = [
        index
        for index, row in enumerate(rows)
        if row.get("selected_answer_source") != "abstain"
    ]
    return {
        "total": len(rows),
        "correct": sum(1 for row in rows if row.get("answer_match")),
        "accepted": len(accepted_indexes),
        "pipeline_accepted": sum(1 for row in rows if row.get("selected_answer_source") == "pipeline"),
        "fallback_accepted": sum(1 for row in rows if row.get("selected_answer_source") == "financial_fallback"),
        "replay_available": sum(1 for index in accepted_indexes if replay_rows[index]["available"]),
        "replay_verified": sum(1 for index in accepted_indexes if replay_rows[index]["verified"]),
    }


def _source_breakdown(rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    sources = ("pipeline", "financial_fallback")
    result: dict[str, dict[str, int]] = {}
    for source in sources:
        indexes = [
            index
            for index, row in enumerate(rows)
            if row.get("selected_answer_source") == source
        ]
        result[source] = {
            "accepted": len(indexes),
            "available": sum(1 for index in indexes if replay_rows[index]["available"]),
            "verified": sum(1 for index in indexes if replay_rows[index]["verified"]),
            "unavailable": sum(1 for index in indexes if not replay_rows[index]["available"]),
            "mismatch": sum(
                1
                for index in indexes
                if replay_rows[index]["available"] and not replay_rows[index]["verified"]
            ),
        }
    return result


def _cache_miss_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if "llm_prompt_cache_miss" in json.dumps(row, ensure_ascii=False))


def _markdown_report(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    by_source = evidence["by_source"]
    return "\n".join(
        [
            "# Trace Replay Conformance v2",
            "",
            f"- total: `{metrics['total']}`",
            f"- correct: `{metrics['correct']}`",
            f"- accepted: `{metrics['accepted']}`",
            f"- replay_verified: `{metrics['replay_verified']}/{metrics['replay_available']}`",
            f"- pipeline: `{by_source['pipeline']['verified']}/{by_source['pipeline']['accepted']}`",
            f"- financial_fallback: `{by_source['financial_fallback']['verified']}/{by_source['financial_fallback']['accepted']}`",
            f"- frozen_decision_diff_count: `{evidence['strict_protocol']['frozen_decision_diff_count']}`",
            f"- run_decision_diff_count: `{evidence['strict_protocol']['run_decision_diff_count']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    evidence = build_replay_conformance(args.frozen, args.run1, args.run2)
    write_outputs(evidence, args.output, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
