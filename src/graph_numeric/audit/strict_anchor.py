from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


VOLATILE_KEYS = {
    "avg_latency_ms",
    "benchmark_attempt",
    "index",
    "latency_ms",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_result_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_result_value(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize_result_value(item) for item in value]
    return value


def compare_run_rows(
    run1_rows: Iterable[dict[str, Any]],
    run2_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    run1 = _rows_by_sample_id(run1_rows)
    run2 = _rows_by_sample_id(run2_rows)
    run1_ids = set(run1)
    run2_ids = set(run2)
    common_ids = sorted(run1_ids & run2_ids)
    diff_sample_ids = [
        sample_id
        for sample_id in common_ids
        if normalize_result_value(run1[sample_id])
        != normalize_result_value(run2[sample_id])
    ]
    return {
        "run1_count": len(run1),
        "run2_count": len(run2),
        "missing_from_run1": sorted(run2_ids - run1_ids),
        "missing_from_run2": sorted(run1_ids - run2_ids),
        "diff_sample_ids": diff_sample_ids,
        "deterministic": not diff_sample_ids and run1_ids == run2_ids,
    }


def merge_shard_payloads(
    shard_paths: Iterable[str | Path],
    *,
    label: str,
) -> dict[str, Any]:
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise ValueError("at least one shard is required")
    payloads = [_read_json(path) for path in paths]
    shard_indexes = [int(payload["shard_index"]) for payload in payloads]
    if len(shard_indexes) != len(set(shard_indexes)):
        raise ValueError("duplicate shard_index")
    expected_shards = int(payloads[0].get("num_shards") or len(payloads))
    if sorted(shard_indexes) != list(range(expected_shards)):
        raise ValueError(
            f"incomplete shard set: got {sorted(shard_indexes)}, expected 0..{expected_shards - 1}"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in sorted(payloads, key=lambda item: int(item["shard_index"])):
        for row in payload.get("rows") or []:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError("row missing sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(dict(row))
    rows.sort(key=lambda row: str(row["sample_id"]))
    correct_count = sum(bool(row.get("answer_match")) for row in rows)
    first = payloads[0]
    return {
        "label": label,
        "benchmark": first.get("benchmark"),
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "seed": first.get("seed"),
        "mode": first.get("mode"),
        "binder_mode": first.get("binder_mode"),
        "llm_cache_only": first.get("llm_cache_only"),
        "num_shards": expected_shards,
        "shard_paths": [str(path) for path in paths],
        "num_examples": len(rows),
        "correct_count": correct_count,
        "answer_accuracy": correct_count / len(rows) if rows else 0.0,
        "rows": rows,
    }


def _rows_by_sample_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("row missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        result[sample_id] = dict(row)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
