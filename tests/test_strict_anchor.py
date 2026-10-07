from __future__ import annotations

import json

import pytest

from graph_numeric.audit.strict_anchor import (
    compare_run_rows,
    merge_shard_payloads,
    sha256_file,
)


def _payload(shard: int, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "dataset": "finqa",
        "split": "test",
        "seed": 7,
        "mode": "llm_table_narrow",
        "binder_mode": "s4prime",
        "llm_cache_only": True,
        "num_shards": 2,
        "shard_index": shard,
        "rows": rows,
    }


def test_compare_run_rows_ignores_latency_but_detects_behavior_changes() -> None:
    run1 = [
        {"sample_id": "a", "answer_match": True, "latency_ms": 1.0},
        {"sample_id": "b", "answer_match": False, "latency_ms": 2.0},
    ]
    run2 = [
        {"sample_id": "a", "answer_match": True, "latency_ms": 9.0},
        {"sample_id": "b", "answer_match": False, "latency_ms": 8.0},
    ]

    assert compare_run_rows(run1, run2)["diff_sample_ids"] == []

    run2[1]["pipeline_decision"] = "answered"
    assert compare_run_rows(run1, run2)["diff_sample_ids"] == ["b"]


def test_merge_shards_rejects_duplicate_samples_and_reports_accuracy(tmp_path) -> None:
    shard0 = tmp_path / "shard0.json"
    shard1 = tmp_path / "shard1.json"
    shard0.write_text(
        json.dumps(_payload(0, [{"sample_id": "a", "answer_match": True}])),
        encoding="utf-8",
    )
    shard1.write_text(
        json.dumps(_payload(1, [{"sample_id": "b", "answer_match": False}])),
        encoding="utf-8",
    )

    merged = merge_shard_payloads([shard0, shard1], label="strict_b")

    assert merged["num_examples"] == 2
    assert merged["correct_count"] == 1
    assert merged["answer_accuracy"] == 0.5

    shard1.write_text(
        json.dumps(_payload(1, [{"sample_id": "a", "answer_match": False}])),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        merge_shard_payloads([shard0, shard1], label="strict_b")


def test_sha256_file_is_stable(tmp_path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"strict-cache-anchor")

    assert sha256_file(path) == "025b3e945fe59817723f14486222fa90a0cf6f50645417d35fc3a840cb6dcf8f"
