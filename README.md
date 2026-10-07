# Numerical Fact Graphs for Auditable Financial Question Answering

Anonymous reproducibility package for the COLING/ARR submission:

> **Numerical Fact Graphs for Auditable Financial Question Answering: Evidence-Bounded Calculation and Failure Localization**

This repository contains the deterministic core used to represent numerical facts, build typed calculation plans, validate operands, execute supported operators, and inspect selected failure conditions in financial numerical question answering.

## What is included

- `src/graph_numeric/`: numerical fact graph structures, typed plans, operators, validation, deterministic execution and replay helpers;
- `tests/`: deterministic tests for graph construction, operator solving, plan validation, verifier behavior, strict anchors and abstention audits;
- `scripts/smoke_query.py`: a small end-to-end query over synthetic facts;
- `scripts/build_replay_conformance.py`: the aggregate replay report builder used by the development audit;
- `docs/reproducibility_scope.md`: the boundary between this package and the private evaluation artifacts.

The package is a code and deterministic-test release. It does not contain the sealed evaluation set, item-level outputs, model checkpoints, prompt caches or private audit records.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

The deterministic tests do not call an LLM or require network access.

## Smoke test

```bash
PYTHONPATH=src python scripts/smoke_query.py
```

Expected output:

```text
ok answer=2150.0 operator=SUM
```

The smoke test creates three synthetic company facts, selects the two `TECH` facts for 2024, and evaluates their revenue sum.

## Deterministic tests

```bash
PYTHONPATH=src pytest -q \
  tests/test_graph_numeric_expression_plan.py \
  tests/test_graph_numeric_full_operators.py \
  tests/test_graph_numeric_operator_solvers.py \
  tests/test_graph_numeric_pipeline_validation.py \
  tests/test_graph_numeric_verifier.py \
  tests/test_strict_anchor.py \
  tests/test_s4prime_abstention_audit.py
```

The prepared environment passes 182 tests. The tests use hand-constructed or synthetic facts and are independent of the paper's sealed evaluation data.

## Replay audit utility

`python scripts/build_replay_conformance.py` builds the aggregate replay report when an evaluator supplies the corresponding frozen item-level inputs. Those inputs are intentionally absent from this repository. The paper reports the two serializer versions separately and describes the answer-conditioned limitation of the revised serializer.

## Reproducibility and anonymity

The paper reports aggregate FinQA, TAT-QA and annual-report results. This package provides the deterministic components needed to inspect the fact graph and calculation contract; private evaluation artifacts remain separate. The repository contains no author names, email addresses, remote URLs, credentials, model caches or workspace-specific paths. Please preserve this boundary when creating an anonymous mirror for review.

See `MANIFEST.json` for the package scope, test results and file hashes.
