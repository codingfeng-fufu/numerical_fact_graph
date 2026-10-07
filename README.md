# Numerical Fact Graphs for Auditable Financial Question Answering

Anonymous code package for the COLING/ARR paper:

> **Numerical Fact Graphs for Auditable Financial Question Answering: Evidence-Bounded Calculation and Failure Localization**

## Purpose

The paper represents financial numbers as typed facts with structural context and evidence-oriented identifiers. A question selects facts from this representation, connects them to a typed calculation plan, checks the plan, and executes supported operators deterministically.

This repository contains the reusable deterministic code for that contract. It is intended to let reviewers inspect and test the numerical fact graph, typed plans, operators, validation rules and replay utilities without access to private evaluation artifacts.

The package corresponds to the following parts of the paper:

| Paper concept | Code area |
|---|---|
| Numerical facts and fact graph projection | `src/graph_numeric/core/` |
| Typed calculation plans and operators | `src/graph_numeric/operators/` and `src/graph_numeric/pipeline/` |
| Evidence-oriented identifiers and binding | `src/graph_numeric/binding/` and `src/graph_numeric/learning/` |
| Deterministic execution and output normalization | `src/graph_numeric/operators/` and `src/graph_numeric/runtime/` |
| Validation and audit utilities | `src/graph_numeric/audit/` |
| Deterministic tests | `tests/` |

## What can be run from this repository

The package provides:

- construction and querying of synthetic numerical fact graphs;
- typed operator plans for aggregation, ratios, differences and related calculations;
- deterministic plan validation and refusal checks;
- evidence-oriented token and anchor utilities;
- canonical derivation and replay helpers;
- unit tests that do not call an LLM or require network access.

The smoke test and deterministic test suite run on hand-constructed or synthetic facts. They verify the calculation and validation core.

## What is not included

The paper's aggregate FinQA, TAT-QA and annual-report results were produced in a separate controlled evaluation workspace. This anonymous package does not include:

- sealed evaluation data or identifiers;
- item-level predictions and private audit labels;
- prompt or extraction caches;
- model checkpoints or external API outputs;
- author information, credentials or workspace-specific paths.

The replay report builder is included for code inspection, but its frozen item-level inputs are not included. The paper reports that the revised serializer reproduces 91/91 accepted development outputs while noting that the serializer can select a matching plan after observing the answer.

## Requirements

- Python 3.11 or newer;
- no network connection for the smoke test or deterministic tests;
- approximately 1 GB of free disk space for a normal Python environment and test run.

## Install

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the smoke test

```bash
PYTHONPATH=src python scripts/smoke_query.py
```

Expected output:

```text
ok answer=2150.0 operator=SUM
```

The script creates three synthetic company facts, selects the two `TECH` facts from 2024, and computes their revenue sum:

```text
1200 + 950 = 2150
```

## Run the deterministic tests

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

The prepared environment passes 182 tests. These tests cover graph construction, operator solving, typed plans, validation, verifier behavior, strict anchors and abstention audits.

## Replay audit utility

`scripts/build_replay_conformance.py` builds the aggregate replay report when an evaluator supplies the corresponding frozen item-level input files. It is included for inspection and controlled reproduction; those inputs are intentionally excluded from this anonymous release.

## Anonymous review boundary

This repository is prepared for double-blind review. It contains no author names, email addresses, remote repository URLs, credentials, model caches or private evidence. Please preserve the repository contents and history when creating an anonymous mirror. `MANIFEST.json` records the package scope, test results and file hashes.
