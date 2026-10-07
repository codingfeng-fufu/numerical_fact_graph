# Anonymous Numerical Fact Graph package

This repository accompanies the anonymous COLING/ARR submission **Numerical Fact Graphs for Auditable Financial Question Answering: Evidence-Bounded Calculation and Failure Localization**.

It contains the reusable Numerical Fact Graph representation, typed operator plans, deterministic execution and validation components, selected audit utilities, and tests for the paper's core behaviors. The package is prepared for anonymous review.

## Scope

The package supports:

- typed numerical facts with structural fields and evidence-oriented identifiers;
- deterministic operators and calculation plans;
- plan validation, unit checks and refusal conditions;
- canonical derivation replay utilities;
- selected audit and storage-contract tests.

The package does not include private item-level outputs, model caches, sealed data, checkpoints, API keys, author information or the internal experiment workspace. Full sealed-set results are reported in the paper and aggregate evidence is retained privately for review verification.

## Install and run the deterministic tests

Use Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
PYTHONPATH=src pytest -q tests/test_graph_numeric_expression_plan.py tests/test_graph_numeric_full_operators.py tests/test_graph_numeric_operator_solvers.py tests/test_graph_numeric_pipeline_validation.py tests/test_graph_numeric_verifier.py tests/test_strict_anchor.py tests/test_s4prime_abstention_audit.py
```

The tests use hand-constructed or synthetic facts. They do not call an LLM or require network access.

## Minimal query example

```bash
PYTHONPATH=src python scripts/smoke_query.py
```

The script builds a small synthetic fact graph, runs a deterministic aggregate query, and checks the returned operator and value.

## Replay utility

`scripts/build_replay_conformance.py` contains the aggregate report builder used for the development replay audit. It requires frozen item-level inputs supplied separately by an authorized evaluator. No such inputs are included in this anonymous package.

## Anonymity and data boundary

This package has no `.git` directory, author names, remote URLs, credentials, model caches, absolute workspace paths or private evidence. Please preserve that boundary when creating the anonymous remote repository.
