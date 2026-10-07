from __future__ import annotations

from graph_numeric.core.attribute_graph import (
    AttributeValueGraph,
    AttributeValueToken,
)
from graph_numeric.audit.s4prime_abstention import (
    binding_requests_from_prompt_cache,
    classify_cached_candidate_reachability,
    classify_program_operand_reachability,
    extract_program_operands,
    locate_operand_sources,
)


def _token(token_id: str, value: float) -> AttributeValueToken:
    return AttributeValueToken(
        token_id=token_id,
        entity_id="company",
        company_name="Company",
        field_name="metric",
        field_label="Metric",
        value=value,
    )


def test_extract_program_operands_ignores_step_references_and_named_constants() -> None:
    operands = extract_program_operands(
        "multiply(342313, 113.39), divide(#0, const_1000000)"
    )

    assert operands == [342313.0, 113.39]


def test_reachability_assigns_missing_document_operand_to_s2() -> None:
    graph = AttributeValueGraph((_token("t1", 23596),), source_name="sample.md")

    result = classify_program_operand_reachability(
        program="subtract(23596, 63003), divide(#0, 63003)",
        question="What was the percentage decrease?",
        graph=graph,
    )

    assert result["classification"] == "s2_gold_token_missing"
    assert result["document_operands"] == [23596.0, 63003.0, 63003.0]
    assert result["missing_operands"] == [63003.0]
    assert result["missing_operand_occurrences"] == [63003.0, 63003.0]


def test_reachability_assigns_fully_reachable_operands_to_s4prime() -> None:
    graph = AttributeValueGraph(
        (_token("t1", 23596), _token("t2", 63003), _token("t3", 63003)),
        source_name="sample.md",
    )

    result = classify_program_operand_reachability(
        program="subtract(23596, 63003), divide(#0, 63003)",
        question="What was the percentage decrease?",
        graph=graph,
    )

    assert result["classification"] == "s4prime_binding_shortfall"
    assert result["missing_operands"] == []
    assert result["matched_token_ids"] == ["t1", "t2", "t3"]


def test_repeated_denominator_can_reuse_one_graph_token() -> None:
    graph = AttributeValueGraph(
        (_token("t1", 23596), _token("t2", 63003)),
        source_name="sample.md",
    )

    result = classify_program_operand_reachability(
        program="subtract(23596, 63003), divide(#0, 63003)",
        question="What was the percentage decrease?",
        graph=graph,
    )

    assert result["classification"] == "s4prime_binding_shortfall"
    assert result["missing_operands"] == []
    assert result["missing_operand_occurrences"] == [63003.0]


def test_question_literal_is_not_charged_to_document_graph() -> None:
    graph = AttributeValueGraph(
        (_token("t1", 100), _token("t2", 25)),
        source_name="sample.md",
    )

    result = classify_program_operand_reachability(
        program="multiply(25, 4), divide(#0, 100)",
        question="What was the total for 4 quarters as a percentage?",
        graph=graph,
    )

    assert result["question_constants"] == [4.0]
    assert result["document_operands"] == [25.0, 100.0]
    assert result["classification"] == "s4prime_binding_shortfall"


def test_binding_request_cache_recovers_closed_set_candidate_tokens() -> None:
    cache = {
        "hash": {
            "request": {
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": (
                            'binding_request:\n{"question":"What changed?",'
                            '"candidate_tokens":[{"id":"t1","value":42.0}]}'
                        ),
                    },
                ]
            },
            "response": {},
        }
    }

    requests = binding_requests_from_prompt_cache(cache)

    assert requests["What changed?"][0]["candidate_tokens"] == [
        {"id": "t1", "value": 42.0}
    ]


def test_cached_candidates_support_exact_gold_token_reachability_audit() -> None:
    result = classify_cached_candidate_reachability(
        program="subtract(42, 30)",
        question="What changed?",
        candidate_tokens=[
            {"id": "t42", "value": 42.0, "metric": "revenue", "year": 2020},
            {"id": "t30", "value": 30.0, "metric": "revenue", "year": 2019},
        ],
    )

    assert result["classification"] == "s4prime_binding_shortfall"
    assert result["matched_token_ids"] == ["t42", "t30"]


def test_operand_source_location_handles_parenthesized_table_values() -> None:
    locations = locate_operand_sources(
        operands=[34.0, 3669.0, 42.0],
        table=[["Project K", "(34)"], ["Underlying SGA", "$3,669"]],
        text=["During 2013, the company recorded $42 million of charges."],
    )

    assert locations == [
        {"operand": 34.0, "location": "table"},
        {"operand": 3669.0, "location": "table"},
        {"operand": 42.0, "location": "text"},
    ]
