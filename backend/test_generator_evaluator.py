import pytest

from evaluation.generator_evaluator import (
    evaluate_generator,
    evaluate_citations,
    extract_citations,
)


@pytest.fixture
def retrieved_chunks():
    return [
        {
            "text": "Customers can request a refund within 30 days of purchase.",
            "metadata": {
                "document_name": "ML_Questions.pdf",
                "page_number": 2,
                "chunk_id": "page_2_start_0",
            },
        }
    ]


def test_evaluate_generator_returns_expected_structure(retrieved_chunks):
    result = evaluate_generator(
        question="Refund period kitna hai?",
        generated_answer="Refund 30 days ke andar available hai.",
        ground_truth_answer="Refund 30 days ke andar available hai.",
        retrieved_chunks=retrieved_chunks,
    )

    assert "question" in result
    assert "generated_answer" in result
    assert "ground_truth_answer" in result
    assert "metrics" in result
    assert "evaluation_note" in result

    metrics = result["metrics"]

    assert "answer_correctness" in metrics
    assert "context_support" in metrics
    assert "answer_completeness" in metrics
    assert "citation_validation" in metrics
    assert "claim_level_faithfulness" in metrics


def test_identical_answer_has_perfect_lexical_correctness(retrieved_chunks):
    answer = "Refund 30 days ke andar available hai."

    result = evaluate_generator(
        question="Refund period kitna hai?",
        generated_answer=answer,
        ground_truth_answer=answer,
        retrieved_chunks=retrieved_chunks,
    )

    correctness = result["metrics"]["answer_correctness"]

    assert correctness["score"] == 1.0
    assert correctness["evaluation_type"] == "lexical_proxy"


@pytest.mark.parametrize(
    "citation",
    [
        "[Source: ML_Questions.pdf, Page 2]",
        "*Source:* ML_Questions.pdf, Page 2",
        "(स्रोत: ML_Questions.pdf, पृष्ठ 2)",
        "【ML_Questions.pdf, p.2】",
        "【ML_Questions.pdf, page 2】",
    ],
)
def test_supported_citation_formats_are_extracted(citation):
    extracted = extract_citations(citation)

    assert len(extracted) >= 1, (
        f"Citation was not extracted: {citation}"
    )

    assert extracted[0]["document"].casefold() == "ml_questions.pdf"
    assert extracted[0]["page"] == 2


def test_filename_without_page_is_not_treated_as_valid_citation():
    extracted = extract_citations(
        "The information came from ML_Questions.pdf."
    )

    assert extracted == []


def test_citation_validation_returns_expected_fields(retrieved_chunks):
    result = evaluate_citations(
        generated_answer=(
            "Refund is available within 30 days. "
            "[Source: ML_Questions.pdf, Page 2]"
        ),
        retrieved_chunks=retrieved_chunks,
    )

    assert "citation_count" in result
    assert "valid_citation_count" in result
    assert "invalid_citation_count" in result
    assert "citation_validity_score" in result
    assert "citations" in result

    assert result["citation_count"] == 1
    assert result["valid_citation_count"] == 1
    assert result["invalid_citation_count"] == 0
    assert result["citation_validity_score"] == 1.0

    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_reference_found"] is True