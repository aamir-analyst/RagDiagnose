import pytest

from evaluation.semantic_evaluator import (
    prepare_context,
    parse_json_response,
    validate_score,
    normalize_claims,
    evaluate_semantic_quality,
)


def test_prepare_context_with_valid_chunks():
    chunks = [
        {
            "page_content": "Machine Learning learns patterns from data.",
            "metadata": {
                "source": "ML_Questions.pdf",
                "page": 1,
            },
        }
    ]

    context = prepare_context(chunks)

    assert "Machine Learning learns patterns from data." in context
    assert "ML_Questions.pdf" in context
    assert "1" in context


def test_prepare_context_with_empty_chunks():
    context = prepare_context([])

    assert context == ""


def test_prepare_context_ignores_invalid_chunks():
    chunks = [
        None,
        {"metadata": "invalid metadata"},
        {"page_content": ""},
        {
            "text": "Valid context text",
            "metadata": {"document_name": "notes.pdf", "page_number": 2},
        },
    ]

    context = prepare_context(chunks)

    assert "Valid context text" in context
    assert "notes.pdf" in context


def test_parse_valid_json_response():
    response = """
    {
        "faithfulness": 0.9,
        "answer_relevance": 0.8,
        "answer_correctness": 0.85,
        "hallucination_risk": 0.1
    }
    """

    result = parse_json_response(response)

    assert result["faithfulness"] == 0.9
    assert result["answer_relevance"] == 0.8
    assert result["answer_correctness"] == 0.85
    assert result["hallucination_risk"] == 0.1


def test_parse_json_inside_markdown_fence():
    response = """```json
    {
        "faithfulness": 0.9
    }
    ```"""

    result = parse_json_response(response)

    assert result["faithfulness"] == 0.9


def test_parse_json_with_surrounding_text():
    response = """
    Evaluation result:
    {"faithfulness": 0.75}
    End of result.
    """

    result = parse_json_response(response)

    assert result["faithfulness"] == 0.75


def test_invalid_json_raises_value_error():
    with pytest.raises(ValueError):
        parse_json_response("This is not valid JSON")


@pytest.mark.parametrize(
    "score",
    [0, 0.25, 0.5, 0.75, 1],
)
def test_validate_score_accepts_valid_scores(score):
    result = validate_score(score, "faithfulness")

    assert result == pytest.approx(float(score))


@pytest.mark.parametrize(
    "score",
    [-0.1, 1.1, "0.5", None, True],
)
def test_validate_score_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        validate_score(score, "faithfulness")


def test_normalize_claims_accepts_supported_status():
    claims = [
        {
            "claim": "Machine Learning learns from data.",
            "status": "SUPPORTED",
            "evidence": "Learns patterns from data.",
            "explanation": "The context supports this claim.",
        }
    ]

    result = normalize_claims(claims)

    assert len(result) == 1
    assert result[0]["status"] == "SUPPORTED"
    assert result[0]["claim"] == "Machine Learning learns from data."


def test_normalize_claims_handles_unknown_status():
    claims = [
        {
            "claim": "Unknown claim",
            "status": "INVALID_STATUS",
        }
    ]

    result = normalize_claims(claims)

    assert result[0]["status"] == "UNCERTAIN"


def test_normalize_claims_handles_invalid_input():
    assert normalize_claims(None) == []
    assert normalize_claims("not a list") == []


def test_empty_context_returns_insufficient_context_without_api_call(
    monkeypatch,
):
    def fail_if_called():
        raise AssertionError(
            "Groq client should not be initialized for empty context."
        )

    monkeypatch.setattr(
        "evaluation.semantic_evaluator.get_groq_client",
        fail_if_called,
    )

    result = evaluate_semantic_quality(
        question="Machine Learning kya hai?",
        generated_answer="Machine Learning data se patterns seekhta hai.",
        ground_truth_answer="Machine Learning data se patterns seekhta hai.",
        retrieved_chunks=[],
    )

    assert result["status"] == "INSUFFICIENT_CONTEXT"
    assert result["metrics"]["faithfulness"] is None
    assert result["metrics"]["answer_relevance"] is None
    assert result["metrics"]["answer_correctness"] is None
    assert result["metrics"]["hallucination_risk"] is None
    assert result["claims"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "question": "",
            "generated_answer": "An answer",
            "ground_truth_answer": "Reference answer",
            "retrieved_chunks": [],
        },
        {
            "question": "A question",
            "generated_answer": "",
            "ground_truth_answer": "Reference answer",
            "retrieved_chunks": [],
        },
        {
            "question": "A question",
            "generated_answer": "An answer",
            "ground_truth_answer": "",
            "retrieved_chunks": [],
        },
    ],
)
def test_empty_required_input_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        evaluate_semantic_quality(**kwargs)


def test_non_list_retrieved_chunks_raises_type_error():
    with pytest.raises(TypeError):
        evaluate_semantic_quality(
            question="Machine Learning kya hai?",
            generated_answer="Machine Learning learns from data.",
            ground_truth_answer="Machine Learning learns from data.",
            retrieved_chunks="invalid",
        )