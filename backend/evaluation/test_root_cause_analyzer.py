import pytest

from evaluation.root_cause_analyzer import analyze_root_causes


def test_healthy_result_when_all_scores_are_good():
    result = analyze_root_causes({
        "test_id": "test_001",
        "retrieval_evaluation": {
            "metrics": {
                "recall@1": 1.0,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
            }
        },
        "generator_evaluation": {
            "metrics": {
                "answer_correctness": {"score": 0.9},
                "context_support": {"score": 0.9},
                "answer_completeness": {"score": 0.9},
                "citation_validation": {
                    "citation_validity_score": 1.0,
                    "citation_count": 2,
                    "invalid_citation_count": 0,
                },
            }
        },
        "semantic_evaluation": {
            "status": "COMPLETED",
            "metrics": {
                "faithfulness": 0.95,
                "answer_relevance": 0.9,
                "answer_correctness": 0.9,
                "hallucination_risk": 0.05,
            },
        },
    })

    assert result["test_id"] == "test_001"
    assert result["status"] == "HEALTHY"
    assert result["root_causes"] == []


def test_low_recall_detects_retrieval_failure():
    result = analyze_root_causes({
        "test_id": "test_002",
        "retrieval_evaluation": {
            "metrics": {
                "recall@1": 0.0,
                "recall@3": 0.5,
                "recall@5": 0.5,
                "mrr": 0.5,
            }
        },
    })

    categories = [
        cause["category"] for cause in result["root_causes"]
    ]

    assert "RETRIEVAL_FAILURE" in categories
    assert result["status"] == "NEEDS_ATTENTION"


def test_low_context_support_detected():
    result = analyze_root_causes({
        "generator_evaluation": {
            "metrics": {
                "context_support": {"score": 0.3}
            }
        }
    })

    categories = [
        cause["category"] for cause in result["root_causes"]
    ]

    assert "LOW_CONTEXT_SUPPORT" in categories


def test_high_hallucination_risk_detected():
    result = analyze_root_causes({
        "semantic_evaluation": {
            "status": "COMPLETED",
            "metrics": {
                "hallucination_risk": 0.9
            },
        }
    })

    categories = [
        cause["category"] for cause in result["root_causes"]
    ]

    assert "POSSIBLE_HALLUCINATION" in categories


def test_missing_citations_detected():
    result = analyze_root_causes({
        "generator_evaluation": {
            "metrics": {
                "citation_validation": {
                    "citation_count": 0,
                    "invalid_citation_count": 0,
                    "citation_validity_score": 1.0,
                }
            }
        }
    })

    categories = [
        cause["category"] for cause in result["root_causes"]
    ]

    assert "MISSING_CITATION" in categories


def test_invalid_citations_detected():
    result = analyze_root_causes({
        "generator_evaluation": {
            "metrics": {
                "citation_validation": {
                    "citation_count": 2,
                    "invalid_citation_count": 1,
                    "citation_validity_score": 0.5,
                }
            }
        }
    })

    categories = [
        cause["category"] for cause in result["root_causes"]
    ]

    assert "INVALID_CITATION" in categories


def test_missing_semantic_evaluation_is_incomplete():
    result = analyze_root_causes({
        "test_id": "test_007",
        "retrieval_evaluation": {
            "metrics": {
                "recall@1": 1.0,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
            }
        }
    })

    assert result["status"] == "INCOMPLETE_DIAGNOSIS"


def test_malformed_input_does_not_crash():
    result = analyze_root_causes({
        "retrieval_evaluation": None,
        "generator_evaluation": "invalid",
        "semantic_evaluation": [],
    })

    assert isinstance(result, dict)
    assert "status" in result
    assert "root_causes" in result
    assert "recommendations" in result


def test_result_contains_diagnostic_limitations():
    result = analyze_root_causes({})

    assert result["diagnostic_type"] == "rule_based"
    assert len(result["limitations"]) > 0