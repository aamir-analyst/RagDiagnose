import pytest

from evaluation.retriever_evaluator import evaluate_retrieval


@pytest.fixture
def retrieved_chunks():
    return [
        {"metadata": {"chunk_id": "chunk_12"}},
        {"metadata": {"chunk_id": "chunk_8"}},
        {"metadata": {"chunk_id": "chunk_15"}},
        {"metadata": {"chunk_id": "chunk_4"}},
        {"metadata": {"chunk_id": "chunk_20"}},
    ]


@pytest.fixture
def relevant_chunk_ids():
    return ["chunk_12", "chunk_15"]


def test_evaluate_retrieval_returns_expected_structure(
    retrieved_chunks,
    relevant_chunk_ids,
):
    result = evaluate_retrieval(
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        k_values=[1, 3, 5],
    )

    assert "retrieved_chunk_ids" in result
    assert "relevant_chunk_ids" in result
    assert "metrics" in result

    assert result["retrieved_chunk_ids"] == [
        "chunk_12",
        "chunk_8",
        "chunk_15",
        "chunk_4",
        "chunk_20",
    ]

    assert result["relevant_chunk_ids"] == relevant_chunk_ids


def test_retrieval_metrics_are_present(
    retrieved_chunks,
    relevant_chunk_ids,
):
    result = evaluate_retrieval(
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        k_values=[1, 3, 5],
    )

    metrics = result["metrics"]

    expected_metrics = [
        "recall@1",
        "precision@1",
        "hit_rate@1",
        "recall@3",
        "precision@3",
        "hit_rate@3",
        "recall@5",
        "precision@5",
        "hit_rate@5",
        "mrr",
    ]

    for metric_name in expected_metrics:
        assert metric_name in metrics, (
            f"Missing retrieval metric: {metric_name}"
        )


def test_recall_and_precision_at_k(
    retrieved_chunks,
    relevant_chunk_ids,
):
    result = evaluate_retrieval(
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        k_values=[1, 3, 5],
    )

    metrics = result["metrics"]

    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["precision@1"] == pytest.approx(1.0)

    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["precision@3"] == pytest.approx(
        2 / 3,
        abs=0.0001,
    )

    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["precision@5"] == pytest.approx(
        2 / 5,
        abs=0.0001,
    )


def test_hit_rate_and_mrr(
    retrieved_chunks,
    relevant_chunk_ids,
):
    result = evaluate_retrieval(
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        k_values=[1, 3, 5],
    )

    metrics = result["metrics"]

    assert metrics["hit_rate@1"] == pytest.approx(1.0)
    assert metrics["hit_rate@3"] == pytest.approx(1.0)
    assert metrics["hit_rate@5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_no_relevant_chunks_retrieved():
    chunks = [
        {"metadata": {"chunk_id": "chunk_8"}},
        {"metadata": {"chunk_id": "chunk_4"}},
    ]

    result = evaluate_retrieval(
        retrieved_chunks=chunks,
        relevant_chunk_ids=["chunk_12"],
        k_values=[1, 2],
    )

    metrics = result["metrics"]

    assert metrics["recall@1"] == pytest.approx(0.0)
    assert metrics["precision@1"] == pytest.approx(0.0)
    assert metrics["hit_rate@1"] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(0.0)


def test_empty_retrieval():
    result = evaluate_retrieval(
        retrieved_chunks=[],
        relevant_chunk_ids=["chunk_12"],
        k_values=[1, 3],
    )

    assert result["retrieved_chunk_ids"] == []

    metrics = result["metrics"]

    assert metrics["recall@1"] == pytest.approx(0.0)
    assert metrics["precision@1"] == pytest.approx(0.0)
    assert metrics["hit_rate@1"] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(0.0)