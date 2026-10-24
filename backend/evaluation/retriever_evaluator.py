
from typing import Any
import math


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Calculate the proportion of relevant chunks retrieved in top-k."""

    if not relevant_ids or k < 1:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return round(len(retrieved & relevant) / len(relevant), 4)


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Calculate the proportion of top-k results that are relevant."""

    if k < 1:
        return 0.0

    top_k_ids = retrieved_ids[:k]

    if not top_k_ids:
        return 0.0

    relevant = set(relevant_ids)

    relevant_count = sum(
        1 for chunk_id in top_k_ids
        if chunk_id in relevant
    )

    # Use the number of returned results, not requested k.
    return round(relevant_count / len(top_k_ids), 4)


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str]
) -> float:
    """Calculate reciprocal rank of the first relevant chunk."""

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return round(1 / rank, 4)

    return 0.0


def calculate_hit_rate(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Return 1 if at least one relevant chunk appears in top-k."""

    if not relevant_ids or k < 1:
        return 0.0

    relevant = set(relevant_ids)

    return float(
        any(chunk_id in relevant for chunk_id in retrieved_ids[:k])
    )


def calculate_dcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    Relevant chunks receive a binary relevance score of 1.
    """

    if k < 1:
        return 0.0

    relevant = set(relevant_ids)
    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1.0 if chunk_id in relevant else 0.0

        # Rank 1 receives full weight; lower ranks are discounted.
        dcg += relevance / math.log2(rank + 1)

    return dcg


def calculate_ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    NDCG ranges from 0 to 1:
    - 1 means ideal ranking.
    - 0 means no relevant chunks were retrieved.
    """

    if not relevant_ids or k < 1:
        return 0.0

    dcg = calculate_dcg_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    # Ideal ranking places all relevant chunks at the top.
    ideal_relevant_count = min(len(set(relevant_ids)), k)

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0.0:
        return 0.0

    return round(dcg / ideal_dcg, 4)


def evaluate_retrieval(
    retrieved_chunks: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    k_values: list[int] | None = None
) -> dict[str, Any]:
    """
    Evaluate retrieved chunks against manually labelled relevant IDs.

    Each retrieved chunk must contain:
        {"metadata": {"chunk_id": "..."}}

    Relevant IDs must use the same ID format.
    """

    if k_values is None:
        k_values = [1, 3, 5, 10]

    if any(k < 1 for k in k_values):
        raise ValueError("All k_values must be >= 1.")

    # Extract chunk IDs while preserving retrieval order.
    retrieved_ids = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") or {}
        chunk_id = metadata.get("chunk_id")

        if chunk_id is not None:
            retrieved_ids.append(str(chunk_id))

    # Remove duplicates while preserving ranking.
    retrieved_ids = list(dict.fromkeys(retrieved_ids))

    relevant_ids = list(dict.fromkeys(
        str(chunk_id)
        for chunk_id in relevant_chunk_ids
        if chunk_id is not None and str(chunk_id)
    ))

    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = calculate_recall_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )

        metrics[f"precision@{k}"] = calculate_precision_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )

        metrics[f"hit_rate@{k}"] = calculate_hit_rate(
            retrieved_ids,
            relevant_ids,
            k
        )

        metrics[f"ndcg@{k}"] = calculate_ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )

    metrics["mrr"] = calculate_mrr(
        retrieved_ids,
        relevant_ids
    )

    return {
        "retrieved_chunk_ids": retrieved_ids,
        "relevant_chunk_ids": relevant_ids,
        "retrieved_count": len(retrieved_ids),
        "relevant_count": len(relevant_ids),
        "metrics": metrics
    }
