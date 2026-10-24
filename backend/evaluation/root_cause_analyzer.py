from typing import Any


def _get_nested(
    data: dict[str, Any],
    *keys: str,
) -> Any:
    """Safely retrieve a nested dictionary value."""

    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def _get_score(
    data: dict[str, Any],
    *keys: str,
) -> float | None:
    """Read a numeric score without accepting booleans."""

    value = _get_nested(data, *keys)

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    return None


def _add_cause(
    causes: list[dict[str, Any]],
    category: str,
    severity: str,
    evidence: str,
) -> None:
    """Add a diagnostic finding."""

    causes.append({
        "category": category,
        "severity": severity,
        "evidence": evidence,
    })


def analyze_root_causes(
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Identify possible RAG failure causes from evaluation evidence.

    This is a rule-based diagnostic system. It suggests possible
    causes; it does not prove the underlying cause.
    """

    test_id = evaluation_result.get("test_id", "UNKNOWN")

    retrieval = evaluation_result.get(
        "retrieval_evaluation", {}
    )
    generator = evaluation_result.get(
        "generator_evaluation", {}
    )
    semantic = evaluation_result.get(
        "semantic_evaluation", {}
    )

    # Defensive checks for malformed or missing inputs.
    if not isinstance(retrieval, dict):
        retrieval = {}

    if not isinstance(generator, dict):
        generator = {}

    if not isinstance(semantic, dict):
        semantic = {}

    retrieval_metrics = retrieval.get("metrics", {})
    generator_metrics = generator.get("metrics", {})
    semantic_metrics = semantic.get("metrics", {})

    if not isinstance(retrieval_metrics, dict):
        retrieval_metrics = {}

    if not isinstance(generator_metrics, dict):
        generator_metrics = {}

    if not isinstance(semantic_metrics, dict):
        semantic_metrics = {}

    # Retrieval scores
    recall_at_1 = _get_score(
        retrieval, "metrics", "recall@1"
    )
    recall_at_3 = _get_score(
        retrieval, "metrics", "recall@3"
    )
    recall_at_5 = _get_score(
        retrieval, "metrics", "recall@5"
    )
    mrr = _get_score(
        retrieval, "metrics", "mrr"
    )

    # Generator scores
    answer_correctness = _get_score(
        generator_metrics, "answer_correctness", "score"
    )
    context_support = _get_score(
        generator_metrics, "context_support", "score"
    )
    completeness = _get_score(
        generator_metrics, "answer_completeness", "score"
    )

    citation_validity = _get_score(
        generator_metrics,
        "citation_validation",
        "citation_validity_score",
    )
    citation_count = _get_score(
        generator_metrics,
        "citation_validation",
        "citation_count",
    )
    invalid_citation_count = _get_score(
        generator_metrics,
        "citation_validation",
        "invalid_citation_count",
    )

    # Semantic scores
    faithfulness = _get_score(
        semantic_metrics, "faithfulness"
    )
    answer_relevance = _get_score(
        semantic_metrics, "answer_relevance"
    )
    semantic_correctness = _get_score(
        semantic_metrics, "answer_correctness"
    )
    hallucination_risk = _get_score(
        semantic_metrics, "hallucination_risk"
    )

    semantic_status = str(
        semantic.get("status", "NOT_AVAILABLE")
    ).upper()

    root_causes: list[dict[str, Any]] = []
    recommendations: list[str] = []

    evidence = {
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "answer_correctness": answer_correctness,
        "context_support": context_support,
        "answer_completeness": completeness,
        "semantic_status": semantic_status,
        "semantic_faithfulness": faithfulness,
        "semantic_answer_relevance": answer_relevance,
        "semantic_answer_correctness": semantic_correctness,
        "hallucination_risk": hallucination_risk,
        "citation_validity": citation_validity,
        "citation_count": citation_count,
        "invalid_citation_count": invalid_citation_count,
    }

    # --------------------------------------------------
    # 1. RETRIEVAL DIAGNOSTICS
    # --------------------------------------------------

    if recall_at_3 is not None and recall_at_3 < 1.0:
        _add_cause(
            root_causes,
            "RETRIEVAL_FAILURE",
            "HIGH",
            (
                f"Recall@3 is {recall_at_3:.4f}. "
                "At least one expected relevant chunk was not "
                "retrieved in the top three results."
            ),
        )

        recommendations.append(
            "Inspect the missing relevant chunk IDs, chunking, "
            "embedding quality, query formulation and top-K settings."
        )

    elif recall_at_3 is None:
        recommendations.append(
            "Retrieval recall is unavailable. Verify that the "
            "evaluation contains retrieval metrics and labelled "
            "relevant chunk IDs."
        )

    # Relevant information was found in top 3, but not at rank 1.
    if (
        recall_at_3 is not None
        and recall_at_3 >= 1.0
        and recall_at_1 is not None
        and recall_at_1 < 1.0
    ):
        _add_cause(
            root_causes,
            "RETRIEVAL_RANKING_ISSUE",
            "MEDIUM",
            (
                f"Recall@1 is {recall_at_1:.4f}, while "
                f"Recall@3 is {recall_at_3:.4f}. "
                "Relevant information appears lower in the ranking."
            ),
        )

        recommendations.append(
            "Inspect the retrieved chunk ranking. Consider "
            "reranking or comparing retrieval configurations."
        )

    elif mrr is not None and mrr < 1.0:
        recommendations.append(
            "Review the rank of the first relevant chunk and "
            "consider evaluating a reranker."
        )

    # --------------------------------------------------
    # 2. GENERATOR / CONTEXT-USE DIAGNOSTICS
    # --------------------------------------------------

    if (
        context_support is not None
        and context_support < 0.6
    ):
        _add_cause(
            root_causes,
            "LOW_CONTEXT_SUPPORT",
            "HIGH",
            (
                f"Context-support score is {context_support:.4f}. "
                "The lexical metric indicates weak overlap between "
                "the answer and the retrieved context."
            ),
        )

        recommendations.append(
            "Inspect individual answer claims against the retrieved "
            "chunks. Check prompt instructions and context construction."
        )

    if (
        completeness is not None
        and completeness < 0.6
    ):
        _add_cause(
            root_causes,
            "POSSIBLY_INCOMPLETE_ANSWER",
            "MEDIUM",
            (
                f"Answer-completeness score is {completeness:.4f}. "
                "The lexical completeness heuristic is low."
            ),
        )

        recommendations.append(
            "Check whether the retrieved context contains all required "
            "information and whether the answer covers it."
        )

    if (
        answer_correctness is not None
        and answer_correctness < 0.6
    ):
        _add_cause(
            root_causes,
            "LOW_LEXICAL_ANSWER_CORRECTNESS",
            "MEDIUM",
            (
                f"Lexical answer-correctness score is "
                f"{answer_correctness:.4f}. This is a heuristic score, "
                "not an independent factual verification."
            ),
        )

        recommendations.append(
            "Compare the generated answer with the ground-truth answer "
            "and review the supporting retrieved evidence."
        )

    # --------------------------------------------------
    # 3. SEMANTIC / HALLUCINATION DIAGNOSTICS
    # --------------------------------------------------

    if semantic_status == "COMPLETED":

        if (
            hallucination_risk is not None
            and hallucination_risk >= 0.7
        ):
            _add_cause(
                root_causes,
                "POSSIBLE_HALLUCINATION",
                "HIGH",
                (
                    "The LLM judge reported hallucination risk "
                    f"{hallucination_risk:.4f}."
                ),
            )

            recommendations.append(
                "Review unsupported claims against retrieved text. "
                "Check whether the model introduced information not "
                "supported by the available context."
            )

        if (
            faithfulness is not None
            and faithfulness < 0.6
        ):
            _add_cause(
                root_causes,
                "LOW_SEMANTIC_FAITHFULNESS",
                "HIGH",
                (
                    "The LLM judge reported faithfulness "
                    f"{faithfulness:.4f}."
                ),
            )

            recommendations.append(
                "Inspect claim-level evidence and revise the prompt "
                "to keep answers grounded in retrieved context."
            )

        if (
            answer_relevance is not None
            and answer_relevance < 0.6
        ):
            _add_cause(
                root_causes,
                "LOW_ANSWER_RELEVANCE",
                "MEDIUM",
                (
                    "The LLM judge reported answer relevance "
                    f"{answer_relevance:.4f}."
                ),
            )

            recommendations.append(
                "Review the question and generated answer. Check "
                "whether retrieval supplied the information needed "
                "to address the actual question."
            )

    else:
        recommendations.append(
            "Semantic evaluation was not completed. Do not infer "
            "that the answer is faithful or hallucinated."
        )

    # --------------------------------------------------
    # 4. CITATION DIAGNOSTICS
    # --------------------------------------------------

    if citation_count is not None and citation_count == 0:
        _add_cause(
            root_causes,
            "MISSING_CITATION",
            "MEDIUM",
            "No recognizable citations were detected.",
        )

        recommendations.append(
            "Check the answer-generation prompt and citation parser."
        )

    if (
        invalid_citation_count is not None
        and invalid_citation_count > 0
    ):
        _add_cause(
            root_causes,
            "INVALID_CITATION",
            "HIGH",
            (
                f"{int(invalid_citation_count)} citation(s) did not "
                "match the available source metadata."
            ),
        )

        recommendations.append(
            "Verify source document names, page numbers and chunk "
            "metadata used by citation validation."
        )

    if (
        citation_validity is not None
        and citation_validity < 1.0
        and invalid_citation_count is None
    ):
        recommendations.append(
            "Review citation-validation details to determine "
            "which source references failed validation."
        )

    # --------------------------------------------------
    # 5. OVERALL STATUS
    # --------------------------------------------------

    has_high_severity = any(
        item["severity"] == "HIGH"
        for item in root_causes
    )

    has_medium_severity = any(
        item["severity"] == "MEDIUM"
        for item in root_causes
    )

    if has_high_severity:
        overall_status = "NEEDS_ATTENTION"
    elif has_medium_severity:
        overall_status = "WARNING"
    elif semantic_status != "COMPLETED":
        overall_status = "INCOMPLETE_DIAGNOSIS"
    else:
        overall_status = "HEALTHY"

    if not recommendations:
        recommendations.append(
            "No configured rule detected a major issue. Validate "
            "the result with a larger, manually reviewed test dataset."
        )

    # De-duplicate recommendations while preserving their order.
    recommendations = list(dict.fromkeys(recommendations))

    return {
        "test_id": test_id,

        # Both keys are provided for compatibility with code that
        # reads either "status" or "overall_status".
        "status": overall_status,
        "overall_status": overall_status,

        "root_causes": root_causes,
        "evidence": evidence,
        "recommendations": recommendations,
        "diagnostic_type": "rule_based",
        "limitations": [
            "Rules identify possible causes, not proven root causes.",
            "Semantic scores depend on an LLM judge and may be wrong.",
            "Lexical metrics do not independently prove factual correctness.",
            "Citation validity does not prove that a source supports every claim.",
            "Low retrieval metrics are meaningful only when relevant chunk "
            "labels are correct.",
            "A healthy status means no configured rule detected an issue; "
            "it does not guarantee that the answer is correct.",
        ],
    }