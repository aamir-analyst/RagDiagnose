import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.test_dataset import TEST_DATASET  # fallback dataset
from evaluation.retriever_evaluator import evaluate_retrieval
from evaluation.generator_evaluator import evaluate_generator
from evaluation.semantic_evaluator import evaluate_semantic_quality
from evaluation.root_cause_analyzer import analyze_root_causes

from rag.retriever import retrieve_documents
from rag.generator import generate_answer


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = int(os.getenv("RAG_EVAL_TOP_K", "10"))
K_VALUES = [1, 3, 5, 10]

# Semantic evaluation makes an additional LLM request for every test.
# Keep it opt-in to reduce Groq TPM usage; enable with RAG_EVAL_SEMANTIC=true.
SEMANTIC_EVALUATION_ENABLED = os.getenv(
    "RAG_EVAL_SEMANTIC", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# Pause between test cases to reduce Groq TPM/rate-limit errors.
# Override with RAG_EVAL_DELAY_SECONDS if needed.
EVALUATION_DELAY_SECONDS = max(
    0.0, float(os.getenv("RAG_EVAL_DELAY_SECONDS", "10"))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)




def load_evaluation_dataset() -> tuple[list[dict[str, Any]], str]:
    """
    Load the synthetic dataset when available; otherwise use TEST_DATASET.
    Expected synthetic format: {"test_dataset": [...]}.
    """
    synthetic_path = RESULTS_DIR / "synthetic_dataset.json"

    if synthetic_path.exists():
        try:
            with synthetic_path.open("r", encoding="utf-8-sig") as file:
                payload = json.load(file)

            cases = payload.get("test_dataset") if isinstance(payload, dict) else None

            if isinstance(cases, list) and cases:
                logger.info(
                    "Loaded %d cases from synthetic dataset: %s",
                    len(cases),
                    synthetic_path,
                )
                return cases, str(synthetic_path)

            logger.warning(
                "Synthetic dataset exists but has no non-empty test_dataset; "
                "falling back to evaluation.test_dataset.TEST_DATASET."
            )
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            logger.warning(
                "Could not load synthetic dataset (%s); falling back to TEST_DATASET.",
                error,
            )

    return TEST_DATASET, "evaluation.test_dataset.TEST_DATASET"


# =========================================================
# DATASET VALIDATION
# =========================================================

def validate_test_case(test_case: dict[str, Any]) -> None:
    """Validate one evaluation dataset entry."""

    required_fields = [
        "test_id",
        "question",
        "ground_truth_answer",
        "relevant_chunk_ids",
    ]

    for field in required_fields:
        if field not in test_case:
            raise ValueError(f"Missing required field: {field}")

    for field in [
        "test_id",
        "question",
        "ground_truth_answer",
    ]:
        value = test_case[field]

        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string.")

        if not value.strip():
            raise ValueError(f"{field} cannot be empty.")

    relevant_ids = test_case["relevant_chunk_ids"]

    if not isinstance(relevant_ids, list) or not relevant_ids:
        raise ValueError(
            "relevant_chunk_ids must be a non-empty list."
        )

    if any(
        not isinstance(chunk_id, str) or not chunk_id.strip()
        for chunk_id in relevant_ids
    ):
        raise ValueError(
            "Every relevant chunk ID must be a non-empty string."
        )


# =========================================================
# RUN ONE EVALUATION
# =========================================================

def run_single_evaluation(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate retrieval first, then generator and optional semantic judge.

    Retrieval metrics are retained even when answer generation fails. A failed
    API call is explicitly marked as failed and is never represented as a score.
    """
    validate_test_case(test_case)

    test_id = test_case["test_id"]
    question = test_case["question"]
    ground_truth = test_case["ground_truth_answer"]
    relevant_chunk_ids = test_case["relevant_chunk_ids"]

    logger.info("Starting evaluation: %s", test_id)
    test_start_time = time.perf_counter()
    latency: dict[str, float] = {}
    retrieved_chunks: list[dict[str, Any]] = []

    # 1. Retrieval: failures here prevent reliable retrieval scoring.
    retrieval_start_time = time.perf_counter()
    retrieved_chunks = retrieve_documents(query=question, top_k=TOP_K)
    latency["retrieval_latency_ms"] = round(
        (time.perf_counter() - retrieval_start_time) * 1000, 2
    )
    if not isinstance(retrieved_chunks, list):
        raise TypeError("retrieve_documents() must return a list.")

    # 2. Retrieval evaluation is independent from generation.
    retrieval_result = evaluate_retrieval(
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        k_values=K_VALUES,
    )

    # These defaults allow the report and diagnostics to remain structured if
    # the generator API fails or returns an empty response.
    generation_result: dict[str, Any] = {"answer": "", "model": None}
    generated_answer = ""
    generation_status = "FAILED"
    generation_error: str | None = None
    generator_result: dict[str, Any] = {
        "question": question,
        "generated_answer": "",
        "ground_truth_answer": ground_truth,
        "metrics": {
            "answer_correctness": {"score": None, "metric": "token_overlap_f1"},
            "context_support": {"score": None, "metric": "answer_context_token_overlap"},
            "answer_completeness": {"score": None, "metric": "reference_token_coverage"},
            "citation_validation": {
                "citation_count": 0,
                "valid_citation_count": 0,
                "invalid_citation_count": 0,
                "citation_validity_score": None,
                "citations": [],
            },
            "claim_level_faithfulness": {
                "claim_count": 0,
                "average_claim_context_overlap": None,
                "claims": [],
            },
        },
        "status": "NOT_RUN",
        "error": None,
    }

    # 3. Generation failure is captured locally so retrieval results survive.
    generation_start_time = time.perf_counter()
    try:
        candidate = generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
        if not isinstance(candidate, dict):
            raise TypeError("generate_answer() must return a dictionary.")

        candidate_answer = candidate.get("answer", "")
        if not isinstance(candidate_answer, str) or not candidate_answer.strip():
            raise ValueError(
                "Generator returned an empty answer. This can happen when the "
                "model exhausts its token budget on reasoning before producing text."
            )

        generation_result = candidate
        generated_answer = candidate_answer.strip()
        generation_status = "COMPLETED"
        generator_result = evaluate_generator(
            question=question,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth,
            retrieved_chunks=retrieved_chunks,
        )
        generator_result["status"] = "COMPLETED"
    except Exception as error:
        generation_error = f"{type(error).__name__}: {error}"
        generator_result["status"] = "FAILED"
        generator_result["error"] = generation_error
        logger.exception("Answer generation/evaluation failed for %s.", test_id)
    finally:
        latency["generation_latency_ms"] = round(
            (time.perf_counter() - generation_start_time) * 1000, 2
        )

    # 4. Semantic evaluation is opt-in because it makes another LLM request.
    semantic_start_time = time.perf_counter()
    if generation_status != "COMPLETED":
        semantic_result = {
            "status": "SKIPPED_GENERATION_FAILED",
            "judge_model": None,
            "metrics": {
                "faithfulness": None,
                "answer_relevance": None,
                "answer_correctness": None,
                "hallucination_risk": None,
            },
            "claims": [],
            "explanation": "Semantic evaluation skipped because no usable answer was generated.",
            "limitations": ["No semantic scores were produced."],
        }
    elif not SEMANTIC_EVALUATION_ENABLED:
        semantic_result = {
            "status": "SKIPPED_BY_CONFIGURATION",
            "judge_model": None,
            "metrics": {
                "faithfulness": None,
                "answer_relevance": None,
                "answer_correctness": None,
                "hallucination_risk": None,
            },
            "claims": [],
            "explanation": (
                "Semantic LLM-judge evaluation is disabled to reduce API usage. "
                "Set RAG_EVAL_SEMANTIC=true to enable it."
            ),
            "limitations": ["Semantic metrics were not evaluated for this test."],
        }
    else:
        try:
            semantic_result = evaluate_semantic_quality(
                question=question,
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as error:
            logger.exception("Semantic evaluation failed for %s.", test_id)
            semantic_result = {
                "status": "ERROR",
                "judge_model": None,
                "metrics": {
                    "faithfulness": None,
                    "answer_relevance": None,
                    "answer_correctness": None,
                    "hallucination_risk": None,
                },
                "claims": [],
                "explanation": f"{type(error).__name__}: {error}",
                "limitations": ["Semantic evaluation failed; no semantic scores were produced."],
            }
    latency["semantic_evaluation_latency_ms"] = round(
        (time.perf_counter() - semantic_start_time) * 1000, 2
    )

    # 5. Root-cause analysis always receives retrieval metrics, plus any
    # available generator/semantic results.
    diagnostic_input = {
        "test_id": test_id,
        "retrieval_evaluation": retrieval_result,
        "generator_evaluation": generator_result,
        "semantic_evaluation": semantic_result,
    }
    try:
        root_cause_result = analyze_root_causes(diagnostic_input)
        if not isinstance(root_cause_result, dict):
            raise TypeError("analyze_root_causes() must return a dictionary.")
        root_cause_result.setdefault(
            "overall_status", root_cause_result.get("status", "UNKNOWN")
        )
        root_cause_result.setdefault("status", root_cause_result["overall_status"])
        root_cause_result.setdefault("test_id", test_id)
        root_cause_result.setdefault("root_causes", [])
        root_cause_result.setdefault("evidence", {})
        root_cause_result.setdefault("recommendations", [])
    except Exception as error:
        logger.exception("Root-cause analysis failed for %s.", test_id)
        root_cause_result = {
            "test_id": test_id,
            "status": "DIAGNOSTIC_ERROR",
            "overall_status": "DIAGNOSTIC_ERROR",
            "root_causes": [],
            "evidence": {},
            "recommendations": ["Review the root-cause analyzer error and rerun diagnostics."],
            "diagnostic_error": f"{type(error).__name__}: {error}",
        }

    latency["total_test_latency_ms"] = round(
        (time.perf_counter() - test_start_time) * 1000, 2
    )

    return {
        "test_id": test_id,
        "question": question,
        "model": generation_result.get("model"),
        "generation_status": generation_status,
        "generation_error": generation_error,
        "generated_answer": generated_answer,
        "ground_truth_answer": ground_truth,
        "retrieved_chunks_count": len(retrieved_chunks),
        "retrieved_chunks": [
            {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "page_number": (chunk.get("metadata") or {}).get(
                    "page_number", chunk.get("page_number")
                ),
                "text": str(chunk.get("text", "")),
                "score": chunk.get("score"),
            }
            for chunk in retrieved_chunks
            if isinstance(chunk, dict)
        ],
        "latency": latency,
        "retrieval_evaluation": retrieval_result,
        "generator_evaluation": generator_result,
        "semantic_evaluation": semantic_result,
        "root_cause_analysis": root_cause_result,
    }


# =========================================================
# COMMON HELPERS
# =========================================================

def mean_or_none(
    values: list[float],
) -> float | None:
    """Return the rounded mean, or None if no values exist."""

    if not values:
        return None

    return round(sum(values) / len(values), 4)


def is_number(value: Any) -> bool:
    """Return True for numeric values, excluding booleans."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


# =========================================================
# AGGREGATE RETRIEVAL METRICS
# =========================================================

def aggregate_retrieval_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate retrieval metrics across successful tests."""

    metric_names = [
        "recall@1",
        "precision@1",
        "hit_rate@1",
        "ndcg@1",
        "recall@3",
        "precision@3",
        "hit_rate@3",
        "ndcg@3",
        "recall@5",
        "precision@5",
        "hit_rate@5",
        "ndcg@5",
        "recall@10",
        "precision@10",
        "hit_rate@10",
        "ndcg@10",
        "mrr",
    ]

    aggregate: dict[str, Any] = {}

    for metric_name in metric_names:
        values = []

        for result in results:
            metrics = result.get(
                "retrieval_evaluation", {}
            ).get("metrics", {})

            value = metrics.get(metric_name)

            if is_number(value):
                values.append(float(value))

        aggregate[metric_name] = mean_or_none(values)

    return aggregate


# =========================================================
# AGGREGATE LATENCY METRICS
# =========================================================

def aggregate_latency_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate timing measurements across successful test cases."""

    latency_names = [
        "retrieval_latency_ms",
        "generation_latency_ms",
        "generator_evaluation_latency_ms",
        "semantic_evaluation_latency_ms",
        "total_test_latency_ms",
    ]

    summary: dict[str, Any] = {}

    for name in latency_names:
        values: list[float] = []

        for result in results:
            latency = result.get("latency", {})
            value = latency.get(name) if isinstance(latency, dict) else None

            if is_number(value):
                values.append(float(value))

        summary[name] = mean_or_none(values)

    return summary


# =========================================================
# AGGREGATE GENERATOR METRICS
# =========================================================

def aggregate_generator_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate generator scores and citation statistics.

    Lexical scores are not treated as proof of factual correctness.
    """

    score_names = [
        "answer_correctness",
        "context_support",
        "answer_completeness",
        "citation_validity_score",
        "average_claim_context_overlap",
    ]

    count_names = [
        "citation_count",
        "valid_citation_count",
        "invalid_citation_count",
        "claim_count",
    ]

    score_values: dict[str, list[float]] = {
        name: [] for name in score_names
    }

    count_values: dict[str, list[float]] = {
        name: [] for name in count_names
    }

    for result in results:
        metrics = result.get(
            "generator_evaluation", {}
        ).get("metrics", {})

        # Lexical generator metrics
        for name in [
            "answer_correctness",
            "context_support",
            "answer_completeness",
        ]:
            value = metrics.get(name, {}).get("score")

            if is_number(value):
                score_values[name].append(float(value))

        # Citation validation
        citation = metrics.get(
            "citation_validation", {}
        )

        for name in [
            "citation_count",
            "valid_citation_count",
            "invalid_citation_count",
        ]:
            value = citation.get(name)

            if is_number(value):
                count_values[name].append(float(value))

        value = citation.get("citation_validity_score")

        if is_number(value):
            score_values[
                "citation_validity_score"
            ].append(float(value))

        # Claim-level lexical overlap
        claim_eval = metrics.get(
            "claim_level_faithfulness", {}
        )

        value = claim_eval.get("claim_count")

        if is_number(value):
            count_values["claim_count"].append(float(value))

        value = claim_eval.get(
            "average_claim_context_overlap"
        )

        if is_number(value):
            score_values[
                "average_claim_context_overlap"
            ].append(float(value))

    summary: dict[str, Any] = {}

    for name, values in score_values.items():
        summary[name] = mean_or_none(values)

    for name, values in count_values.items():
        summary[name] = (
            int(sum(values)) if values else 0
        )

    return summary


# =========================================================
# AGGREGATE SEMANTIC METRICS
# =========================================================

def aggregate_semantic_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate completed semantic evaluations."""

    score_names = [
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
        "hallucination_risk",
    ]

    score_values: dict[str, list[float]] = {
        name: [] for name in score_names
    }

    status_counts: dict[str, int] = {}
    claim_status_counts: dict[str, int] = {}
    evaluated_tests = 0

    for result in results:
        semantic = result.get(
            "semantic_evaluation", {}
        )

        status = semantic.get("status", "UNKNOWN")

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        if status != "COMPLETED":
            continue

        evaluated_tests += 1

        metrics = semantic.get("metrics", {})

        for name in score_names:
            value = metrics.get(name)

            if is_number(value):
                score_values[name].append(float(value))

        for claim in semantic.get("claims", []):
            claim_status = str(
                claim.get("status", "UNKNOWN")
            ).upper()

            claim_status_counts[claim_status] = (
                claim_status_counts.get(claim_status, 0) + 1
            )

    return {
        "evaluated_tests": evaluated_tests,
        "status_counts": status_counts,
        "scores": {
            name: mean_or_none(values)
            for name, values in score_values.items()
        },
        "claim_status_counts": claim_status_counts,
    }


# =========================================================
# AGGREGATE ROOT-CAUSE RESULTS
# =========================================================

def aggregate_root_cause_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate diagnosis status, causes, severity and recommendations."""

    status_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    total_recommendations = 0

    for result in results:
        diagnosis = result.get(
            "root_cause_analysis", {}
        )

        if not isinstance(diagnosis, dict):
            diagnosis = {}

        status = diagnosis.get(
            "overall_status",
            diagnosis.get("status", "UNKNOWN"),
        )

        status = str(status).upper()

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        causes = diagnosis.get("root_causes", [])

        if not isinstance(causes, list):
            causes = []

        for cause in causes:
            if not isinstance(cause, dict):
                continue

            category = str(
                cause.get("category", "UNKNOWN")
            ).upper()

            severity = str(
                cause.get("severity", "UNKNOWN")
            ).upper()

            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

            severity_counts[severity] = (
                severity_counts.get(severity, 0) + 1
            )

        recommendations = diagnosis.get(
            "recommendations", []
        )

        if isinstance(recommendations, list):
            total_recommendations += len(recommendations)

    return {
        "status_counts": status_counts,
        "root_cause_category_counts": category_counts,
        "severity_counts": severity_counts,
        "total_recommendations": total_recommendations,
    }


# =========================================================
# PRINT ROOT-CAUSE ANALYSIS
# =========================================================

def print_root_cause_analysis(
    diagnosis: dict[str, Any],
) -> None:
    """Print diagnosis, possible causes and recommendations."""

    if not isinstance(diagnosis, dict):
        diagnosis = {}

    status = diagnosis.get(
        "overall_status",
        diagnosis.get("status", "UNKNOWN"),
    )

    print("\nRoot-Cause Analysis (rule-based)")
    print(f"  overall status: {status}")

    causes = diagnosis.get("root_causes", [])

    if causes:
        print("  possible root causes:")

        for cause in causes:
            if not isinstance(cause, dict):
                continue

            print(
                f"    [{cause.get('severity', 'UNKNOWN')}] "
                f"{cause.get('category', 'UNKNOWN')}: "
                f"{cause.get('evidence', '')}"
            )
    else:
        print(
            "  possible root causes: "
            "none detected by configured rules"
        )

    recommendations = diagnosis.get(
        "recommendations", []
    )

    if recommendations:
        print("  recommendations:")

        for recommendation in recommendations:
            print(f"    - {recommendation}")

    diagnostic_error = diagnosis.get("diagnostic_error")

    if diagnostic_error:
        print(f"  diagnostic error: {diagnostic_error}")


# =========================================================
# PRINT SEMANTIC METRICS
# =========================================================

def print_semantic_metrics(
    semantic_evaluation: dict[str, Any],
) -> None:
    """Print LLM-judge semantic evaluation results."""

    if not isinstance(semantic_evaluation, dict):
        semantic_evaluation = {}

    print("\nSemantic evaluation (LLM judge):")

    print(
        f"  status: {semantic_evaluation.get('status')}"
    )

    print(
        f"  judge model: {semantic_evaluation.get('judge_model')}"
    )

    metrics = semantic_evaluation.get("metrics", {})

    for name in [
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
        "hallucination_risk",
    ]:
        print(f"  {name}: {metrics.get(name)}")

    explanation = semantic_evaluation.get("explanation")

    if explanation:
        print(f"  explanation: {explanation}")

    claims = semantic_evaluation.get("claims", [])

    if claims:
        print("  claims:")

        for claim in claims:
            print(
                f"    [{claim.get('status', 'UNKNOWN')}] "
                f"{claim.get('claim', '')}"
            )

            if claim.get("evidence"):
                print(
                    f"      evidence: {claim.get('evidence')}"
                )


# =========================================================
# PRINT GENERATOR METRICS
# =========================================================

def print_generator_metrics(
    generator_evaluation: dict[str, Any],
) -> None:
    """Print lexical scores, citations and claim-level metrics."""

    metrics = generator_evaluation.get("metrics", {})

    print("\nGenerator metrics:")

    for name in [
        "answer_correctness",
        "context_support",
        "answer_completeness",
    ]:
        details = metrics.get(name, {})

        print(
            f"  {name}: {details.get('score')}"
        )

    # Citation validation
    citation = metrics.get(
        "citation_validation", {}
    )

    print("\n  citation_validation:")
    print(
        f"    citations found: {citation.get('citation_count')}"
    )
    print(
        f"    valid references: "
        f"{citation.get('valid_citation_count')}"
    )
    print(
        f"    invalid references: "
        f"{citation.get('invalid_citation_count')}"
    )
    print(
        f"    validity score: "
        f"{citation.get('citation_validity_score')}"
    )

    for item in citation.get("citations", []):
        print(
            f"    citation: {item.get('citation_text')}"
        )
        print(
            f"    source reference found: "
            f"{item.get('source_reference_found')}"
        )

    # Claim-level lexical heuristic
    claim_eval = metrics.get(
        "claim_level_faithfulness", {}
    )

    print("\n  claim_level_faithfulness:")
    print(
        f"    claims found: {claim_eval.get('claim_count')}"
    )
    print(
        f"    average context overlap: "
        f"{claim_eval.get('average_claim_context_overlap')}"
    )

    for claim in claim_eval.get("claims", []):
        print(f"    claim: {claim.get('claim')}")
        print(
            f"    token overlap: "
            f"{claim.get('context_token_overlap')}"
        )
        print(
            f"    status: {claim.get('support_status')}"
        )


# =========================================================
# RUN COMPLETE EVALUATION SUITE
# =========================================================

def run_evaluation_suite() -> dict[str, Any]:
    """Run all dataset cases and save a JSON evaluation report."""

    dataset, dataset_source = load_evaluation_dataset()

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "Evaluation dataset must be a non-empty list."
        )

    test_ids = set()

    for test_case in dataset:
        if not isinstance(test_case, dict):
            raise TypeError(
                "Every dataset item must be a dictionary."
            )

        validate_test_case(test_case)

        test_id = test_case["test_id"]

        if test_id in test_ids:
            raise ValueError(
                f"Duplicate test_id found: {test_id}"
            )

        test_ids.add(test_id)

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    logger.info(
        "Starting evaluation run %s with %d test cases.",
        run_id,
        len(dataset),
    )

    # -----------------------------------------------------
    # Run all test cases
    # -----------------------------------------------------

    for index, test_case in enumerate(dataset):
        try:
            result = run_single_evaluation(test_case)
            results.append(result)
            if result.get("generation_status") != "COMPLETED":
                errors.append({
                    "test_id": result.get("test_id", "UNKNOWN"),
                    "question": result.get("question", ""),
                    "error_type": "GenerationFailed",
                    "error": result.get("generation_error") or "No usable answer generated.",
                    "retrieval_metrics_preserved": True,
                })
        except Exception as error:
            logger.exception(
                "Test failed before a complete result could be produced: %s",
                test_case.get("test_id", "UNKNOWN"),
            )
            errors.append({
                "test_id": test_case.get("test_id", "UNKNOWN"),
                "question": test_case.get("question", ""),
                "error_type": type(error).__name__,
                "error": str(error),
                "retrieval_metrics_preserved": False,
            })

        if index < len(dataset) - 1 and EVALUATION_DELAY_SECONDS > 0:
            logger.info("Waiting %.1f seconds before next test case.", EVALUATION_DELAY_SECONDS)
            time.sleep(EVALUATION_DELAY_SECONDS)

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    retrieval_summary = aggregate_retrieval_metrics(results)
    latency_summary = aggregate_latency_metrics(results)
    generator_summary = aggregate_generator_metrics(results)
    semantic_summary = aggregate_semantic_metrics(results)

    root_cause_summary = aggregate_root_cause_results(
        results
    )

    # -----------------------------------------------------
    # Build report
    # -----------------------------------------------------

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_source": dataset_source,
        "total_tests": len(dataset),
        "completed_pipeline_tests": len(results),
        "successful_tests": sum(
            1 for item in results if item.get("generation_status") == "COMPLETED"
        ),
        "failed_tests": len(errors),
        "semantic_evaluation_enabled": SEMANTIC_EVALUATION_ENABLED,
        "evaluation_delay_seconds": EVALUATION_DELAY_SECONDS,
        "aggregate_retrieval_metrics": retrieval_summary,
        "aggregate_latency_metrics": latency_summary,
        "aggregate_generator_metrics": generator_summary,
        "aggregate_semantic_metrics": semantic_summary,
        "aggregate_root_cause_analysis": root_cause_summary,
        "results": results,
        "errors": errors,
        "evaluation_notes": [
            (
                "Retrieval metrics are evaluated before generation and retained when generation fails."
            ),
            (
                "Semantic LLM-judge evaluation is disabled by default to reduce API token usage; "
                "enable it with RAG_EVAL_SEMANTIC=true."
            ),
            (
                "A non-empty model response is required for generation status COMPLETED. Empty responses are failures, not scores."
            ),
            (
                "A successful test means the evaluation pipeline completed; "
                "it does not mean the answer is factually correct."
            ),
            (
                "Correctness, context support, completeness and claim overlap "
                "are lexical or heuristic metrics, not verified semantic scores."
            ),
            (
                "Citation validation checks whether a cited document/page "
                "matches retrieved metadata. It does not prove claim support."
            ),
            (
                "Claim-level status remains NEEDS_SEMANTIC_VERIFICATION "
                "unless a separate semantic verification method is implemented."
            ),
            (
                "Retrieval scores depend on manually labelled relevant "
                "chunk IDs in TEST_DATASET."
            ),
            (
                "Semantic scores are estimates from an LLM judge, not objective "
                "proof of correctness. Judge errors and model variability are possible."
            ),
            (
                "Semantic evaluation failures are recorded per test and do not "
                "invalidate retrieval or lexical generator metrics."
            ),
            (
                "Root-cause analysis uses configurable rules and evaluation "
                "evidence; it identifies possible causes rather than proving causality."
            ),
            (
                "Latency values are wall-clock timings measured with perf_counter; "
                "they vary with network conditions, model load, and hardware."
            ),
        ],
    }

    # -----------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------

    report_path = (
        RESULTS_DIR / f"evaluation_{run_id}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("RAGDIAGNOSE - AUTOMATED EVALUATION REPORT")
    print("=" * 65)

    print(f"Run ID          : {run_id}")
    print(f"Total tests     : {report['total_tests']}")
    print(f"Pipeline results: {report['completed_pipeline_tests']}")
    print(f"Generation OK   : {report['successful_tests']}")
    print(f"Failed tests    : {report['failed_tests']}")
    print(f"Semantic judge  : {'ENABLED' if SEMANTIC_EVALUATION_ENABLED else 'DISABLED'}")

    print("\nAggregate Retrieval Metrics")
    print("-" * 40)

    for metric, value in retrieval_summary.items():
        print(f"{metric:20}: {value}")

    print("\nAggregate Latency Metrics (milliseconds)")
    print("-" * 40)

    for metric, value in latency_summary.items():
        print(f"{metric:35}: {value}")

    print("\nAggregate Generator Metrics")
    print("-" * 40)

    for metric, value in generator_summary.items():
        print(f"{metric:35}: {value}")

    print("\nAggregate Semantic Metrics (LLM judge)")
    print("-" * 40)

    print(
        f"Evaluated tests: {semantic_summary['evaluated_tests']}"
    )
    print(
        f"Status counts  : {semantic_summary['status_counts']}"
    )

    for metric, value in semantic_summary["scores"].items():
        print(f"{metric:25}: {value}")

    print(
        f"Claim statuses : "
        f"{semantic_summary['claim_status_counts']}"
    )

    print("\nAggregate Root-Cause Analysis")
    print("-" * 40)

    print(
        f"Status counts          : "
        f"{root_cause_summary['status_counts']}"
    )
    print(
        f"Root-cause categories  : "
        f"{root_cause_summary['root_cause_category_counts']}"
    )
    print(
        f"Severity counts        : "
        f"{root_cause_summary['severity_counts']}"
    )
    print(
        f"Total recommendations : "
        f"{root_cause_summary['total_recommendations']}"
    )

    # -----------------------------------------------------
    # Print individual test results
    # -----------------------------------------------------

    for result in results:
        print("\n" + "-" * 65)

        print(f"Test ID : {result['test_id']}")
        print(f"Question: {result['question']}")
        print(f"Model   : {result['model']}")
        print(f"Answer  : {result['generated_answer']}")

        print("\nLatency metrics (milliseconds):")
        for metric, value in result.get("latency", {}).items():
            print(f"  {metric}: {value}")

        print("\nRetrieval metrics:")

        metrics = result.get(
            "retrieval_evaluation", {}
        ).get("metrics", {})

        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        print_generator_metrics(
            result.get("generator_evaluation", {})
        )

        print_semantic_metrics(
            result.get("semantic_evaluation", {})
        )

        print_root_cause_analysis(
            result.get("root_cause_analysis", {})
        )

    # -----------------------------------------------------
    # Print failed tests
    # -----------------------------------------------------

    if errors:
        print("\nFailed Tests")
        print("-" * 40)

        for error in errors:
            print(
                f"{error['test_id']} | "
                f"{error['error_type']}: {error['error']}"
            )

    print(f"\nJSON report saved to:\n{report_path}")
    print("=" * 65)

    return report


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_evaluation_suite()
