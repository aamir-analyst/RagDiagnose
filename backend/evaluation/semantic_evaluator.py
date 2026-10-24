
"""
RAGDiagnose - Semantic Evaluator

Purpose:
1. Evaluate answer faithfulness against retrieved context.
2. Evaluate answer relevance to the question.
3. Compare the generated answer with the ground-truth answer.
4. Identify potentially unsupported claims.
5. Estimate hallucination risk.

Important:
These are LLM-as-a-Judge estimates, not guaranteed factual truth.
"""

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"


# --------------------------------------------------
# CLIENT INITIALIZATION
# --------------------------------------------------

def get_groq_client() -> Groq:
    """
    Create a Groq client using GROQ_API_KEY from .env.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to your backend .env file."
        )

    return Groq(api_key=api_key)


# --------------------------------------------------
# CONTEXT PREPARATION
# --------------------------------------------------

def prepare_context(
    retrieved_chunks: list[dict[str, Any]]
) -> str:
    """
    Convert retrieved chunks into readable context.

    Supports common chunk formats:
    - page_content
    - text
    - content

    Metadata may contain document name and page number.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        text = (
            chunk.get("page_content")
            or chunk.get("text")
            or chunk.get("content")
            or ""
        )

        document_name = (
            metadata.get("source")
            or metadata.get("document_name")
            or metadata.get("filename")
            or "Unknown document"
        )

        page_number = metadata.get(
            "page",
            metadata.get("page_number", "Unknown")
        )

        if not isinstance(text, str) or not text.strip():
            continue

        context_parts.append(
            f"""
[CONTEXT {index}]
Document: {document_name}
Page: {page_number}
Text:
{text.strip()}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# JSON RESPONSE PARSING
# --------------------------------------------------

def parse_json_response(
    response_text: str
) -> dict[str, Any]:
    """
    Parse JSON returned by the judge model.

    Handles JSON enclosed in Markdown code fences.
    """

    if not isinstance(response_text, str):
        raise ValueError(
            "Judge returned a non-text response."
        )

    cleaned = response_text.strip()

    # Remove optional Markdown code fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        parsed = json.loads(cleaned)

    except json.JSONDecodeError:
        # Attempt to extract a JSON object from surrounding text.
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "Judge response did not contain valid JSON."
            )

        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError(
            "Judge response must be a JSON object."
        )

    return parsed


# --------------------------------------------------
# SCORE VALIDATION
# --------------------------------------------------

def validate_score(
    value: Any,
    field_name: str
) -> float:
    """
    Validate and normalize a score to the 0-1 range.
    """

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a number, not a boolean."
        )

    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a numeric score."
        )

    score = float(value)

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return round(score, 4)


# --------------------------------------------------
# NORMALIZE CLAIM RESULTS
# --------------------------------------------------

def normalize_claims(
    claims: Any
) -> list[dict[str, Any]]:
    """
    Validate individual claim-level judge results.
    """

    if not isinstance(claims, list):
        return []

    normalized_claims = []

    allowed_statuses = {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "UNCERTAIN",
    }

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        claim_text = claim.get("claim", "")

        if not isinstance(claim_text, str):
            continue

        status = str(
            claim.get("status", "UNCERTAIN")
        ).upper()

        if status not in allowed_statuses:
            status = "UNCERTAIN"

        normalized_claims.append(
            {
                "claim": claim_text,
                "status": status,
                "evidence": str(
                    claim.get("evidence", "")
                ),
                "explanation": str(
                    claim.get("explanation", "")
                ),
            }
        )

    return normalized_claims


# --------------------------------------------------
# SEMANTIC EVALUATION
# --------------------------------------------------

def evaluate_semantic_quality(
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a generated RAG answer using a Groq LLM judge.

    Parameters
    ----------
    question:
        Original user question.

    generated_answer:
        Answer produced by the RAG generator.

    ground_truth_answer:
        Expected/reference answer from the test dataset.

    retrieved_chunks:
        Chunks returned by the existing retriever.

    model:
        Optional judge model override.

    Returns
    -------
    A structured dictionary containing semantic metrics,
    claim analysis and an overall explanation.
    """

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "question must be a non-empty string."
        )

    if (
        not isinstance(generated_answer, str)
        or not generated_answer.strip()
    ):
        raise ValueError(
            "generated_answer must be a non-empty string."
        )

    if (
        not isinstance(ground_truth_answer, str)
        or not ground_truth_answer.strip()
    ):
        raise ValueError(
            "ground_truth_answer must be a non-empty string."
        )

    if not isinstance(retrieved_chunks, list):
        raise TypeError(
            "retrieved_chunks must be a list."
        )

    # --------------------------------------------------
    # 2. Prepare context
    # --------------------------------------------------

    context = prepare_context(retrieved_chunks)

    if not context.strip():
        return {
            "status": "INSUFFICIENT_CONTEXT",
            "judge_model": model or os.getenv(
                "RAG_EVALUATOR_MODEL",
                DEFAULT_MODEL
            ),
            "metrics": {
                "faithfulness": None,
                "answer_relevance": None,
                "answer_correctness": None,
                "hallucination_risk": None,
            },
            "claims": [],
            "explanation": (
                "No usable retrieved context was available. "
                "Semantic faithfulness and hallucination risk "
                "cannot be reliably evaluated."
            ),
            "limitations": [
                "Semantic scores were not generated because "
                "the retrieved context was empty."
            ],
        }

    # --------------------------------------------------
    # 3. Configure judge
    # --------------------------------------------------

    judge_model = (
        model
        or os.getenv(
            "RAG_EVALUATOR_MODEL",
            DEFAULT_MODEL
        )
    )

    client = get_groq_client()

    # --------------------------------------------------
    # 4. Build the judge prompt
    # --------------------------------------------------

    system_prompt = """
You are an independent evaluator for a Retrieval-Augmented
Generation (RAG) system.

Evaluate the generated answer using ONLY the supplied
retrieved context, question and ground-truth answer.

Do not assume that a statement is supported just because
it sounds plausible or is generally true.

Evaluation rules:

1. FAITHFULNESS:
   Check whether the factual claims in the generated answer
   are supported by the retrieved context.

2. ANSWER RELEVANCE:
   Check whether the answer directly addresses the question.

3. ANSWER CORRECTNESS:
   Compare the meaning of the generated answer with the
   ground-truth answer. Equivalent wording is acceptable.

4. HALLUCINATION RISK:
   Estimate how much of the generated answer contains claims
   that are unsupported or contradicted by the context.
   A high score means higher hallucination risk.

5. CLAIM ANALYSIS:
   Identify important factual claims in the generated answer.
   For every claim, provide:
   - claim
   - status
   - evidence
   - explanation

Allowed claim statuses:
- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED
- UNCERTAIN

Use UNCERTAIN when the available context does not allow
a confident decision.

Do not treat a missing detail as automatically false.
Do not invent supporting evidence or quotations.

Return ONLY a valid JSON object with this exact structure:

{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "answer_correctness": 0.0,
  "hallucination_risk": 0.0,
  "claims": [
    {
      "claim": "A factual claim from the answer",
      "status": "SUPPORTED",
      "evidence": "Relevant text from the supplied context",
      "explanation": "Why the claim received this status"
    }
  ],
  "explanation": "Concise explanation of the overall scores"
}

All scores must be numbers between 0 and 1.

For hallucination_risk:
- 0.0 means no apparent unsupported claims.
- 1.0 means the answer appears to consist entirely of
  unsupported or contradicted claims.

Be conservative and evidence-based.
"""

    user_prompt = f"""
QUESTION:
{question}

GROUND-TRUTH ANSWER:
{ground_truth_answer}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}

Evaluate the generated answer now.
"""

    # --------------------------------------------------
    # 5. Call Groq
    # --------------------------------------------------

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

    except Exception as error:
        logger.exception(
            "Semantic evaluation request failed."
        )

        raise RuntimeError(
            "Groq semantic evaluation failed. "
            "Check the API key, model availability, "
            "network connection and API limits. "
            f"Original error: {error}"
        ) from error

    # --------------------------------------------------
    # 6. Parse judge response
    # --------------------------------------------------

    if not response.choices:
        raise RuntimeError(
            "Groq judge returned no response choices."
        )

    message_content = response.choices[0].message.content

    if not message_content:
        raise RuntimeError(
            "Groq judge returned an empty response."
        )

    raw_result = parse_json_response(message_content)

    # --------------------------------------------------
    # 7. Validate metrics
    # --------------------------------------------------

    required_scores = [
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
        "hallucination_risk",
    ]

    metrics = {}

    for metric_name in required_scores:
        if metric_name not in raw_result:
            raise ValueError(
                f"Judge response is missing '{metric_name}'."
            )

        metrics[metric_name] = validate_score(
            raw_result[metric_name],
            metric_name
        )

    claims = normalize_claims(
        raw_result.get("claims", [])
    )

    explanation = raw_result.get(
        "explanation",
        "No explanation was provided by the judge."
    )

    if not isinstance(explanation, str):
        explanation = str(explanation)

    # --------------------------------------------------
    # 8. Return normalized result
    # --------------------------------------------------

    result = {
        "status": "COMPLETED",
        "judge_model": judge_model,
        "metrics": metrics,
        "claims": claims,
        "explanation": explanation,
        "limitations": [
            (
                "Scores are estimates produced by an LLM judge "
                "and may contain evaluation errors."
            ),
            (
                "Ground-truth correctness is estimated by comparing "
                "meaning, not by independently verifying external facts."
            ),
            (
                "A judge can miss subtle contradictions or unsupported "
                "claims; human review may still be necessary."
            ),
        ],
    }

    logger.info(
        "Semantic evaluation completed using model %s.",
        judge_model
    )

    return result
