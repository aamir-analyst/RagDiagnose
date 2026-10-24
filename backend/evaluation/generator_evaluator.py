import re
import unicodedata
from typing import Any


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """Normalize text for lexical comparison."""

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# BASIC LEXICAL METRICS
# =========================================================

def calculate_token_overlap(
    answer: str,
    reference: str
) -> float:
    """
    Calculate token-overlap F1.

    This is a lexical proxy, not a semantic correctness metric.
    """

    answer_tokens = set(normalize_text(answer).split())
    reference_tokens = set(normalize_text(reference).split())

    if not answer_tokens or not reference_tokens:
        return 0.0

    overlap = len(answer_tokens & reference_tokens)

    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return round(
        2 * precision * recall / (precision + recall),
        4
    )


def evaluate_answer_correctness(
    generated_answer: str,
    ground_truth_answer: str
) -> dict[str, Any]:
    """Compare generated answer with the reference answer."""

    score = calculate_token_overlap(
        generated_answer,
        ground_truth_answer
    )

    return {
        "score": score,
        "metric": "token_overlap_f1",
        "evaluation_type": "lexical_proxy",
        "warning": (
            "Lexical overlap does not independently establish "
            "semantic or factual correctness."
        )
    }


def evaluate_context_support(
    generated_answer: str,
    retrieved_chunks: list[dict]
) -> dict[str, Any]:
    """
    Measure the fraction of unique answer tokens found in context.

    This is not a verified faithfulness score.
    """

    context_parts = [
        chunk.get("text", "")
        for chunk in retrieved_chunks
        if isinstance(chunk.get("text", ""), str)
    ]

    context = " ".join(context_parts)

    answer_tokens = set(
        normalize_text(generated_answer).split()
    )

    context_tokens = set(
        normalize_text(context).split()
    )

    if not answer_tokens:
        score = 0.0
    else:
        overlap = len(answer_tokens & context_tokens)
        score = round(overlap / len(answer_tokens), 4)

    return {
        "score": score,
        "metric": "answer_context_token_overlap",
        "evaluation_type": "lexical_proxy",
        "warning": (
            "Token overlap cannot establish whether every answer "
            "claim is entailed by the retrieved context."
        )
    }


def evaluate_answer_completeness(
    generated_answer: str,
    ground_truth_answer: str
) -> dict[str, Any]:
    """Measure reference-token coverage in the generated answer."""

    answer_tokens = set(
        normalize_text(generated_answer).split()
    )

    reference_tokens = set(
        normalize_text(ground_truth_answer).split()
    )

    if not reference_tokens:
        score = 0.0
    else:
        overlap = len(answer_tokens & reference_tokens)
        score = round(overlap / len(reference_tokens), 4)

    return {
        "score": score,
        "metric": "reference_token_coverage",
        "evaluation_type": "lexical_proxy",
        "warning": (
            "Reference-token coverage does not independently "
            "establish that an answer is complete."
        )
    }


# =========================================================
# CITATION VALIDATION
# =========================================================

# Supported citation formats include:
# 【Source: ML_Questions.pdf, Page 1】
# [ML_Questions.pdf, Page 1]
# 【ML_Questions.pdf, page 1】
# 【ML_Questions.pdf, p.1】
# 【ML_Questions.pdf, p. 1】
# [Source: C:\docs\ML_Questions.pdf, Page 1]
#
# Note: citations must be wrapped in 【】 or [] and include a document
# filename plus a page indicator. This validates source references only;
# it does not prove that the cited source supports the claim.

# Require a filename and explicit page number. Supports bracketed citations,
# parenthesized Hindi/English citations, and labelled source lines.
# Unicode spaces are normalized before matching.
CITATION_PATTERN = re.compile(
    r"(?P<opening>[【\[\(])?\s*"
    r"(?:(?:\*{1,2})?(?:Source|स्रोत)(?:\*{1,2})?\s*[:：]\s*)?"
    r"(?P<document>[^【\[\]】(),\n]+?\.(?:pdf|docx?|txt|md|pptx?|csv|html?))"
    r"\s*,\s*(?:Page\s*|p\.?\s*|पृष्ठ\s*)"
    r"(?P<page>\d+)\s*[】\]\)]?",
    re.IGNORECASE
)


def normalize_document_name(document: Any) -> str:
    """Normalize a document filename for source-reference matching."""

    if not isinstance(document, str):
        return ""

    # Support Windows and POSIX paths, and ignore case/outer whitespace.
    return document.strip().replace(chr(92), "/").rsplit("/", 1)[-1].casefold()



def extract_citations(answer: str) -> list[dict[str, Any]]:
    """Extract explicit document/page references from supported citation formats."""

    if not isinstance(answer, str):
        return []

    # Normalize Unicode spacing/punctuation, including narrow no-break spaces.
    normalized_answer = unicodedata.normalize("NFKC", answer)
    normalized_answer = normalized_answer.replace("\\u200b", "").replace("\\ufeff", "")

    citations = []
    seen = set()

    for match in CITATION_PATTERN.finditer(normalized_answer):
        document = match.group("document").strip().strip("*").strip()
        page = int(match.group("page"))
        citation_text = match.group(0).strip()

        key = (normalize_document_name(document), page, citation_text.casefold())
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "document": document,
            "page": page,
            "citation_text": citation_text
        })

    return citations


def evaluate_citations(
    generated_answer: str,
    retrieved_chunks: list[dict]
) -> dict[str, Any]:
    """
    Check whether cited document/page pairs occur in retrieved sources.

    This validates source-reference matching only. It does not prove
    that the source supports the claim.
    """

    available_sources = set()

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") or {}

        document = (
            metadata.get("document_name")
            or metadata.get("source")
            or metadata.get("filename")
            or metadata.get("file_name")
        )
        page = metadata.get("page_number", metadata.get("page"))

        if document is None or page is None:
            continue

        try:
            page = int(page)
        except (TypeError, ValueError):
            continue

        normalized_document = normalize_document_name(document)
        if normalized_document:
            available_sources.add((normalized_document, page))

    citations = extract_citations(generated_answer)

    checked_citations = []

    for citation in citations:
        key = (
            normalize_document_name(citation["document"]),
            citation["page"]
        )

        matched = key in available_sources

        checked_citations.append({
            **citation,
            "source_reference_found": matched
        })

    valid_count = sum(
        item["source_reference_found"]
        for item in checked_citations
    )

    citation_count = len(checked_citations)

    if citation_count:
        validity_score = round(
            valid_count / citation_count,
            4
        )
    else:
        validity_score = None

    return {
        "citation_count": citation_count,
        "valid_citation_count": valid_count,
        "invalid_citation_count": citation_count - valid_count,
        "citation_validity_score": validity_score,
        "citations": checked_citations,
        "evaluation_type": "source_reference_validation",
        "warning": (
            "A matching document/page reference does not prove that "
            "the cited text supports the associated claim. "
            "No citations is reported as None, not as a verified failure."
        )
    }


# =========================================================
# CLAIM-LEVEL FAITHFULNESS: LEXICAL BASELINE
# =========================================================

def split_into_claims(answer: str) -> list[str]:
    """
    Split answer text into simple sentence-like units.

    This is a heuristic, not a full linguistic claim extractor.
    """

    if not isinstance(answer, str) or not answer.strip():
        return []

    # Remove source citations before splitting claims.
    text = CITATION_PATTERN.sub("", answer)

    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)

    claims = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return claims


def evaluate_claim_support(
    generated_answer: str,
    retrieved_chunks: list[dict]
) -> dict[str, Any]:
    """
    Evaluate each sentence-like claim using token overlap.

    Results are heuristic indicators only. They do not determine
    semantic entailment or confirm hallucination.
    """

    claims = split_into_claims(generated_answer)

    context_parts = [
        chunk.get("text", "")
        for chunk in retrieved_chunks
        if isinstance(chunk.get("text", ""), str)
    ]

    context = " ".join(context_parts)
    context_tokens = set(normalize_text(context).split())

    claim_results = []

    for claim in claims:
        claim_tokens = set(normalize_text(claim).split())

        if not claim_tokens:
            overlap_score = 0.0
        else:
            overlap_score = round(
                len(claim_tokens & context_tokens)
                / len(claim_tokens),
                4
            )

        claim_results.append({
            "claim": claim,
            "context_token_overlap": overlap_score,
            "support_status": "NEEDS_SEMANTIC_VERIFICATION"
        })

    if claim_results:
        average_overlap = round(
            sum(
                item["context_token_overlap"]
                for item in claim_results
            ) / len(claim_results),
            4
        )
    else:
        average_overlap = 0.0

    return {
        "claim_count": len(claim_results),
        "average_claim_context_overlap": average_overlap,
        "claims": claim_results,
        "evaluation_type": "claim_level_lexical_heuristic",
        "warning": (
            "Claims are split heuristically. Token overlap cannot "
            "verify entailment, so every claim requires semantic "
            "verification before being marked supported or unsupported."
        )
    }


# =========================================================
# COMPLETE GENERATOR EVALUATION
# =========================================================

def evaluate_generator(
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    retrieved_chunks: list[dict]
) -> dict[str, Any]:
    """Run all current generator-evaluation checks."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if (
        not isinstance(generated_answer, str)
        or not generated_answer.strip()
    ):
        raise ValueError("Generated answer cannot be empty.")

    if (
        not isinstance(ground_truth_answer, str)
        or not ground_truth_answer.strip()
    ):
        raise ValueError("Ground-truth answer cannot be empty.")

    if not isinstance(retrieved_chunks, list):
        raise TypeError("retrieved_chunks must be a list.")

    correctness = evaluate_answer_correctness(
        generated_answer,
        ground_truth_answer
    )

    context_support = evaluate_context_support(
        generated_answer,
        retrieved_chunks
    )

    completeness = evaluate_answer_completeness(
        generated_answer,
        ground_truth_answer
    )

    citation_evaluation = evaluate_citations(
        generated_answer,
        retrieved_chunks
    )

    claim_evaluation = evaluate_claim_support(
        generated_answer,
        retrieved_chunks
    )

    return {
        "question": question,
        "generated_answer": generated_answer,
        "ground_truth_answer": ground_truth_answer,
        "metrics": {
            "answer_correctness": correctness,
            "context_support": context_support,
            "answer_completeness": completeness,
            "citation_validation": citation_evaluation,
            "claim_level_faithfulness": claim_evaluation
        },
        "evaluation_note": (
            "Correctness, context support, completeness and claim "
            "support currently use lexical or heuristic methods. "
            "Citation validation checks document/page references only. "
            "These checks do not independently prove factual correctness, "
            "semantic faithfulness or hallucination."
        )
    }