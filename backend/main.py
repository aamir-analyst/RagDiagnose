
from pathlib import Path
import logging
import shutil
import uuid
import math

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import APIStatusError

from ingestion.loader import extract_text_from_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import store_chunks
from rag.retriever import retrieve_documents
from rag.generator import generate_answer

from evaluation.retriever_evaluator import evaluate_retrieval
from evaluation.generator_evaluator import evaluate_generator


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )
)

logger = logging.getLogger("ragdiagnose")


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="RAGDiagnose API",
    description=(
        "RAG Evaluation, Monitoring and "
        "Root-Cause Analysis Platform"
    ),
    version="1.4.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================================
# DIRECTORIES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "data" / "documents"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# REQUEST MODELS
# =====================================================

class EvaluationRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=2,
        description="Question to evaluate"
    )

    ground_truth_answer: str = Field(
        ...,
        min_length=1,
        description="Expected correct answer"
    )

    relevant_chunk_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Expected relevant chunk IDs"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )


# =====================================================
# CONFIDENCE ESTIMATE
# =====================================================

def calculate_confidence(chunks: list[dict]) -> float:
    """
    Calculate average retrieval similarity.

    This is a retrieval-based estimate, not a verified
    measure of answer correctness.
    """

    if not chunks:
        return 0.0

    scores = []

    for chunk in chunks:
        try:
            score = float(
                chunk.get("similarity", 0.0)
            )

            if not math.isfinite(score):
                continue

            scores.append(
                max(0.0, min(score, 1.0))
            )

        except (TypeError, ValueError):
            continue

    if not scores:
        return 0.0

    return round(
        sum(scores) / len(scores),
        4
    )


# =====================================================
# ROOT-CAUSE DIAGNOSIS
# =====================================================

def diagnose_root_cause(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    generator_metrics: dict
) -> dict:
    """
    Initial heuristic diagnosis based on retrieval evidence
    and lexical evaluation proxies.

    This function does not independently prove hallucination
    or factual correctness.
    """

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    relevant_found = bool(
        retrieved_set.intersection(relevant_set)
    )

    metrics = generator_metrics

    correctness = metrics[
        "answer_correctness"
    ]["score"]

    completeness = metrics[
        "answer_completeness"
    ]["score"]

    context_support = metrics[
        "context_support"
    ]["score"]

    evidence = {
        "relevant_chunk_found": relevant_found,
        "retrieved_chunk_ids": retrieved_ids,
        "expected_relevant_chunk_ids": relevant_ids,
        "answer_correctness_proxy": correctness,
        "answer_completeness_proxy": completeness,
        "context_support_proxy": context_support
    }

    # -------------------------------------------------
    # CASE 1: RETRIEVAL FAILURE
    # -------------------------------------------------

    if not relevant_found:

        return {
            "root_cause": "RETRIEVAL_FAILURE",
            "status": "FAIL",
            "explanation": (
                "Koi expected relevant chunk retrieve nahi hua. "
                "Retrieval failure ka strong signal mila hai."
            ),
            "evidence": evidence,
            "recommendations": [
                "Chunking strategy review karo.",
                "Embedding model ki relevance evaluate karo.",
                "Hybrid search test karo.",
                "Top-K aur reranking configurations compare karo."
            ]
        }

    # -------------------------------------------------
    # CASE 2: POSSIBLE GENERATOR FAILURE
    # -------------------------------------------------

    if correctness < 0.5:

        return {
            "root_cause": "POSSIBLE_GENERATOR_FAILURE",
            "status": "NEEDS_REVIEW",
            "explanation": (
                "Generated answer aur ground truth ka "
                "lexical overlap low hai. Semantic equivalence "
                "verify karna zaroori hai."
            ),
            "evidence": evidence,
            "recommendations": [
                "Generated answer ko ground truth se compare karo.",
                "Retrieved context inspect karo.",
                "Generation prompt review karo."
            ]
        }

    # -------------------------------------------------
    # CASE 3: POSSIBLE INCOMPLETE ANSWER
    # -------------------------------------------------

    if completeness < 0.5:

        return {
            "root_cause": "POSSIBLE_INCOMPLETE_ANSWER",
            "status": "NEEDS_REVIEW",
            "explanation": (
                "Generated answer mein ground truth ke "
                "kaafi tokens missing hain. Completeness "
                "verify karni hogi."
            ),
            "evidence": evidence,
            "recommendations": [
                "Expected answer ke required facts identify karo.",
                "Retrieved context ki completeness check karo.",
                "Answer generation prompt improve karo."
            ]
        }

    # -------------------------------------------------
    # CASE 4: POSSIBLE UNSUPPORTED ANSWER
    # -------------------------------------------------

    if context_support < 0.3:

        return {
            "root_cause": "POSSIBLE_UNSUPPORTED_ANSWER",
            "status": "NEEDS_REVIEW",
            "explanation": (
                "Answer aur retrieved context ka lexical "
                "overlap low hai. Yeh hallucination ka proof "
                "nahi hai; claim-level verification required hai."
            ),
            "evidence": evidence,
            "recommendations": [
                "Answer ko individual claims mein divide karo.",
                "Har claim ke liye supporting context identify karo.",
                "Faithfulness evaluation add karo."
            ]
        }

    # -------------------------------------------------
    # CASE 5: NO CLEAR FAILURE
    # -------------------------------------------------

    return {
        "root_cause": "NO_CLEAR_FAILURE_DETECTED",
        "status": "NEEDS_REVIEW",
        "explanation": (
            "Basic evaluation metrics mein clear failure "
            "identify nahi hua. PASS declare karne se pehle "
            "answer correctness aur citations verify karo."
        ),
        "evidence": evidence,
        "recommendations": [
            "Generated answer aur source evidence verify karo.",
            "Citation correctness evaluate karo.",
            "Claim-level faithfulness checks implement karo."
        ]
    }


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "RAGDiagnose API is running",
        "version": "1.4.0",
        "endpoints": [
            "/",
            "/health",
            "/documents/upload",
            "/query",
            "/evaluation/run",
            "/docs"
        ]
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():

    return {
        "success": True,
        "status": "healthy"
    }


# =====================================================
# PDF UPLOAD
# =====================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload PDF, extract text, create chunks
    and store embeddings in ChromaDB.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name missing hai."
        )

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Sirf PDF files allowed hain."
        )

    safe_filename = Path(file.filename).name

    unique_name = (
        f"{uuid.uuid4().hex}_{safe_filename}"
    )

    file_path = UPLOAD_DIR / unique_name

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        pages = extract_text_from_pdf(
            str(file_path)
        )

        if not pages:
            raise HTTPException(
                status_code=400,
                detail=(
                    "PDF mein readable text nahi mila. "
                    "Scanned PDF ke liye OCR required hoga."
                )
            )

        chunks = create_chunks(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="PDF se koi valid text chunk nahi bana."
            )

        stored_count = store_chunks(
            chunks=chunks,
            document_name=safe_filename
        )

        logger.info(
            "Document uploaded: %s, pages=%s, chunks=%s",
            safe_filename,
            len(pages),
            stored_count
        )

        return {
            "success": True,
            "message": "Document successfully uploaded.",
            "filename": safe_filename,
            "pages": len(pages),
            "chunks": stored_count
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Document processing failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed. "
                "Backend logs check karo."
            )
        ) from error

    finally:
        await file.close()


# =====================================================
# DOCUMENT QUERY
# =====================================================

@app.get("/query")
def query_document(
    question: str = Query(
        ...,
        min_length=2
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=10
    )
):
    """
    Retrieve context, generate answer and return citations.
    """

    try:

        retrieved_chunks = retrieve_documents(
            query=question,
            top_k=top_k
        )

        if not retrieved_chunks:

            return {
                "success": True,
                "question": question,
                "answer": (
                    "Uploaded documents mein is question "
                    "ka relevant answer nahi mila."
                ),
                "model": None,
                "confidence": 0.0,
                "sources": [],
                "retrieved_chunks": 0
            }

        generated = generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        confidence = calculate_confidence(
            retrieved_chunks
        )

        sources = []

        for chunk in retrieved_chunks:

            metadata = chunk.get("metadata") or {}

            sources.append({
                "document": metadata.get("document_name"),
                "page": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id"),
                "similarity": chunk.get("similarity", 0.0),
                "distance": chunk.get("distance")
            })

        return {
            "success": True,
            "question": question,
            "answer": generated["answer"],
            "model": generated.get("model"),
            "confidence": confidence,
            "sources": sources,
            "retrieved_chunks": len(retrieved_chunks)
        }

    except APIStatusError as error:

        logger.exception("Groq API request failed.")

        raise HTTPException(
            status_code=502,
            detail=(
                "Groq API request failed. "
                "Model availability aur API configuration check karo."
            )
        ) from error

    except Exception as error:

        logger.exception("Document query failed.")

        raise HTTPException(
            status_code=500,
            detail="Query processing failed. Backend logs check karo."
        ) from error


# =====================================================
# COMBINED RETRIEVER + GENERATOR EVALUATION
# =====================================================

@app.post("/evaluation/run")
def run_rag_evaluation(
    request: EvaluationRequest
):
    """
    Evaluate retrieval and answer generation, then return
    an initial heuristic root-cause diagnosis.
    """

    try:

        # ---------------------------------------------
        # STEP 1: RETRIEVAL
        # ---------------------------------------------

        retrieved_chunks = retrieve_documents(
            query=request.question,
            top_k=request.top_k
        )

        # ---------------------------------------------
        # STEP 2: RETRIEVER EVALUATION
        # ---------------------------------------------

        retrieval_result = evaluate_retrieval(
            retrieved_chunks=retrieved_chunks,
            relevant_chunk_ids=request.relevant_chunk_ids,
            k_values=[1, 3, 5, 10]
        )

        # ---------------------------------------------
        # STEP 3: ANSWER GENERATION
        # ---------------------------------------------

        if retrieved_chunks:

            generated = generate_answer(
                question=request.question,
                retrieved_chunks=retrieved_chunks
            )

            generated_answer = generated["answer"]
            model_name = generated.get("model")

        else:

            generated_answer = (
                "The answer is not available "
                "in the uploaded documents."
            )

            model_name = None

        # ---------------------------------------------
        # STEP 4: GENERATOR EVALUATION
        # ---------------------------------------------

        generator_result = evaluate_generator(
            question=request.question,
            generated_answer=generated_answer,
            ground_truth_answer=request.ground_truth_answer,
            retrieved_chunks=retrieved_chunks
        )

        # ---------------------------------------------
        # STEP 5: ROOT-CAUSE DIAGNOSIS
        # ---------------------------------------------

        diagnosis = diagnose_root_cause(
            retrieved_ids=retrieval_result[
                "retrieved_chunk_ids"
            ],
            relevant_ids=retrieval_result[
                "relevant_chunk_ids"
            ],
            generator_metrics=generator_result["metrics"]
        )

        # ---------------------------------------------
        # STEP 6: SOURCE INFORMATION
        # ---------------------------------------------

        sources = []

        for chunk in retrieved_chunks:

            metadata = chunk.get("metadata") or {}

            sources.append({
                "document": metadata.get("document_name"),
                "page": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id"),
                "similarity": chunk.get("similarity", 0.0),
                "distance": chunk.get("distance")
            })

        # ---------------------------------------------
        # STEP 7: FINAL REPORT
        # ---------------------------------------------

        return {
            "success": True,
            "question": request.question,
            "model": model_name,
            "generated_answer": generated_answer,
            "sources": sources,
            "retrieval_evaluation": retrieval_result,
            "generator_evaluation": generator_result,
            "root_cause_diagnosis": diagnosis
        }

    except APIStatusError as error:

        logger.exception(
            "Groq API failed during RAG evaluation."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Groq API request failed. "
                "Check model availability, API key and rate limits."
            )
        ) from error

    except Exception as error:

        logger.exception(
            "Combined RAG evaluation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG evaluation failed. "
                "Backend logs check karo."
            )
        ) from error