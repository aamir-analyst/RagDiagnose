import json
import os
from datetime import datetime

import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAGDiagnose | AI Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 180


def get_api_url():
    """Read the backend URL from Streamlit secrets or environment."""

    try:
        configured_url = st.secrets.get(
            "API_BASE_URL",
            os.getenv("API_BASE_URL", DEFAULT_API_URL),
        )
    except Exception:
        configured_url = os.getenv(
            "API_BASE_URL",
            DEFAULT_API_URL,
        )

    return configured_url.strip().rstrip("/")


API_URL = get_api_url()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "chat_history": [],
    "uploaded_documents": [],
    "latest_evaluation": None,
    "api_status": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# API CLIENT
# ============================================================

def api_request(method, endpoint, **kwargs):
    """Send an HTTP request to the FastAPI backend."""

    url = f"{API_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = {
                "detail": response.text or "Invalid response from backend."
            }

        if not response.ok:
            detail = payload.get(
                "detail",
                f"Backend returned HTTP {response.status_code}.",
            )

            st.error(
                f"**API Error — HTTP {response.status_code}**\n\n"
                f"{detail}"
            )

            return None

        return payload

    except requests.Timeout:
        st.error(
            "Backend request timed out. "
            "Document processing ya LLM response mein zyada time lag sakta hai."
        )

    except requests.ConnectionError:
        st.error(
            "Backend se connection nahi ho raha.\n\n"
            "Check karo ki FastAPI server port 8000 par running hai."
        )

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")

    return None


def check_backend():
    """Check whether the backend is healthy."""

    result = api_request(
        "GET",
        "/health",
        timeout=10,
    )

    is_healthy = bool(
        result and result.get("status") == "healthy"
    )

    st.session_state.api_status = is_healthy

    return is_healthy


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                ellipse at top right,
                rgba(37, 99, 235, 0.10),
                transparent 42%
            ),
            #080d19;

        color: #e8edf7;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    p, li, label {
        line-height: 1.65;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.4px;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #0d1424;
        border-right: 1px solid #243047;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.6rem;
    }

    [data-testid="stSidebar"] p {
        color: #c5d0e2;
    }

    /* ---------- Hero ---------- */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 34px;
        border-radius: 22px;
        margin-bottom: 28px;

        background:
            linear-gradient(
                120deg,
                rgba(30, 64, 175, 0.48),
                rgba(67, 56, 202, 0.34),
                rgba(8, 145, 178, 0.20)
            );

        border: 1px solid rgba(125, 165, 255, 0.25);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.18);
    }

    .hero h1 {
        font-size: clamp(30px, 3.2vw, 44px);
        font-weight: 800;
        margin: 12px 0;
        color: #ffffff !important;
    }

    .hero p {
        font-size: 16px;
        color: #d6e3ff;
        max-width: 760px;
        margin-bottom: 12px;
    }

    .eyebrow {
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 12px;
        margin-top: 8px;

        background: rgba(15, 23, 42, 0.48);
        border: 1px solid rgba(191, 219, 254, 0.20);
        border-radius: 30px;

        font-size: 12px;
        color: #dbeafe;
    }

    /* ---------- Cards ---------- */

    .feature-card {
        height: 100%;
        min-height: 145px;

        padding: 21px;
        margin-bottom: 16px;

        border: 1px solid #263550;
        border-radius: 16px;
        background: linear-gradient(145deg, #121c30, #0d1525);
    }

    .feature-card h3 {
        margin-top: 12px;
        margin-bottom: 8px;
        font-size: 18px;
    }

    .feature-card p {
        color: #aebbd0;
        font-size: 14px;
    }

    .feature-icon {
        font-size: 26px;
    }

    .section-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 800;
        color: #8ea5c9;
        margin-bottom: 8px;
    }

    .muted {
        color: #9eacc2;
        font-size: 14px;
    }

    /* ---------- Metrics ---------- */

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131f34, #0f1729);
        border: 1px solid #293b59;
        border-radius: 15px;
        padding: 19px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b5c5dc;
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        color: #93c5fd;
        font-weight: 750;
    }

    /* ---------- Buttons ---------- */

    .stButton button,
    .stDownloadButton button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        transition: all 0.15s ease;
    }

    .stButton button:hover {
        border-color: #60a5fa;
    }

    /* ---------- Inputs ---------- */

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: #101a2d;
        color: #f1f5f9;
        border-radius: 10px;
        border-color: #2c3d5a;
    }

    div[data-testid="stFileUploader"] {
        background: #10192b;
        border: 1px dashed #40577b;
        border-radius: 15px;
        padding: 16px;
    }

    /* ---------- Expanders ---------- */

    [data-testid="stExpander"] {
        border: 1px solid #293954;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.50);
    }

    /* ---------- Dividers ---------- */

    hr {
        border-color: #25324a;
    }

    /* ---------- Footer ---------- */

    .app-footer {
        color: #7f90aa;
        font-size: 12px;
        text-align: center;
        padding: 18px 0 4px 0;
    }

    /* ---------- Tables ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #293954;
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REUSABLE UI HELPERS
# ============================================================

def render_page_heading(title, description, icon=""):
    """Render a consistent page heading."""

    st.markdown(
        f"""
        <div class="section-label">{icon} RAGDIAGNOSE WORKSPACE</div>
        <h1 style="font-size: 32px; margin-bottom: 4px;">
            {title}
        </h1>
        <p class="muted">{description}</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()


def render_source_list(sources):
    """Display source metadata returned by the backend."""

    if not sources:
        st.info("Is query ke liye koi source retrieve nahi hua.")
        return

    st.markdown("### 📚 Retrieved Sources")

    for index, source in enumerate(sources, start=1):
        document = source.get("document") or "Unknown document"
        page = source.get("page")
        similarity = source.get("similarity", 0)

        with st.expander(
            f"Source {index} · {document}",
            expanded=index == 1,
        ):
            col1, col2 = st.columns(2)

            col1.write(f"**Page:** {page if page is not None else 'Unknown'}")
            col2.write(f"**Similarity:** {similarity}")

            st.write(
                f"**Chunk ID:** {source.get('chunk_id') or 'Unknown'}"
            )

            if source.get("distance") is not None:
                st.write(f"**Distance:** {source['distance']}")


def render_evaluation_result(result):
    """Display the evaluation response."""

    st.success("Evaluation request completed.")

    st.markdown("### 📝 Generated Answer")
    st.write(result.get("generated_answer", "No answer returned."))

    retrieval = result.get("retrieval_evaluation", {})
    generator = result.get("generator_evaluation", {})
    diagnosis = result.get("root_cause_diagnosis", {})
    sources = result.get("sources", [])

    st.markdown("### 📊 Evaluation Results")

    tab1, tab2, tab3 = st.tabs(
        [
            "Retrieval",
            "Generator",
            "Diagnosis",
        ]
    )

    with tab1:
        st.json(retrieval)

    with tab2:
        st.json(generator)

    with tab3:
        st.markdown(f"**Root cause:** {diagnosis.get('root_cause', 'Unknown')}")
        st.markdown(f"**Status:** {diagnosis.get('status', 'Unknown')}")
        st.write(diagnosis.get("explanation", "No explanation returned."))

        st.markdown("**Recommendations**")

        recommendations = diagnosis.get("recommendations", [])

        if recommendations:
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        else:
            st.info("No recommendations returned.")

        with st.expander("View diagnostic evidence"):
            st.json(diagnosis.get("evidence", {}))

    st.markdown("### 📚 Sources")

    if sources:
        st.dataframe(
            sources,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No retrieved sources were returned.")

    st.download_button(
        label="⬇️ Download Evaluation Report",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="ragdiagnose_evaluation.json",
        mime="application/json",
        use_container_width=False,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0 0 15px 0;">
            <div style="font-size: 28px; font-weight: 850;">
                🔬 RAG<span style="color:#60a5fa;">Diagnose</span>
            </div>
            <div style="color:#93a4bd; font-size:12px;">
                AI Evaluation & Observability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Knowledge Base",
            "RAG Chat",
            "Evaluation Lab",
            "Root Cause Analysis",
        ],
        key="main_navigation",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### SYSTEM STATUS")

    if st.button("↻ Check Backend", use_container_width=True):
        check_backend()

    if st.session_state.api_status is True:
        st.success("Backend Online")
    elif st.session_state.api_status is False:
        st.error("Backend Offline")
    else:
        st.caption("Click above to check the API.")

    st.caption("FastAPI · ChromaDB · Groq")

    st.divider()

    st.markdown("#### CONFIGURATION")
    st.caption(f"API endpoint: {API_URL}")

    st.markdown(
        """
        <div style="
            margin-top:20px;
            padding:12px;
            border-radius:12px;
            background:#121d30;
            border:1px solid #263650;
            color:#a8bad3;
            font-size:12px;
        ">
            Your documents are processed by the configured backend.
            Do not upload confidential data unless you are authorised
            to do so.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">RAG INTELLIGENCE PLATFORM</div>
        <h1>Understand. Evaluate. Diagnose.</h1>
        <p>
            Monitor your Retrieval-Augmented Generation pipeline,
            inspect source evidence, evaluate answers, and investigate
            potential retrieval or generation failures.
        </p>
        <span class="hero-badge">● Document-grounded AI</span>
        <span class="hero-badge">● Retrieval evaluation</span>
        <span class="hero-badge">● Root-cause signals</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    render_page_heading(
        "Overview",
        "Your central workspace for document-grounded AI and RAG evaluation.",
        "01",
    )

    health_col, model_col, version_col = st.columns(3)

    with health_col:
        st.markdown("##### Backend")

        if st.button(
            "Test API Connection",
            key="overview_health",
            use_container_width=True,
        ):
            check_backend()

        if st.session_state.api_status is True:
            st.success("Online")
        elif st.session_state.api_status is False:
            st.error("Unavailable")
        else:
            st.info("Not checked")

    with model_col:
        st.markdown("##### Language Model")
        st.metric("Provider", "Groq")

    with version_col:
        st.markdown("##### Application")
        st.metric("API Version", "1.4.0")

    st.divider()

    st.markdown("## Your workflow")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <h3>Knowledge Base</h3>
                <p>
                    Upload a PDF, extract its text, generate embeddings,
                    and index document chunks in ChromaDB.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Knowledge Base →",
            key="go_knowledge",
            use_container_width=True,
        ):
            st.session_state.main_navigation = "Knowledge Base"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <h3>RAG Chat</h3>
                <p>
                    Ask questions, retrieve relevant document chunks,
                    and inspect answer sources and similarity scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open RAG Chat →",
            key="go_chat",
            use_container_width=True,
        ):
            st.session_state.main_navigation = "RAG Chat"
            st.rerun()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧪</div>
                <h3>Evaluation Lab</h3>
                <p>
                    Compare retrieved chunks against expected relevant
                    chunks and evaluate generated answers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Evaluation Lab →",
            key="go_evaluation",
            use_container_width=True,
        ):
            st.session_state.main_navigation = "Evaluation Lab"
            st.rerun()

    with c4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Root Cause Analysis</h3>
                <p>
                    Review heuristic failure signals, inspect supporting
                    evidence, and explore recommended improvements.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Diagnosis →",
            key="go_diagnosis",
            use_container_width=True,
        ):
            st.session_state.main_navigation = "Root Cause Analysis"
            st.rerun()

    st.divider()

    st.markdown("## Architecture")

    st.code(
        """
PDF Document
     |
     v
Text Extraction -> Chunking -> Embeddings
                                  |
                                  v
                              ChromaDB
                                  |
User Question -> Embedding -> Retrieval
                                  |
                                  v
                            Groq Generator
                                  |
                                  v
                          Answer + Sources
                                  |
                                  v
                    Evaluation & Diagnosis
        """,
        language="text",
    )

    st.caption(
        "Architecture reflects the current application design. "
        "The diagnosis module provides heuristic signals, not guaranteed "
        "proof of causation or hallucination."
    )


# ============================================================
# PAGE 2 — KNOWLEDGE BASE
# ============================================================

elif page == "Knowledge Base":

    render_page_heading(
        "Knowledge Base",
        "Ingest PDF documents and prepare them for semantic retrieval.",
        "02",
    )

    st.markdown("### Upload a document")

    st.markdown(
        """
        <div class="muted">
            Supported format: PDF · Scanned PDFs may require OCR.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Select your PDF document",
        type=["pdf"],
        key="knowledge_base_uploader",
    )

    if uploaded_file:
        info1, info2 = st.columns(2)

        info1.metric("Selected File", uploaded_file.name)
        info2.metric(
            "File Size",
            f"{uploaded_file.size / (1024 * 1024):.2f} MB",
        )

        if st.button(
            "Process & Index Document",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Extracting text, creating chunks and storing embeddings..."
            ):
                result = api_request(
                    "POST",
                    "/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                )

            if result and result.get("success"):
                document_info = {
                    "filename": result.get("filename", uploaded_file.name),
                    "pages": result.get("pages", 0),
                    "chunks": result.get("chunks", 0),
                    "uploaded_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }

                documents = st.session_state.uploaded_documents

                if not any(
                    doc["filename"] == document_info["filename"]
                    for doc in documents
                ):
                    documents.append(document_info)

                st.success("Document indexed successfully.")

                m1, m2 = st.columns(2)

                m1.metric("Pages Processed", document_info["pages"])
                m2.metric("Chunks Indexed", document_info["chunks"])

                with st.expander("Processing response"):
                    st.json(result)

    st.divider()

    st.markdown("### Documents processed in this session")

    documents = st.session_state.uploaded_documents

    if documents:
        st.dataframe(
            documents,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No successful uploads have been recorded in this Streamlit "
            "session yet. Existing ChromaDB documents may still be available."
        )

    st.warning(
        "This list records successful uploads in the current Streamlit "
        "session only. It is not a complete inventory of ChromaDB. "
        "The existing backend does not currently expose a document-list "
        "endpoint."
    )


# ============================================================
# PAGE 3 — RAG CHAT
# ============================================================

elif page == "RAG Chat":

    render_page_heading(
        "RAG Chat",
        "Ask questions and inspect the context retrieved by your RAG pipeline.",
        "03",
    )

    settings_col, help_col = st.columns([1, 2])

    with settings_col:
        top_k = st.slider(
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=5,
        )

    with help_col:
        st.info(
            "Answers depend on the documents available in the backend's "
            "ChromaDB collection. Upload a document first if needed."
        )

    st.divider()

    for index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                metadata = message.get("metadata", {})

                if metadata.get("sources"):
                    with st.expander("View retrieved sources"):
                        render_source_list(metadata["sources"])

                if metadata.get("confidence") is not None:
                    st.caption(
                        "Retrieval similarity estimate: "
                        f"{metadata['confidence'] * 100:.1f}%"
                    )

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating answer..."):
                result = api_request(
                    "GET",
                    "/query",
                    params={
                        "question": question,
                        "top_k": top_k,
                    },
                )

            if result and result.get("success"):
                answer = result.get(
                    "answer",
                    "The backend returned no answer.",
                )

                st.markdown(answer)

                confidence = result.get("confidence", 0) or 0

                metadata = {
                    "sources": result.get("sources", []),
                    "confidence": confidence,
                    "model": result.get("model"),
                }

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                    }
                )

                if metadata["sources"]:
                    with st.expander("View retrieved sources"):
                        render_source_list(metadata["sources"])

                st.caption(
                    f"Similarity estimate: {confidence * 100:.1f}%"
                    f" · Model: {metadata['model'] or 'Not available'}"
                )

            else:
                st.warning(
                    "Answer generate nahi ho paya. "
                    "Backend status aur logs check karo."
                )

    if st.session_state.chat_history:
        st.divider()

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    st.caption(
        "Similarity-based confidence is not a calibrated probability "
        "of answer correctness."
    )


# ============================================================
# PAGE 4 — EVALUATION LAB
# ============================================================

elif page == "Evaluation Lab":

    render_page_heading(
        "Evaluation Lab",
        "Run a test case against your existing retrieval and generation pipeline.",
        "04",
    )

    st.info(
        "Evaluation ke liye question, expected answer aur kam se kam ek "
        "relevant chunk ID zaroori hai. Relevant IDs tumhare dataset ya "
        "document metadata se lene honge."
    )

    with st.form("evaluation_form"):

        eval_question = st.text_area(
            "Test Question",
            placeholder="Enter the question you want to evaluate...",
            height=90,
        )

        ground_truth = st.text_area(
            "Expected Answer / Ground Truth",
            placeholder="Write the expected answer...",
            height=100,
        )

        relevant_ids_text = st.text_area(
            "Expected Relevant Chunk IDs",
            placeholder="Enter one chunk ID per line",
            height=100,
        )

        eval_top_k = st.slider(
            "Evaluation Top-K",
            min_value=1,
            max_value=10,
            value=5,
        )

        run_evaluation = st.form_submit_button(
            "Run Full Evaluation",
            type="primary",
            use_container_width=True,
        )

    if run_evaluation:

        relevant_ids = [
            line.strip()
            for line in relevant_ids_text.splitlines()
            if line.strip()
        ]

        if len(eval_question.strip()) < 2:
            st.error("Enter a valid test question.")

        elif not ground_truth.strip():
            st.error("Expected answer is required.")

        elif not relevant_ids:
            st.error("Enter at least one relevant chunk ID.")

        else:
            payload = {
                "question": eval_question.strip(),
                "ground_truth_answer": ground_truth.strip(),
                "relevant_chunk_ids": relevant_ids,
                "top_k": eval_top_k,
            }

            with st.spinner(
                "Evaluating retrieval, generation and diagnosis..."
            ):
                result = api_request(
                    "POST",
                    "/evaluation/run",
                    json=payload,
                )

            if result and result.get("success"):
                st.session_state.latest_evaluation = result

                render_evaluation_result(result)

    previous_result = st.session_state.latest_evaluation

    if previous_result and not run_evaluation:
        st.divider()

        with st.expander("View latest evaluation from this session"):
            render_evaluation_result(previous_result)


# ============================================================
# PAGE 5 — ROOT CAUSE ANALYSIS
# ============================================================

elif page == "Root Cause Analysis":

    render_page_heading(
        "Root Cause Analysis",
        "Investigate heuristic failure signals from your latest evaluation.",
        "05",
    )

    result = st.session_state.latest_evaluation

    if not result:
        st.info(
            "No evaluation is available in this session. "
            "Open Evaluation Lab and run a test case first."
        )

    else:
        diagnosis = result.get("root_cause_diagnosis", {})
        evidence = diagnosis.get("evidence", {})
        recommendations = diagnosis.get("recommendations", [])

        root_cause = diagnosis.get(
            "root_cause",
            "UNKNOWN",
        )

        status = diagnosis.get(
            "status",
            "NEEDS_REVIEW",
        )

        st.markdown("### Diagnosis Summary")

        c1, c2 = st.columns(2)

        c1.metric("Detected Signal", root_cause)
        c2.metric("Review Status", status)

        st.markdown("### Explanation")
        st.write(
            diagnosis.get(
                "explanation",
                "No explanation returned.",
            )
        )

        st.divider()

        st.markdown("### Evidence")

        relevant_found = evidence.get("relevant_chunk_found")

        if relevant_found is True:
            st.success("At least one expected relevant chunk was retrieved.")
        elif relevant_found is False:
            st.error("No expected relevant chunk was found in the retrieval results.")
        else:
            st.info("Relevant-chunk evidence is unavailable.")

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Correctness Proxy",
            str(evidence.get("answer_correctness_proxy", "N/A")),
        )

        e2.metric(
            "Completeness Proxy",
            str(evidence.get("answer_completeness_proxy", "N/A")),
        )

        e3.metric(
            "Context Support Proxy",
            str(evidence.get("context_support_proxy", "N/A")),
        )

        with st.expander("View full diagnostic evidence"):
            st.json(evidence)

        st.divider()

        st.markdown("### Recommended Actions")

        if recommendations:
            for number, recommendation in enumerate(
                recommendations,
                start=1,
            ):
                st.markdown(
                    f"**{number}.** {recommendation}"
                )
        else:
            st.info("No recommendations returned by the backend.")

        st.divider()

        st.markdown("### Retrieved Evidence")

        sources = result.get("sources", [])

        render_source_list(sources)

        st.download_button(
            "⬇️ Export Diagnosis Report",
            data=json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="ragdiagnose_diagnosis_report.json",
            mime="application/json",
            use_container_width=True,
        )

        st.warning(
            "Important: This diagnosis uses heuristic and lexical metrics. "
            "It does not independently prove hallucination, factual error, "
            "or a causal mechanism. Review the answer and source evidence."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="app-footer">
        <strong>RAGDiagnose</strong> · AI Evaluation & Observability
        <br>
        FastAPI · ChromaDB · Sentence Transformers · Groq
        <br>
        Built for inspecting and improving retrieval-augmented generation.
    </div>
    """,
    unsafe_allow_html=True,
)