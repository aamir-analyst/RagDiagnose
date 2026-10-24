import os
import streamlit as st
from dotenv import load_dotenv

# 1. API Key & Environment Setup
load_dotenv() # Local testing ke liye

try:
    # Streamlit Cloud (Deployment) ke liye secrets se API key lena
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    # Local system ke liye .env se API key lena
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning("⚠️ API Key missing! Kripaya local .env ya Streamlit Secrets mein OPENAI_API_KEY set karein.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# 2. Linux-Friendly Relative Paths (Hardcoded C:/ drive hata diya gaya hai)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "backend", "chroma_db")
DOCS_PATH = os.path.join(BASE_DIR, "backend", "data", "documents")

# Cloud par folder na hone par automatically create karna
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(DOCS_PATH, exist_ok=True)

# 3. Streamlit UI Design
st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("📊 RAG Evaluation & Diagnose Dashboard")
st.markdown("Upload your documents to analyze completeness, factual accuracy, and hallucinations.")

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)

# File Upload Section
uploaded_file = st.file_uploader("Upload Document (PDF/TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    file_path = os.path.join(DOCS_PATH, uploaded_file.name)
    
    # File ko backend/data/documents folder mein save karna
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ '{uploaded_file.name}' successfully uploaded!")
    
    # Process Button
    if st.button("Run RAG Pipeline"):
        with st.spinner("Processing document and generating evaluation metrics..."):
            try:
                # ---------------------------------------------------------
                # YAHAN AAPKA ASLI BACKEND FUNCTION CALL HOGA
                # Example: 
                # from backend.rag_pipeline import run_evaluation
                # results = run_evaluation(file_path, DB_PATH)
                # st.write(results)
                # ---------------------------------------------------------
                
                # Yeh sirf testing ke liye ek dummy output hai
                st.info("System Ready! Apna backend logic upar comment ki jagah par call karein.")
            except Exception as e:
                st.error(f"Error during processing: {e}")
