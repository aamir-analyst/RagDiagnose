from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_db"

# Lightweight aur reliable embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


def store_chunks(chunks, document_name):
    """
    Chunks ko embeddings ke saath ChromaDB mein save karta hai.
    """

    if not chunks:
        return 0

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = [
        f"{document_name}_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "document_name": document_name,
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    # Same file dobara upload hone par duplicate error avoid karne ke liye
    try:
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as error:
        raise RuntimeError(
            f"Chunks store karte waqt error: {error}"
        )

    return len(chunks)