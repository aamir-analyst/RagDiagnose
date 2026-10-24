from ingestion.embedder import (
    collection,
    embedding_model
)


def retrieve_documents(query: str, top_k: int = 5):
    """
    Query ke basis par relevant chunks retrieve karta hai.
    """

    query = query.strip()

    if not query:
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, text in enumerate(documents):
        metadata = metadatas[index]
        distance = distances[index]

        # Cosine distance ko similarity score mein convert kar rahe hain
        similarity = round(1 - distance, 4)

        retrieved_chunks.append({
            "text": text,
            "similarity": similarity,
            "distance": round(distance, 4),
            "metadata": metadata
        })

    return retrieved_chunks