def create_chunks(
    pages,
    chunk_size=1200,
    overlap=200
):
    """
    Pages ke text ko small chunks mein divide karta hai.
    """

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"page_{page_number}_start_{start}",
                    "text": chunk_text,
                    "page_number": page_number
                })

            next_start = end - overlap

            if next_start <= start:
                break

            start = next_start

    return chunks