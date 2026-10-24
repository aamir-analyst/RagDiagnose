from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str):
    """
    PDF ke har page ka text extract karta hai.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages