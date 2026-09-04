import pdfplumber
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_chunks(pdf_path: str) -> list[dict]:
    chunks = []
    path = Path(pdf_path)

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            text = _clean_text(text)
            page_chunks = _chunk_text(text, page_num, path.name)
            chunks.extend(page_chunks)

    return chunks


def _clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 3:
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def _chunk_text(text: str, page_num: int, filename: str) -> list[dict]:
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "source_filename": filename,
                "page_number": page_num,
                "chunk_index": chunk_index
            })
            chunk_index += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks