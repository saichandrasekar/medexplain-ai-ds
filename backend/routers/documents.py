import shutil
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.services.ocr_service import extract_chunks
from backend.services.rag_service import ingest_documents, list_documents

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "rag_docs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict:
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    dest_path = UPLOAD_DIR / file.filename
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks = extract_chunks(str(dest_path))
        count = ingest_documents(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "filename": file.filename,
        "chunks_stored": count,
        "status": "success"
    }


@router.get("/documents/list")
def list_docs() -> Dict:
    docs = list_documents()
    return {
        "documents": docs,
        "total": len(docs)
    }