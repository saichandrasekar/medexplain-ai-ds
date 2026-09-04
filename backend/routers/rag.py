from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.rag_service import query

router = APIRouter()


class RAGRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/rag/query")
def rag_query(payload: RAGRequest) -> Dict:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = query(payload.query, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}")

    return {
        "query": payload.query,
        "results": results,
        "total_results": len(results)
    }