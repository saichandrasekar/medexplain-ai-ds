from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.agent_service import analyze_report, chat

router = APIRouter()


class AnalyzeRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@router.post("/agent/analyze")
def analyze(payload: AnalyzeRequest) -> Dict:
    try:
        result = analyze_report(payload.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.post("/agent/chat")
def chat_endpoint(payload: ChatRequest) -> Dict:
    try:
        result = chat(payload.question, payload.history)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")