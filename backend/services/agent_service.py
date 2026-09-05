import pandas as pd
from backend.services.rag_service import query as rag_query
from backend.services.prompt_service import build_analysis_prompt, build_chat_prompt
from backend.services.openrouter_service import call_llm
from backend.routers.predict import predict
from backend.services.rag_service import query


def _run_predict(parameters: dict) -> tuple[float, str, str]:
    response = predict(parameters)
    return response['risk_score'], response['risk_level'], response['model_version']

def _collect_rag_context(parameters: dict) -> list[dict]:
    search_terms = [
        "cardiovascular risk factors",
        "cholesterol blood pressure heart disease",
        "blood glucose diabetes risk",
    ]
    all_results = []
    seen_texts = set()

    for term in search_terms:
        results = rag_query(term, top_k=2)
        for r in results:
            if r["text"] not in seen_texts:
                all_results.append(r)
                seen_texts.add(r["text"])

    return all_results[:6]


def analyze_report(parameters: dict) -> dict:
    risk_score, risk_level, model_version = _run_predict(parameters)
    rag_results = _collect_rag_context(parameters)
    messages = build_analysis_prompt(risk_score, risk_level, parameters, rag_results)
    report = call_llm(messages)

    citations = list({
        f"{r['source']} p.{r['page']}" for r in rag_results
    })

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "report": report,
        "citations": citations,
        "model_version": model_version
    }


def chat(question: str, history: list[dict]) -> dict:
    rag_results = rag_query(question, top_k=3)
    messages = build_chat_prompt(question, history, rag_results)
    response = call_llm(messages, temperature=0.5)

    sources = list({
        f"{r['source']} p.{r['page']}" for r in rag_results
    })

    return {
        "response": response,
        "sources": sources
    }