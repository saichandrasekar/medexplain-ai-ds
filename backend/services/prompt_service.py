def build_analysis_prompt(
    risk_score: float,
    risk_level: str,
    parameters: dict,
    rag_results: list[dict]
) -> list[dict]:

    rag_context = ""
    for i, r in enumerate(rag_results, start=1):
        rag_context += f"\n[{i}] Source: {r['source']}, Page {r['page']}\n{r['text']}\n"

    system_prompt = """You are a medical report analyst assistant.
Your job is to explain health report findings in clear, plain language.

STRICT RULES:
- Use ONLY the provided context passages — do not add outside knowledge
- Cite every claim using [Source: filename, page N] format
- Never diagnose — only explain and contextualize findings
- Always include a disclaimer at the end
- Structure your response exactly as specified"""

    user_prompt = f"""Patient health report analysis:

RISK SCORE: {risk_score}/10 — {risk_level.upper()} RISK

HEALTH PARAMETERS:
{chr(10).join([f"- {k}: {v}" for k, v in parameters.items()])}

MEDICAL REFERENCE CONTEXT:
{rag_context}

Provide a structured analysis using this exact format:

## Overall Health Risk
[Interpret the risk score in plain language]

## Key Findings
[Explain the most significant parameters using the context above]

## What This Means
[Plain language summary of what these results indicate]

## Sources Cited
[List all sources you referenced]

## Disclaimer
This report is not a substitute for professional medical advice. Please consult a qualified healthcare provider."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def build_chat_prompt(
    question: str,
    history: list[dict],
    rag_results: list[dict]
) -> list[dict]:

    rag_context = ""
    for i, r in enumerate(rag_results, start=1):
        rag_context += f"\n[{i}] Source: {r['source']}, Page {r['page']}\n{r['text']}\n"

    system_prompt = """You are a helpful medical information assistant.
Answer questions using ONLY the provided context.
Always cite your sources using [Source: filename, page N] format.
Never diagnose. Always recommend consulting a doctor for personal medical advice."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"""Question: {question}

Medical Reference Context:
{rag_context}

Answer using only the context above and cite your sources."""
    })

    return messages