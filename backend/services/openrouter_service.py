from openai import OpenAI
from backend.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client

def call_llm(messages: list[dict], temperature: float = 0.3) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content