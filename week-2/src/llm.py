from functools import lru_cache

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE


@lru_cache(maxsize=None)
def get_llm(model=None, temperature=None):
    """Chat model used for every generation step (rewrite, rerank, answer).

    Cached so Streamlit reruns reuse one client instead of opening a new one per keystroke.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model or GROQ_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
    )
