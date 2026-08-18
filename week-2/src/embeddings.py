from functools import lru_cache

from .config import EMBEDDING_BACKEND, EMBEDDING_MODEL


@lru_cache(maxsize=None)
def get_embeddings(backend=None, model=None):
    """Embedding model, swappable so you can compare retrieval quality across encoders.

    Both backends return LangChain `Embeddings`, so everything downstream — Chroma,
    the retrievers, the chain — is unaffected by which one you pick. The only rule is
    that a collection must be queried with the same model it was indexed with; changing
    `EMBEDDING_MODEL` means re-running the ingest.
    """
    backend = (backend or EMBEDDING_BACKEND).lower()
    model = model or EMBEDDING_MODEL

    if backend == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=model)

    if backend == "huggingface":
        # Needs `pip install langchain-huggingface sentence-transformers` (pulls torch).
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})

    raise ValueError(f"Unknown EMBEDDING_BACKEND {backend!r}; expected 'fastembed' or 'huggingface'.")
