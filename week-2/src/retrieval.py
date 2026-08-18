"""Retrieval strategies.

Every function here takes a query and returns a ranked list of Documents, so the chain
can swap strategies without knowing how any of them work.
"""

import json
import re

from langchain_community.retrievers import BM25Retriever

from .config import FETCH_K, MMR_LAMBDA, RETRIEVAL_STRATEGY, TOP_K, USE_RERANKER


def similarity_search(vectorstore, query, k=None):
    """Plain nearest-neighbour search over the embeddings.

    The baseline, and the failure mode worth knowing: when a document repeats itself,
    the top k are often k near-copies of the same passage, so the context window is
    spent on one fact.
    """
    return vectorstore.similarity_search(query, k=k or TOP_K)


def mmr_search(vectorstore, query, k=None, fetch_k=None, lambda_mult=None):
    """Maximal Marginal Relevance — relevance discounted by redundancy.

    Pulls `fetch_k` candidates, then picks `k` of them greedily: each pick maximizes
    lambda * (similarity to query) - (1 - lambda) * (max similarity to anything already
    picked). Lower lambda buys diversity at the cost of relevance. Use it when a question
    spans several parts of a document.
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k or TOP_K,
        fetch_k=fetch_k or FETCH_K,
        lambda_mult=MMR_LAMBDA if lambda_mult is None else lambda_mult,
    )


def hybrid_search(vectorstore, query, k=None, fetch_k=None, bm25_retriever=None):
    """Dense + sparse retrieval, combined by reciprocal rank fusion.

    Embeddings match meaning but miss rare literal tokens — an error code, a part number,
    an unusual surname all embed to something generic. BM25 matches those exactly but is
    blind to paraphrase. Running both and fusing the rankings covers each one's blind spot.
    """
    k = k or TOP_K
    fetch_k = fetch_k or FETCH_K

    dense = vectorstore.similarity_search(query, k=fetch_k)
    if bm25_retriever is None:
        bm25_retriever = build_bm25(fetch_all_documents(vectorstore), k=fetch_k)
    sparse = bm25_retriever.invoke(query) if bm25_retriever else []

    return reciprocal_rank_fusion([dense, sparse])[:k]


def reciprocal_rank_fusion(rankings, k=60):
    """Merge ranked lists by summing 1 / (k + rank) across them.

    Rank-based rather than score-based on purpose: a cosine similarity and a BM25 score
    are not on a comparable scale, and normalizing them is guesswork. Positions are
    comparable. The constant k=60 is the value from the original RRF paper; it damps the
    influence of the very top position so one retriever cannot single-handedly decide the
    result.
    """
    scores = {}
    documents = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.metadata.get("chunk_id") or doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            documents.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [documents[key] for key in ordered]


def fetch_all_documents(vectorstore):
    """Pull the whole collection back out of Chroma as Documents.

    BM25 is an in-memory index over the full corpus — unlike the vector store, it cannot
    be queried incrementally. Fine at this scale; a larger corpus would use a search
    engine (Elasticsearch, OpenSearch) for the sparse half instead.
    """
    from langchain_core.documents import Document

    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def build_bm25(documents, k=None):
    if not documents:
        return None
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k or FETCH_K
    return retriever


def rerank(llm, query, documents, top_n=None):
    """Re-score candidates with the LLM and keep the best `top_n`.

    Degrades to the original order if the model returns something unparseable — a
    malformed score list is not a reason to fail a user's question.
    """
    from prompts.rerank import RERANK_PROMPT, format_candidates

    top_n = top_n or TOP_K
    if len(documents) <= top_n:
        return documents

    response = (RERANK_PROMPT | llm).invoke(
        {"query": query, "candidates": format_candidates(documents)}
    )
    scores = _parse_scores(response.content, len(documents))
    if not scores:
        return documents[:top_n]

    ranked = sorted(range(len(documents)), key=lambda i: scores.get(i + 1, 0), reverse=True)
    return [documents[i] for i in ranked[:top_n]]


def _parse_scores(text, count):
    """Extract {id: score} from the model's reply, tolerating markdown fences and stray prose."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return {}
    try:
        entries = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {
        int(e["id"]): float(e["score"])
        for e in entries
        if isinstance(e, dict) and "id" in e and "score" in e and 1 <= int(e["id"]) <= count
    }


def retrieve(vectorstore, query, strategy=None, k=None, llm=None, use_reranker=None, **kwargs):
    """Single entry point the chain calls; picks a strategy and optionally re-ranks.

    With re-ranking on, retrieval widens to `fetch_k` first and the re-ranker narrows
    back down to `k` — otherwise there would be nothing for it to choose between.
    """
    strategy = (strategy or RETRIEVAL_STRATEGY).lower()
    k = k or TOP_K
    use_reranker = USE_RERANKER if use_reranker is None else use_reranker
    wants_rerank = use_reranker and llm is not None
    fetch = kwargs.pop("fetch_k", FETCH_K)
    retrieve_k = fetch if wants_rerank else k

    if strategy == "similarity":
        documents = similarity_search(vectorstore, query, k=retrieve_k)
    elif strategy == "mmr":
        documents = mmr_search(vectorstore, query, k=retrieve_k, fetch_k=fetch, **kwargs)
    elif strategy == "hybrid":
        documents = hybrid_search(vectorstore, query, k=retrieve_k, fetch_k=fetch, **kwargs)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'similarity', 'mmr', or 'hybrid'.")

    return rerank(llm, query, documents, top_n=k) if wants_rerank else documents
