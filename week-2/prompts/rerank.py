"""Re-ranking: score a candidate set for relevance before it reaches the answer prompt.

Retrieval optimizes for recall - pull 20 candidates so the right chunk is somewhere in
there. Generation wants precision, because irrelevant context measurably degrades the
answer. A re-ranker bridges the two: it reads the query and each candidate together
(the embedding never did - it encoded them separately) and keeps only the top few.

Production systems use a cross-encoder for this. Scoring with the LLM costs an extra
call but needs no additional model, which keeps this project to one provider.
"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You score how well each document excerpt answers a specific query.

Score each excerpt 0-10:
- 0-2: unrelated to the query.
- 3-5: same broad topic, but does not address the query.
- 6-8: contains part of the answer.
- 9-10: directly and completely answers the query.

Judge only whether the excerpt answers THIS query. Ignore how well written it is.

Return a JSON array of objects with keys "id" and "score", one per excerpt, and nothing else.
Example: [{{"id": 1, "score": 9}}, {{"id": 2, "score": 3}}]"""

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Query: {query}\n\nExcerpts:\n{candidates}"),
    ]
)


def format_candidates(documents, max_chars=600):
    """Truncate candidates before scoring - the judgment needs the gist, not the full chunk."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        text = doc.page_content[:max_chars].replace("\n", " ")
        blocks.append(f"[{i}] {text}")
    return "\n\n".join(blocks)
