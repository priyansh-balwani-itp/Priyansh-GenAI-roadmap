"""Context injection: how retrieved chunks are structured in the prompt.

Two choices here matter more than the wording. First, each chunk is wrapped in a
numbered block carrying its filename and page, which gives the model a stable handle to
cite — asking for citations without giving it identifiers just produces invented ones.
Second, the grounding rule is stated as a hard constraint before the context rather than
after it, so it is not buried under whatever the retriever happened to return.
"""

import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM = """You are a document question-answering assistant. You answer strictly from the \
excerpts provided below, which were retrieved from the user's own documents.

Grounding rules:
- Use only the excerpts. Do not use outside knowledge, even if you are confident it is correct.
- Cite the excerpts you used inline as [1], [2], etc., matching the numbers below. Use square \
brackets and the excerpt number only — never any other citation syntax, and never line or \
character ranges.
- If the excerpts do not contain the answer, say so plainly and name what is missing. \
Do not guess, and do not pad the answer with related-but-unasked-for material.
- If the excerpts disagree with each other, surface the disagreement instead of picking one.
- Quote exact figures, names, and dates rather than rounding or paraphrasing them.

Answer in the user's language, in prose or a short list — whatever fits the question. Be direct.

Retrieved excerpts:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

NO_CONTEXT_MESSAGE = (
    "I could not find anything relevant in the indexed documents for that question. "
    "Try rephrasing it, or upload a document that covers the topic."
)


def format_context(documents):
    """Render retrieved documents into the numbered blocks the system prompt refers to."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] {source} (page {page}):\n{doc.page_content}")
    return "\n\n".join(blocks)


# Some models — gpt-oss among them — were trained to cite with 【n】 or 【n†L1-L4】 and
# reach for it regardless of what the prompt asks for. The number is right, so rewriting
# the delimiters is enough; no need to fight the model over punctuation it cannot unlearn.
CITATION_PATTERN = re.compile(r"【(\d+)(?:†[^】]*)?】")


def normalize_citations(text):
    return CITATION_PATTERN.sub(r"[\1]", text)
