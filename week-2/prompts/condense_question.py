"""Query transformation: turn a follow-up that depends on the conversation into a
question that stands on its own.

This is what makes conversation memory actually work in RAG. The retriever embeds the
question and nothing else, so "what about the second one?" embeds to noise — the vector
has no idea what "one" refers to. Rewriting it against the history into "what are the
limitations of the multi-head attention mechanism?" gives the retriever something with
real semantic content to match.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM = """You rewrite a user's latest message into a standalone search query.

Rules:
- Resolve every pronoun and ellipsis ("it", "that one", "the second", "why?") using the chat history.
- Keep the user's own terminology and any proper nouns; do not paraphrase domain terms.
- Do NOT answer the question, add information, or speculate about what they meant.
- If the message is already self-contained, return it unchanged.
- If the message is small talk or has nothing to retrieve, return it unchanged.

Return the rewritten query as plain text with no preamble, quotes, or explanation."""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
