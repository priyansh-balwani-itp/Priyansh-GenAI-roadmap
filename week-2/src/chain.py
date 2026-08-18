"""The conversational RAG pipeline, composed with LCEL.

    question + chat_history
        -> rewrite into a standalone question   (query transformation)
        -> retrieve chunks for it               (similarity / MMR / hybrid, optional rerank)
        -> inject them into the answer prompt   (context injection)
        -> generate a cited answer

Written with LCEL primitives rather than a prebuilt chain constructor. It is the same
composition either way, but here every stage is visible and individually testable - and
it does not break when the convenience wrappers get reshuffled between LangChain releases.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableGenerator, RunnableLambda, RunnablePassthrough

from prompts.answer import (
    ANSWER_PROMPT,
    NO_CONTEXT_MESSAGE,
    format_context,
    normalize_citations,
)
from prompts.condense_question import CONDENSE_QUESTION_PROMPT

from .config import RETRIEVAL_STRATEGY, TOP_K, USE_RERANKER
from .history import window
from .llm import get_llm
from .retrieval import build_bm25, fetch_all_documents, retrieve


def build_condense_chain(llm=None):
    """history + question -> standalone question.

    Short-circuits on the first turn: with no history there is nothing to resolve, so
    spending a model call to echo the question back would add latency for nothing.
    """
    llm = llm or get_llm()
    rewrite = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    def condense(inputs):
        history = window(inputs.get("chat_history") or [])
        if not history:
            return inputs["question"]
        return rewrite.invoke({"question": inputs["question"], "chat_history": history}).strip()

    return RunnableLambda(condense)


def build_answer_chain(llm=None):
    """question + chat_history + context documents -> cited answer.

    Windows the history here rather than at the call site so both chains bound it the
    same way and neither caller has to remember to.
    """
    llm = llm or get_llm()
    return (
        RunnablePassthrough.assign(
            context=lambda x: format_context(x["context"]),
            chat_history=lambda x: window(x.get("chat_history") or []),
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
        | RunnableGenerator(_normalize_citation_stream)
    )


def _normalize_citation_stream(tokens):
    """Rewrite citation markers mid-stream.

    A generator rather than a plain function on purpose: a plain one would be wrapped in a
    RunnableLambda, which buffers the entire response before emitting and would kill the
    token-by-token rendering in the UI. This holds back only the text following an unclosed
    marker, so a marker split across two tokens still gets rewritten whole.
    """
    buffer = ""
    for token in tokens:
        buffer += token
        head, marker, tail = buffer.rpartition("【")
        if marker and "】" not in tail:
            if head:
                yield normalize_citations(head)
            buffer = marker + tail
            continue
        yield normalize_citations(buffer)
        buffer = ""
    if buffer:
        yield normalize_citations(buffer)


def build_rag_chain(vectorstore, llm=None, **retrieval_options):
    """The full pipeline as one Runnable.

    Input:  {"question": str, "chat_history": [BaseMessage]}
    Output: the input plus "standalone_question", "context" (the retrieved Documents),
            and "answer". The intermediate values are kept rather than discarded because
            the UI cites the documents and debugging needs the rewritten query.
    """
    llm = llm or get_llm()

    def retrieve_step(inputs):
        return retrieve(vectorstore, inputs["standalone_question"], llm=llm, **retrieval_options)

    return (
        RunnablePassthrough.assign(standalone_question=build_condense_chain(llm))
        | RunnablePassthrough.assign(context=RunnableLambda(retrieve_step))
        | RunnablePassthrough.assign(answer=build_answer_chain(llm))
    )


class RAGChatbot:
    """Stateless-per-call wrapper around the chain, with the bits an interactive UI needs.

    Holds no conversation state of its own - history is passed in on every call, so one
    instance can serve several sessions. What it does hold is the BM25 index, which is
    expensive to rebuild and belongs to the corpus rather than to any one conversation.
    """

    def __init__(self, vectorstore, llm=None, strategy=None, k=None, use_reranker=None):
        self.vectorstore = vectorstore
        self.llm = llm or get_llm()
        self.strategy = strategy or RETRIEVAL_STRATEGY
        self.k = k or TOP_K
        self.use_reranker = USE_RERANKER if use_reranker is None else use_reranker
        self._bm25 = None
        self._bm25_size = None
        self.condense_chain = build_condense_chain(self.llm)
        self.answer_chain = build_answer_chain(self.llm)

    def bm25(self):
        """Lazily build the sparse index, rebuilding only when the collection size changed."""
        size = self.vectorstore._collection.count()
        if self._bm25 is None or self._bm25_size != size:
            self._bm25 = build_bm25(fetch_all_documents(self.vectorstore))
            self._bm25_size = size
        return self._bm25

    def invalidate(self):
        """Call after ingesting new documents."""
        self._bm25 = None

    def retrieve(self, question, chat_history=None):
        """Rewrite then retrieve, returning both so the caller can show its work."""
        standalone = self.condense_chain.invoke(
            {"question": question, "chat_history": chat_history or []}
        )
        options = {"strategy": self.strategy, "k": self.k, "use_reranker": self.use_reranker}
        if self.strategy == "hybrid":
            options["bm25_retriever"] = self.bm25()
        documents = retrieve(self.vectorstore, standalone, llm=self.llm, **options)
        return standalone, documents

    def answer(self, question, chat_history=None):
        standalone, documents = self.retrieve(question, chat_history)
        if not documents:
            return {"answer": NO_CONTEXT_MESSAGE, "context": [], "standalone_question": standalone}
        answer = self.answer_chain.invoke(
            {"question": question, "chat_history": chat_history or [], "context": documents}
        )
        return {"answer": answer, "context": documents, "standalone_question": standalone}

    def stream(self, question, chat_history=None):
        """Yield `(documents, token_generator)`.

        Sources come back before the first token so the UI can render citations while the
        answer is still being written - retrieval is already finished by then anyway.
        """
        standalone, documents = self.retrieve(question, chat_history)
        if not documents:
            return [], iter([NO_CONTEXT_MESSAGE])
        tokens = self.answer_chain.stream(
            {"question": question, "chat_history": chat_history or [], "context": documents}
        )
        return documents, tokens
