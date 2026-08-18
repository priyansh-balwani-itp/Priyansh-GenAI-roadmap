"""Streamlit UI for the RAG chatbot.

Upload PDFs in the sidebar, ask questions in the chat. Nothing is baked in - the
knowledge base is whatever the user indexes, so the same app works for a company
handbook, a textbook, or a stack of research papers.

    streamlit run app.py

Kept to presentation and session state: uploads go to `src.ingest`, questions go to
`RAGChatbot`, and this file does not know how either of them works.
"""

import uuid

import streamlit as st

from src.chain import RAGChatbot
from src.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, SESSION_ISOLATION, TOP_K
from src.history import to_messages
from src.ingest import ingest_uploads
from src.vectorstore import clear_collection, count, delete_source, get_vectorstore, list_sources

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the shared knowledge base…")
def shared_vectorstore():
    """The one on-disk collection - used only when SESSION_ISOLATION is off.

    `st.cache_resource` is shared across every user and session the server handles, which
    is precisely what "one shared corpus" means, and precisely why it must not be the
    default for a public deployment.
    """
    return get_vectorstore()


def session_vectorstore():
    """A private, in-memory collection belonging to one browser session.

    `st.session_state` is the only genuinely per-session store Streamlit offers. The
    collection name has to be unique because chromadb shares a single in-memory system
    between clients - two ephemeral stores named the same would see each other's
    documents. Nothing is written to disk, so nothing outlives the session.
    """
    if "vectorstore" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
        st.session_state["vectorstore"] = get_vectorstore(
            collection_name=f"session_{st.session_state['session_id']}", ephemeral=True
        )
    return st.session_state["vectorstore"]


def load_session():
    """Resolve this session's vectorstore and chatbot.

    The chatbot is per-session either way: it caches a BM25 index that has to match
    whatever corpus this session can actually see.
    """
    vectorstore = session_vectorstore() if SESSION_ISOLATION else shared_vectorstore()
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = RAGChatbot(vectorstore)
    return vectorstore, st.session_state["chatbot"]


def main():
    if not GROQ_API_KEY:
        st.error("`GROQ_API_KEY` is not set. Copy `.env.example` to `.env` and add your key.")
        st.stop()

    vectorstore, chatbot = load_session()
    st.session_state.setdefault("messages", [])

    render_sidebar(vectorstore, chatbot)

    st.title("📄 Chat with your documents")
    indexed = count(vectorstore)
    if indexed == 0:
        st.info("Upload one or more PDFs in the sidebar to build your knowledge base.")
    else:
        st.caption(f"{indexed} chunks indexed from {len(list_sources(vectorstore))} document(s).")

    render_history()
    handle_input(chatbot, disabled=indexed == 0)


def render_sidebar(vectorstore, chatbot):
    with st.sidebar:
        st.header("Knowledge base")
        if SESSION_ISOLATION:
            st.caption(
                "Your documents are private to this browser session, held in memory only. "
                "They are never written to the server and are gone when the session ends."
            )
        else:
            st.caption(
                ":red[Shared mode.] Every visitor reads and writes one common index - "
                "anything you upload is visible to everyone else using this app."
            )

        uploads = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True, key=upload_key()
        )
        if uploads and st.button("Index uploaded files", type="primary", width="stretch"):
            index_uploads(uploads, vectorstore, chatbot)

        sources = list_sources(vectorstore)
        if sources:
            st.subheader("Indexed documents")
            for source in sources:
                row, remove = st.columns([5, 1])
                row.write(f"📄 {source}")
                if remove.button("✕", key=f"del_{source}", help=f"Remove {source}"):
                    delete_source(vectorstore, source)
                    chatbot.invalidate()
                    st.rerun()

            if st.button("Clear everything", width="stretch"):
                clear_all(vectorstore, chatbot)

        st.divider()
        st.header("Retrieval")
        chatbot.strategy = st.selectbox(
            "Strategy",
            ["hybrid", "similarity", "mmr"],
            index=["hybrid", "similarity", "mmr"].index(chatbot.strategy),
            help=(
                "similarity: nearest neighbours by embedding. "
                "mmr: penalizes chunks that duplicate one another. "
                "hybrid: fuses embedding search with BM25 keyword search."
            ),
        )
        chatbot.k = st.slider("Chunks to retrieve", 1, 10, TOP_K)
        chatbot.use_reranker = st.toggle(
            "LLM re-ranking",
            value=chatbot.use_reranker,
            help="Retrieve a wider candidate set, then score each one against the question. "
            "More accurate, one extra model call per question.",
        )

        st.divider()
        st.caption(f"**LLM** {GROQ_MODEL} (Groq)")
        st.caption(f"**Embeddings** {EMBEDDING_MODEL}")

        if st.session_state["messages"] and st.button("New conversation", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()


def index_uploads(uploads, vectorstore, chatbot):
    with st.spinner("Reading, chunking and embedding…"):
        summary = ingest_uploads([(f.name, f.getvalue()) for f in uploads], vectorstore)
    chatbot.invalidate()
    total = sum(entry["chunks"] for entry in summary)
    st.success(f"Indexed {len(summary)} file(s), {total} chunks.")
    # Bump the widget key so the uploader empties; otherwise the same files sit in the
    # widget and can be indexed again on the next click.
    st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
    st.rerun()


def upload_key():
    return f"uploader_{st.session_state.get('upload_round', 0)}"


def clear_all(vectorstore, chatbot):
    """Empty the index in place rather than deleting it from disk - the client is still open."""
    clear_collection(vectorstore)
    chatbot.invalidate()
    st.session_state["messages"] = []
    st.rerun()


def render_history():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])


def render_sources(sources):
    """Show the exact excerpts behind an answer, so a citation can be checked rather than trusted."""
    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {source['source']} - page {source['page']}**")
            st.caption(source["excerpt"])


def handle_input(chatbot, disabled):
    question = st.chat_input(
        "Ask a question about your documents…",
        disabled=disabled,
    )
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # History excludes the question just appended - the chain takes it separately.
    history = to_messages(st.session_state["messages"][:-1])

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            documents, tokens = chatbot.stream(question, history)
        answer = st.write_stream(tokens)
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "excerpt": doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""),
            }
            for doc in documents
        ]
        if sources:
            render_sources(sources)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
