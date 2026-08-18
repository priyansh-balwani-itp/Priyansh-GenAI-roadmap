import shutil

from langchain_chroma import Chroma

from .config import CHROMA_DIR, COLLECTION_NAME
from .embeddings import get_embeddings


def get_vectorstore(persist_directory=None, collection_name=None, embeddings=None):
    """Open (or create) the persistent Chroma collection.

    Chroma writes to disk, so an ingest survives a restart of the app — the Streamlit UI
    reconnects to whatever was indexed previously instead of re-embedding on every launch.
    """
    return Chroma(
        collection_name=collection_name or COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_directory or CHROMA_DIR),
    )


def add_chunks(vectorstore, chunks):
    """Index chunks, keyed by their content hash so repeated ingests upsert rather than duplicate."""
    if not chunks:
        return 0
    vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(chunks)


def count(vectorstore):
    return vectorstore._collection.count()


def list_sources(vectorstore):
    """Distinct filenames currently indexed — what the UI shows as the knowledge base."""
    metadatas = vectorstore._collection.get(include=["metadatas"])["metadatas"]
    return sorted({m.get("source", "unknown") for m in metadatas})


def delete_source(vectorstore, source):
    """Drop every chunk that came from one file, so a document can be removed without a full rebuild."""
    vectorstore._collection.delete(where={"source": source})


def clear_collection(vectorstore):
    """Empty an open collection in place.

    Use this instead of `reset()` whenever a Chroma client is live. On Windows the client
    keeps a handle on its SQLite file, so deleting the directory underneath it half-succeeds
    and leaves a store that fails on the next write.
    """
    ids = vectorstore._collection.get(include=[])["ids"]
    if ids:
        vectorstore._collection.delete(ids=ids)
    return len(ids)


def reset(persist_directory=None):
    """Delete the store from disk. Only safe when no client has it open — see `clear_collection`."""
    directory = persist_directory or CHROMA_DIR
    shutil.rmtree(directory, ignore_errors=True)
