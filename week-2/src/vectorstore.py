import shutil

from langchain_chroma import Chroma

from .config import CHROMA_DIR, COLLECTION_NAME
from .embeddings import get_embeddings


def get_vectorstore(persist_directory=None, collection_name=None, embeddings=None, ephemeral=False):
    """Open (or create) a Chroma collection.

    Persistent by default: the ingest survives a restart, which is what the CLI and the
    notebook want. `ephemeral=True` keeps the collection in memory instead, so it touches
    no disk and cannot outlive the process - that is what the web UI uses to give each
    visitor a private index.

    Note that `collection_name` is what actually separates two ephemeral stores. chromadb
    shares one in-memory system between clients, so two ephemeral collections with the
    same name see each other's documents.
    """
    return Chroma(
        collection_name=collection_name or COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=None if ephemeral else str(persist_directory or CHROMA_DIR),
    )


def drop_collection(vectorstore):
    """Delete the collection outright, freeing the memory an ephemeral one holds."""
    try:
        vectorstore._client.delete_collection(vectorstore._collection.name)
    except Exception:
        # Already gone, or the client is shutting down - either way there is nothing to free.
        pass


def add_chunks(vectorstore, chunks):
    """Index chunks, keyed by their content hash so repeated ingests upsert rather than duplicate."""
    if not chunks:
        return 0
    vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(chunks)


def count(vectorstore):
    return vectorstore._collection.count()


def list_sources(vectorstore):
    """Distinct filenames currently indexed - what the UI shows as the knowledge base."""
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
    """Delete the store from disk. Only safe when no client has it open - see `clear_collection`."""
    directory = persist_directory or CHROMA_DIR
    shutil.rmtree(directory, ignore_errors=True)
