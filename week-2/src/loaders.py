import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(path):
    """One Document per page, with citation-ready metadata.

    PyPDFLoader's own `page` is 0-indexed and `source` is an absolute path - neither is
    what you want to show a user, so both are normalized here rather than at display time.
    """
    path = Path(path)
    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata["source"] = path.name
        page.metadata["page"] = page.metadata.get("page", 0) + 1
    return pages


def load_pdfs(directory):
    """Load every PDF in a directory (non-recursive), sorted for reproducible ordering."""
    directory = Path(directory)
    docs = []
    for pdf in sorted(directory.glob("*.pdf")):
        docs.extend(load_pdf(pdf))
    return docs


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """Split pages into chunks sized for the embedding model's context and the LLM's prompt.

    Recursive splitting tries paragraph breaks before sentence breaks before raw
    characters, so a chunk boundary lands mid-sentence only as a last resort. The overlap
    carries a tail of the previous chunk forward, so a fact that straddles a boundary
    still appears whole in at least one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = chunk_id(chunk)
    return chunks


def chunk_id(chunk):
    """Content-addressed id, so re-ingesting the same PDF updates rows instead of duplicating them.

    Chroma upserts on id collision. Hashing the text (plus its source and page) means an
    unchanged chunk keeps its id across runs, while an edited one gets a new row.
    """
    key = f"{chunk.metadata.get('source', '')}:{chunk.metadata.get('page', '')}:{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
