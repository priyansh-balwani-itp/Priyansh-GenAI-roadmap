"""Ingestion: PDF -> pages -> chunks -> embeddings -> Chroma.

Shared by the CLI (`ingest.py` at the project root) and the Streamlit uploader, so a
document indexed either way lands in the same collection with the same metadata.
"""

import tempfile
from pathlib import Path

from .config import PDF_DIR
from .loaders import load_pdf, split_documents
from .vectorstore import add_chunks, get_vectorstore


def ingest_paths(paths, vectorstore=None, chunk_size=None, chunk_overlap=None):
    """Index the given PDF files. Returns a per-file summary of pages and chunks."""
    vectorstore = vectorstore or get_vectorstore()
    summary = []
    for path in paths:
        path = Path(path)
        pages = load_pdf(path)
        chunks = split_documents(pages, chunk_size, chunk_overlap)
        add_chunks(vectorstore, chunks)
        summary.append({"source": path.name, "pages": len(pages), "chunks": len(chunks)})
    return summary


def ingest_directory(directory=None, vectorstore=None, **kwargs):
    directory = Path(directory or PDF_DIR)
    return ingest_paths(sorted(directory.glob("*.pdf")), vectorstore, **kwargs)


def ingest_uploads(uploads, vectorstore=None, **kwargs):
    """Index in-memory uploads given as `(filename, bytes)` pairs.

    PyPDFLoader reads from a path, so the bytes are staged in a temp directory that is
    removed as soon as the pages are parsed - nothing the user uploads is written into
    the repo. The original filename is preserved on the way through, because it is what
    citations display.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for name, data in uploads:
            path = Path(tmpdir) / Path(name).name
            path.write_bytes(data)
            paths.append(path)
        return ingest_paths(paths, vectorstore, **kwargs)
