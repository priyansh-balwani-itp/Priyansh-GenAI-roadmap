"""Index PDFs from the command line.

    python ingest.py                       # everything in data/pdfs/
    python ingest.py path/to/file.pdf ...  # specific files
    python ingest.py --reset               # wipe the store first

The Streamlit app can do this through its uploader; this exists for bulk loading a folder
and for rebuilding the index after changing CHUNK_SIZE or EMBEDDING_MODEL.
"""

import argparse
import sys
from pathlib import Path

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, PDF_DIR
from src.ingest import ingest_paths
from src.vectorstore import count, get_vectorstore, list_sources, reset


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into the Chroma vector store.")
    parser.add_argument("paths", nargs="*", help="PDF files or directories (default: data/pdfs/)")
    parser.add_argument("--reset", action="store_true", help="Delete the existing store first")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    if args.reset:
        reset()
        print(f"Reset vector store at {CHROMA_DIR}")

    pdfs = collect_pdfs(args.paths)
    if not pdfs:
        print(f"No PDFs found. Put some in {PDF_DIR}, or pass paths as arguments.")
        return 1

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunking: size={args.chunk_size} overlap={args.chunk_overlap}\n")

    vectorstore = get_vectorstore()
    summary = ingest_paths(pdfs, vectorstore, args.chunk_size, args.chunk_overlap)
    for entry in summary:
        print(f"  {entry['source']}: {entry['pages']} pages -> {entry['chunks']} chunks")

    print(f"\nCollection now holds {count(vectorstore)} chunks from {len(list_sources(vectorstore))} file(s).")
    print(f"Stored at {CHROMA_DIR}")
    return 0


def collect_pdfs(paths):
    """Expand the arguments into a sorted list of PDF files; default to the data folder."""
    if not paths:
        return sorted(PDF_DIR.glob("*.pdf"))
    pdfs = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            pdfs.append(path)
        else:
            print(f"Skipping {path} (not a PDF or directory)")
    return pdfs


if __name__ == "__main__":
    sys.exit(main())
