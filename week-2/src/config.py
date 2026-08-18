import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WEEK_DIR = Path(__file__).resolve().parent.parent

# --- Generation (Groq stands in for OpenAI; the roadmap's stack is provider-agnostic) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Check https://console.groq.com/docs/models before changing this — Groq decommissions
# models on a rolling basis, and a retired id fails with a 404 at call time, not at import.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# --- Embeddings ---
# "fastembed" runs quantized ONNX models locally (no torch, no API key, works on a
# free Hugging Face Space). "huggingface" swaps in sentence-transformers if you want
# to compare BGE/E5/MiniLM variants that fastembed doesn't ship.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid")  # similarity | mmr | hybrid
TOP_K = int(os.getenv("TOP_K", "4"))
FETCH_K = int(os.getenv("FETCH_K", "20"))  # candidates pulled before MMR / fusion / rerank
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # 1.0 = pure relevance, 0.0 = pure diversity
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

# --- Storage ---
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", WEEK_DIR / "vectorstore"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
PDF_DIR = Path(os.getenv("PDF_DIR", WEEK_DIR / "data" / "pdfs"))

# --- Memory ---
# How many prior turns are shown to the question-rewriter. Full history goes to the
# UI; only the tail is needed to resolve "it" / "that one" in a follow-up.
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "6"))
