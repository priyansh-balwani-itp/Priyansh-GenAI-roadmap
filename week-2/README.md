# Week 2 — RAG Chatbot

GenAI Roadmap Week 2 mini-project: a conversational Q&A system over your own PDFs, built
with **LangChain + ChromaDB + Groq**. Upload documents in the browser, ask questions, get
answers cited back to the page they came from. Follow-up questions work, because the
pipeline rewrites them against the conversation before it retrieves.

Nothing about the corpus is hardcoded — the knowledge base is whatever you upload, so the
same app serves a company handbook, a textbook, or a stack of research papers.

## Pipeline

```
PDF ──► pages ──► chunks ──► embeddings ──► ChromaDB          (ingestion, once per document)
                                                │
question + chat history ──► standalone question ─┤            (query transformation)
                                                 ▼
                                          retrieve top-k       (similarity / MMR / hybrid)
                                                 │
                                          re-rank (optional)
                                                 ▼
                                   numbered, cited context ──► Groq ──► answer
```

The rewrite step is what makes it conversational. A retriever embeds the question and
nothing else, so a follow-up like *"why is that?"* embeds to noise — the vector carries no
record of what "that" referred to. Resolving it against the history first is the difference
between a chatbot and a search box that forgets you between queries.

## Structure

- **[`app.py`](./app.py)** — Streamlit UI: PDF uploader, chat, streamed answers, expandable
  source excerpts, live retrieval settings. Presentation and session state only.
- **[`ingest.py`](./ingest.py)** — CLI for bulk-indexing a folder and for rebuilding the
  index after changing chunking or the embedding model.
- **[`rag_pipeline.ipynb`](./rag_pipeline.ipynb)** — the pipeline opened up stage by stage:
  chunk-size trade-offs, what an embedding actually is, the three retrieval strategies
  compared side by side, re-ranking, the exact prompt sent to the model, and the follow-up
  question retrieved with and without the rewrite.
- **[`prompts/`](./prompts)** — one module per prompt, holding the text and the reasoning
  behind it. No API or orchestration code.
  - `condense_question.py` — rewrites a follow-up into a standalone query.
  - `answer.py` — grounding rules and the numbered-context format that makes citation possible.
  - `rerank.py` — relevance scoring rubric for the optional re-ranker.
- **[`src/`](./src)** — the pipeline, each stage independently testable:
  - `config.py` — every tunable, read from `.env`.
  - `llm.py` / `embeddings.py` — the two models, both swappable behind one function.
  - `loaders.py` — PDF loading and chunking, with citation-ready metadata and content-hash ids.
  - `vectorstore.py` — Chroma collection: open, add, list, delete by source, reset.
  - `retrieval.py` — similarity, MMR, hybrid (BM25 + dense fused by RRF), LLM re-ranking.
  - `chain.py` — the LCEL chain and `RAGChatbot`, the object the UI drives.
  - `history.py` — conversation memory as a windowed message list.
  - `ingest.py` — PDF → chunks → Chroma, shared by the CLI and the uploader.
- **`data/pdfs/`** — optional drop folder for CLI ingestion. Gitignored: your documents are
  yours.
- **`vectorstore/`** — the persisted Chroma index. Gitignored and fully regenerable.
- **[`outputs/`](./outputs)** — saved notebook runs.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (free at
   [console.groq.com/keys](https://console.groq.com/keys)). Every other value has a working
   default.
3. `streamlit run app.py`
4. Upload PDFs in the sidebar, then ask away.

No OpenAI key needed anywhere. Embeddings run locally through fastembed — an ONNX model,
no torch, no second API account — and Groq handles generation, which is also what Week 1
used.

To index a folder instead of uploading:

```bash
python ingest.py data/pdfs
```

Re-running the ingest on unchanged files is a no-op: chunk ids are content hashes, so
Chroma upserts the same rows rather than accumulating duplicates.

## Configuration

Everything is env-driven, so you can retune without touching code. The retrieval settings
are also exposed live in the sidebar.

| Variable | Default | What it changes |
| --- | --- | --- |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Generation model. |
| `LLM_TEMPERATURE` | `0.1` | Low on purpose — answering from retrieved text should not be creative. |
| `EMBEDDING_BACKEND` | `fastembed` | `fastembed` (local ONNX) or `huggingface` (sentence-transformers). |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Changing it requires `python ingest.py --reset`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Smaller = sharper embeddings, more fragmented context. |
| `RETRIEVAL_STRATEGY` | `hybrid` | `similarity`, `mmr`, or `hybrid`. |
| `TOP_K` / `FETCH_K` | `4` / `20` | Chunks sent to the model / candidates considered first. |
| `MMR_LAMBDA` | `0.5` | 1.0 = pure relevance, 0.0 = pure diversity. |
| `USE_RERANKER` | `false` | More accurate, one extra model call per question. |
| `HISTORY_WINDOW` | `6` | Turns shown to the rewriter. |

## RAG concepts covered

| Concept | Where |
| --- | --- |
| Document loading & chunking | `src/loaders.py`, notebook §2 |
| Embedding models | `src/embeddings.py`, notebook §3 |
| Vector store operations | `src/vectorstore.py`, notebook §4 |
| Retrieval: similarity, MMR, hybrid | `src/retrieval.py`, notebook §5 |
| Re-ranking | `src/retrieval.py:rerank`, notebook §6 |
| Context injection | `prompts/answer.py`, notebook §7 |
| Query transformation | `prompts/condense_question.py`, notebook §8 |
| Conversation memory | `src/history.py`, `src/chain.py` |
| Grounding / hallucination refusal | `prompts/answer.py`, notebook §10 |

## Deploying to Hugging Face Spaces

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space), SDK
   **Streamlit**.
2. Push `app.py`, `ingest.py`, `requirements.txt`, `src/`, and `prompts/`. Leave
   `vectorstore/` and `data/` out — documents get uploaded through the UI.
3. Add `GROQ_API_KEY` under **Settings → Variables and secrets** as a *secret*, not a
   variable.
4. Prepend the Space header to the README that ships with the Space:

   ```yaml
   ---
   title: RAG Chatbot
   emoji: 📄
   sdk: streamlit
   app_file: app.py
   ---
   ```

One caveat worth knowing before you rely on it: a free Space has ephemeral storage, so the
Chroma index is wiped when the Space sleeps or restarts and visitors re-upload their PDFs.
That is acceptable for a demo — and arguably correct, since one shared persistent index
would mean every visitor could query every other visitor's documents. For a real
deployment, attach persistent storage and scope collections per user.

## Notes on the stack

The roadmap specifies LangChain + ChromaDB + OpenAI. Groq replaces OpenAI for generation
and fastembed replaces OpenAI embeddings; both sit behind `src/llm.py` and
`src/embeddings.py`, so swapping either one back is a one-line change with no effect on the
rest of the pipeline.

The chain is composed from LCEL primitives rather than a prebuilt constructor like
`create_retrieval_chain`. Same composition, but every stage stays visible and individually
testable — and it does not break when the convenience wrappers get reshuffled between
LangChain releases.
