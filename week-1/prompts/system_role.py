SYSTEM = (
    "You are a senior technical editor at a software company. You write concise, "
    "jargon-free summaries for non-technical executives."
)

ARTICLE = (
    "Retrieval-Augmented Generation (RAG) is a technique that lets a language model answer questions "
    "using information it was never trained on. Instead of relying only on what the model memorized "
    "during training, the system first searches an external knowledge base - usually a vector database "
    "of document embeddings - for the passages most relevant to the user's query. Those passages are "
    "then inserted into the prompt as context before the model generates its answer. This approach "
    "reduces hallucination, keeps answers grounded in a specific, updatable source of truth, and avoids "
    "the cost of retraining the model whenever the underlying data changes."
)

PROMPT = f"Summarize the following in 3 bullet points for a non-technical executive:\n\n{ARTICLE}"
