# Week 1 — Prompt Engineering Portfolio

GenAI Roadmap Week 1 mini-project: 12 prompt engineering techniques applied to text
classification, summarization, code generation, and data extraction, comparing
OpenAI GPT, Anthropic Claude, and a local Ollama model on each.

## Structure

- `prompt_engineering_portfolio.ipynb` — runs each technique and displays results.
  Thin by design: it wires prompts and logic together, it doesn't contain either.
- `prompts/` — one module per technique holding the actual prompt text (and, for
  multi-step techniques, small template functions). No API or orchestration code.
- `src/` — reusable logic, independent of any specific prompt:
  - `config.py` — loads API keys / settings from `.env`.
  - `model_clients.py` — `ask_gpt`, `ask_claude`, `ask_ollama`, `compare_models`, `show`.
  - `safe_eval.py` — restricted arithmetic evaluator used by the ReAct loop.
  - `techniques.py` — multi-step procedures (ReAct, prompt chaining, reflection,
    self-consistency); each takes prompts as arguments instead of importing them,
    so this module has no knowledge of today's specific examples.
  - `persistence.py` — `save_result()`, writes each technique's run to `outputs/`.
- `outputs/` — one JSON file per technique (prompt + responses from all three
  models), written when you run the notebook — durable evidence of a run that
  doesn't require re-executing the notebook to inspect.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
3. Install [Ollama](https://ollama.ai), run `ollama serve`, then `ollama pull llama3.1`
   (or set `OLLAMA_MODEL` in `.env` to whichever model you pulled instead).
4. Open `prompt_engineering_portfolio.ipynb` from within this folder (so `prompts`
   and `src` are importable as packages) and run all cells.
