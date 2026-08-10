# Cross-Model Comparison — Findings

Generated from the results in this folder (`01_zero_shot.json` … `12_cross_model.json`),
produced by running [`prompt_engineering_portfolio.ipynb`](../prompt_engineering_portfolio.ipynb)
against three backends:

| Provider | Model |
|---|---|
| Groq | `llama-3.3-70b-versatile` |
| api.airforce | `llama-4-scout-17b-16e-instruct` |
| Ollama (local) | `llama3.1` (8B) |

## Summary table

| # | Technique | All three agree on the answer? | Notable difference |
|---|-----------|:---:|---|
| 01 | Zero-Shot | Yes ("Mixed") | None — clean-cut case, no disagreement |
| 02 | Few-Shot | Yes (Billing) | Ollama matched the examples' bare single-word format; Groq and Airforce added unrequested explanatory prose |
| 03 | Chain-of-Thought | Yes ($37.17) | Nearly identical step-by-step breakdowns across all three |
| 04 | System Role | Yes (same 3 points) | All respected persona + bullet constraint; content overlapped heavily |
| 05 | Structured Output | **No** | Groq/Airforce returned clean JSON; Ollama wrapped its JSON in ` ``` ` fences despite being told not to, breaking `json.loads` |
| 06 | ReAct | N/A — technique didn't execute | See [limitation](#limitation-react-never-actually-called-the-tool) below |
| 07 | Prompt Chaining | N/A (single model, Groq) | Summary → tweet both landed on-topic and within the length constraint |
| 08 | Code Generation | Yes (all correct) | Ollama's prose claimed its docstring "includes three test cases," but the generated docstring had none — a self-report that didn't match its own code |
| 09 | Reflection | N/A (single model, Groq) | Critique pass caught real gaps (no input validation, no error handling) the first draft missed, and the revision fixed both |
| 10 | Persona | Yes (all in-character) | All three converged on the same infomercial tropes ("BUT WAIT, THERE'S MORE") independently |
| 11 | Self-Consistency | Yes (5/5 samples) | Zero variance across 5 samples on this question — no disagreement to resolve |
| 12 | Cross-Model | Yes (same explanation) | All three honored "exactly 2 sentences," and Airforce/Ollama both independently reached for the same "spotlight" metaphor |

## Where the three backends agree

Classification (01, 02), arithmetic reasoning (03), factual explanation (12), and persona
adoption (10) were **consistent across all three backends** — correct answers, similar
content, no real quality gap between the 70B hosted model, the 17B hosted model, and the
8B local model. For tasks like these, provider/model size didn't matter much for correctness.

## Where they diverge

- **Format compliance is where the local model struggled most.** Ollama (8B, local) was the
  only backend that broke the structured-output contract (05) by adding markdown fences after
  being explicitly told not to. It also produced the most literally-formatted few-shot answer (02),
  so format-following isn't simply "smaller model = worse" — it's inconsistent across constraint types.
- **Verbosity.** Groq and Airforce (the two larger hosted models) consistently added explanatory
  framing beyond what was asked (e.g. 02's category classification), while Ollama tended toward
  more literal, terser answers when it did follow format — except when it didn't (05).
- **Self-reported accuracy isn't reliable.** In 08, Ollama's explanation text asserted its code
  included three doctest examples; the actual generated code didn't. Treat a model's description
  of its own output as unverified, not as a substitute for checking the output itself.

## Which backend to reach for

| Use case | Recommendation | Why |
|---|---|---|
| Strict, machine-parsed output (JSON, schemas) | **Groq or api.airforce** | Both returned clean, directly-parseable JSON in 05; Ollama needs a fence-stripping step first, or the response silently fails to parse |
| Quick classification / simple factual Q&A | **Any of the three** | 01, 02, 12 were correct and consistent everywhere — pick by cost/latency/privacy, not quality |
| Code generation | **Any of the three, but verify manually** | All three produced correct, working code, but Ollama's own description of its output (08) didn't match what it actually generated — don't trust a model's self-report as a substitute for reading the code |
| Local-only / private / no per-call cost | **Ollama** | Only backend that needs no network call or API key; trade-off is slower (CPU-bound) and, per 05, the least reliable at strict format instructions |
| Fast iteration during development | **Groq** | Hosted, no rate-limit friction observed in this run, and used as the default `ask_fn` for every multi-step technique (chaining, reflection, self-consistency, ReAct) |
| Agentic / tool-calling workflows | **None confirmed yet** | 06 shows the ReAct loop didn't actually invoke the tool with Groq — this needs the `run_react` fix (see below) before any model can be judged on this axis |
| Anything production-facing | **Not api.airforce as-is** | Its free tier is throttled to 1 request/minute (hit and handled in this run — see `src/model_clients.py`'s retry logic) and, per the earlier discussion, several of its listed models (`*-rp`, `unmoderated-gpt`) suggest it isn't simply reselling authorized access — treat it as a demo/comparison data point, not a dependency |

## Limitation: ReAct never actually called the tool

Technique 06 is the one result that didn't work as designed. The prompt instructs the model to
stop right after an `Action:` line and wait for an `Observation:`, but Groq's completion ignored
that and generated the entire Thought → Action → Observation → Final Answer sequence in a single
response — including fake, empty `Observation:` lines and a literal `Final Answer: the answer`
that was never actually computed. Because `src/techniques.py`'s `run_react` only checks for the
string `"Final Answer:"` to decide the loop is done, it accepted this one-shot completion instead
of catching the model's non-compliance and forcing a real, single-step exchange.

**Takeaway:** ReAct's reliability depends on the model actually honoring turn-taking instructions,
not just the prompt template being well-written. A prompt can be correct and the technique can
still fail if the model doesn't cooperate with the intended protocol.

## Overall takeaways

- **Most reliable techniques observed:** zero/few-shot classification, chain-of-thought arithmetic,
  system-role summarization, persona rewriting, self-consistency, prompt chaining, reflection.
- **Least reliable in this run:** structured JSON output on the local model, and ReAct across the
  board (the failure mode wasn't provider-specific — Groq is the one used here, but nothing about
  the fix would be Groq-specific either).
- **Cheapest technique to add for a real reliability win:** structured output would benefit from
  either a stricter system instruction or a fence-stripping step before `json.loads`, given how
  cleanly Groq/Airforce complied without either.
