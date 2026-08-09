import re
from collections import Counter

from .model_clients import ask_gpt
from .safe_eval import safe_eval


def run_react(question, instructions_template, ask_fn=ask_gpt, max_steps=4):
    transcript = instructions_template.format(question=question)
    for _ in range(max_steps):
        response = ask_fn(transcript)
        transcript += response
        if "Final Answer:" in response:
            break
        match = re.search(r"Action:\s*calculator\[(.*?)\]", response)
        if not match:
            break
        try:
            result = safe_eval(match.group(1))
        except Exception as e:
            result = f"Error: {e}"
        transcript += f"\nObservation: {result}\n"
    return transcript


def run_prompt_chain(source_text, summary_prompt_fn, tweet_prompt_fn, ask_fn=ask_gpt):
    summary = ask_fn(summary_prompt_fn(source_text))
    tweet = ask_fn(tweet_prompt_fn(summary))
    return {"summary": summary, "tweet": tweet}


def run_reflection(code_prompt, critique_prompt_fn, ask_fn=ask_gpt):
    first_pass = ask_fn(code_prompt)
    critique = ask_fn(critique_prompt_fn(first_pass))
    return {"first_pass": first_pass, "critique": critique}


def run_self_consistency(prompt, ask_fn=ask_gpt, n=5):
    samples = [ask_fn(prompt) for _ in range(n)]
    answers = [m.group(1).strip() for m in (re.search(r"Answer:\s*(.+)", s) for s in samples) if m]
    counts = Counter(answers)
    majority = counts.most_common(1)[0][0] if answers else "n/a"
    return {"samples": samples, "answers": answers, "distribution": dict(counts), "majority": majority}
