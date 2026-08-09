import anthropic
import ollama
from openai import OpenAI

from .config import ANTHROPIC_API_KEY, OLLAMA_MODEL, OPENAI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def ask_gpt(prompt, system=None, model="gpt-4o-mini", temperature=0.7):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = openai_client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return response.choices[0].message.content


def ask_claude(prompt, system=None, model="claude-sonnet-5", temperature=0.7, max_tokens=1024):
    response = anthropic_client.messages.create(
        model=model,
        system=system or "",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def ask_ollama(prompt, system=None, model=None, temperature=0.7):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=model or OLLAMA_MODEL, messages=messages, options={"temperature": temperature},
    )
    return response["message"]["content"]


def compare_models(prompt, system=None):
    return {
        "GPT": ask_gpt(prompt, system=system),
        "Claude": ask_claude(prompt, system=system),
        "Ollama": ask_ollama(prompt, system=system),
    }


def show(results):
    for name, text in results.items():
        print(f"--- {name} ---\n{text}\n")
