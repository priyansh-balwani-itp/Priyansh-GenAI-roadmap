import time

import ollama
from groq import Groq
from openai import APIStatusError, OpenAI

from .config import (
    AIRFORCE_API_KEY,
    AIRFORCE_BASE_URL,
    AIRFORCE_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_MODEL,
)

groq_client = Groq(api_key=GROQ_API_KEY)
# max_retries=0: we handle 429 backoff ourselves, using the wait time airforce reports,
# instead of the SDK's default retry schedule which is too short for a 1-request/minute plan.
airforce_client = OpenAI(api_key=AIRFORCE_API_KEY, base_url=AIRFORCE_BASE_URL, max_retries=0)


def _ask_openai_compatible(client, model, prompt, system=None, temperature=0.7, max_attempts=4):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            return response.choices[0].message.content
        except APIStatusError as e:
            if e.status_code != 429 or attempt == max_attempts - 1:
                raise
            wait_seconds = 30
            try:
                wait_seconds = e.response.json()["error"]["retry_after_seconds"]
            except Exception:
                pass
            time.sleep(wait_seconds + 1)


def ask_groq(prompt, system=None, model=None, temperature=0.7):
    return _ask_openai_compatible(groq_client, model or GROQ_MODEL, prompt, system, temperature)


def ask_airforce(prompt, system=None, model=None, temperature=0.7):
    return _ask_openai_compatible(airforce_client, model or AIRFORCE_MODEL, prompt, system, temperature)


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
        "Groq": ask_groq(prompt, system=system),
        "Airforce": ask_airforce(prompt, system=system),
        "Ollama": ask_ollama(prompt, system=system),
    }


def show(results):
    for name, text in results.items():
        print(f"--- {name} ---\n{text}\n")
