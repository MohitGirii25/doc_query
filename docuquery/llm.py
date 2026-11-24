"""Prompt building and answer generation.

Set LLM_PROVIDER=gemini (GEMINI_API_KEY) or LLM_PROVIDER=openai (OPENAI_API_KEY) to use a
hosted LLM. With no provider configured, a simple extractive fallback returns the best
retrieved passage, so the project runs offline and is easy to test.
"""
import os

import requests

SYSTEM_RULES = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know."
)


def build_prompt(question, contexts):
    context_text = "\n\n".join(f"[{c['source']}] {c['text']}" for c in contexts)
    return f"{SYSTEM_RULES}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"


def _call_gemini(prompt):
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = requests.post(
        url,
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai(prompt):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_answer(question, contexts):
    if not contexts:
        return "I don't know - no relevant information was found in the documents."
    provider = os.getenv("LLM_PROVIDER", "").lower()
    prompt = build_prompt(question, contexts)
    try:
        if provider == "gemini":
            return _call_gemini(prompt)
        if provider == "openai":
            return _call_openai(prompt)
    except (requests.RequestException, KeyError) as exc:
        return f"(LLM call failed: {exc}) Best match: {contexts[0]['text']}"
    return contexts[0]["text"]  # offline extractive fallback
