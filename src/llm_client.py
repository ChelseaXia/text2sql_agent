"""Small DeepSeek API client used by the naive baseline."""

import json
import os
from typing import Optional


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


class LLMClientError(RuntimeError):
    """Raised when the LLM request cannot be completed."""


def get_api_key(api_key: Optional[str] = None) -> str:
    """Read the DeepSeek API key from an argument or environment variable."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise LLMClientError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in the environment.")
    return key


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """Send a prompt to DeepSeek and return the raw assistant response text."""
    import requests

    headers = {
        "Authorization": f"Bearer {get_api_key(api_key)}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text-to-SQL assistant. Return only one SQLite SQL query. "
                    "Do not include explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise LLMClientError(f"DeepSeek API error {response.status_code}: {response.text}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError(f"Unexpected DeepSeek response: {json.dumps(data)[:1000]}") from exc


if __name__ == "__main__":
    print(call_llm("SELECT 1;"))
