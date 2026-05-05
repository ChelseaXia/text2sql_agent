"""Small DeepSeek API client used by the experiment pipelines."""

import hashlib
import json
import os

from text2sql.config import PROJECT_ROOT

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "llm"
DEFAULT_USE_CACHE = True
DEFAULT_SEED = 42


class LLMClientError(RuntimeError):
    pass


def get_api_key(api_key=None):
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise LLMClientError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in the environment.")
    return key


def _cache_key(model, prompt):
    payload = f"{model}\n{prompt}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _cache_path(model, prompt, cache_dir):
    return cache_dir / f"{_cache_key(model, prompt)}.json"


def call_llm(
    prompt,
    model=DEFAULT_MODEL,
    api_key=None,
    temperature=0.0,
    top_p=1.0,
    seed=DEFAULT_SEED,
    timeout=60,
    use_cache=DEFAULT_USE_CACHE,
    cache_dir=DEFAULT_CACHE_DIR,
):
    import requests

    cache_file = _cache_path(model, prompt, cache_dir)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached["raw_response"]

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
        "top_p": top_p,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400 and "seed" in payload:
        retry_payload = dict(payload)
        retry_payload.pop("seed", None)
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=retry_payload, timeout=timeout)

    if response.status_code >= 400:
        raise LLMClientError(f"DeepSeek API error {response.status_code}: {response.text}")

    data = response.json()
    try:
        raw_response = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError(f"Unexpected DeepSeek response: {json.dumps(data)[:1000]}") from exc

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "model": model,
                    "prompt_md5": cache_file.stem,
                    "raw_response": raw_response,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return raw_response
