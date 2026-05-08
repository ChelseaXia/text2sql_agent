"""LLM-decided tool selection policy for the Text2SQL iterative agent."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import requests

from text2sql.config import PROJECT_ROOT
from text2sql.llm import DEFAULT_MODEL, DEFAULT_SEED, DEEPSEEK_API_URL, LLMClientError, get_api_key
from text2sql.agents.tool_schemas import format_tools_for_prompt, validate_tool_call

DEFAULT_TOOL_POLICY_CACHE_DIR = PROJECT_ROOT / "cache" / "llm_tool_policy"


def _cache_key(model: str, system_prompt: str, prompt: str) -> str:
    payload = f"{model}\n{system_prompt}\n{prompt}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _cache_path(model: str, system_prompt: str, prompt: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_cache_key(model, system_prompt, prompt)}.json"


def call_tool_policy_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = DEFAULT_SEED,
    timeout: int = 60,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_TOOL_POLICY_CACHE_DIR,
) -> str:
    """Call the LLM with a JSON-only system prompt for tool selection."""
    system_prompt = (
        "You are a Text2SQL tool-selection controller. Return only one strict JSON object. "
        "Do not return SQL unless it is inside the JSON args."
    )
    cache_file = _cache_path(model, system_prompt, prompt, cache_dir)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached["raw_response"]

    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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


def parse_tool_call_json(raw_text: str) -> dict[str, Any]:
    """Parse a strict tool-call JSON object, tolerating fenced JSON wrappers."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Tool call must be a JSON object.")
    return parsed


def build_autonomous_tool_prompt(
    sample: dict[str, Any],
    intent_plan: dict[str, Any],
    working_memory_summary: dict[str, Any],
    tool_history: list[dict[str, Any]],
    last_observation: dict[str, Any] | None,
    current_sql: str,
    last_successful_sql: str,
    budget_state: dict[str, Any],
) -> str:
    history_preview = tool_history[-8:]
    return f"""Choose the next Text2SQL tool call.

Question:
{sample["question"]}

Evidence:
{sample.get("evidence", "")}

db_id:
{sample["db_id"]}

Intent plan / current working memory:
{json.dumps({"intent_plan": intent_plan, "working_memory": working_memory_summary}, ensure_ascii=False, indent=2)}

Current SQL:
{current_sql or ""}

Last successful SQL:
{last_successful_sql or ""}

Tool history:
{json.dumps(history_preview, ensure_ascii=False, indent=2)}

Last observation:
{json.dumps(last_observation or {}, ensure_ascii=False, indent=2)}

Budget state:
{json.dumps(budget_state, ensure_ascii=False, indent=2)}

Policy constraints:
- Select exactly one tool.
- Use retrieve_schema before relying on table or column names if schema context is missing.
- Use execute_sql only for read-only answer candidates or targeted probes.
- When executing SQL that you may want to submit as the final answer, include "final_candidate": true in args.
- Do not finish unless final_sql is non-empty and was successfully executed as a final_candidate.
- If the last execute_sql failed, repair by executing a corrected SQL before finish.
- Avoid repeated exploration when budget is low.

{format_tools_for_prompt()}"""


def build_json_repair_prompt(raw_response: str, error: str) -> str:
    return f"""Repair the previous response into one strict JSON object for a Text2SQL tool call.

Validation or parse error:
{error}

Previous response:
{raw_response}

Return only:
{{
  "thought": "...",
  "tool": "...",
  "args": {{...}}
}}"""


def select_autonomous_tool_call(
    sample: dict[str, Any],
    intent_plan: dict[str, Any],
    working_memory_summary: dict[str, Any],
    tool_history: list[dict[str, Any]],
    last_observation: dict[str, Any] | None,
    current_sql: str,
    last_successful_sql: str,
    budget_state: dict[str, Any],
) -> dict[str, Any]:
    """Ask the LLM for one tool call and validate it.

    JSON parse failures get one repair attempt. Validation failures are returned
    to the caller without raising so the trace can record them.
    """
    prompt = build_autonomous_tool_prompt(
        sample=sample,
        intent_plan=intent_plan,
        working_memory_summary=working_memory_summary,
        tool_history=tool_history,
        last_observation=last_observation,
        current_sql=current_sql,
        last_successful_sql=last_successful_sql,
        budget_state=budget_state,
    )
    raw_response = call_tool_policy_llm(prompt)
    json_parse_error = None
    repair_raw_response = None
    repair_attempted = False

    try:
        parsed = parse_tool_call_json(raw_response)
    except Exception as exc:
        json_parse_error = str(exc)
        repair_attempted = True
        repair_raw_response = call_tool_policy_llm(build_json_repair_prompt(raw_response, json_parse_error))
        try:
            parsed = parse_tool_call_json(repair_raw_response)
        except Exception as repair_exc:
            return {
                "raw_response": raw_response,
                "repair_raw_response": repair_raw_response,
                "repair_attempted": repair_attempted,
                "parsed": None,
                "thought": "",
                "tool": None,
                "args": {},
                "json_parse_error": str(repair_exc),
                "validation_result": {
                    "valid": False,
                    "error": f"JSON parse failed after repair: {repair_exc}",
                },
            }

    tool_name = parsed.get("tool")
    args = parsed.get("args", {})
    validation_result = validate_tool_call(tool_name, args)
    return {
        "raw_response": raw_response,
        "repair_raw_response": repair_raw_response,
        "repair_attempted": repair_attempted,
        "parsed": parsed,
        "thought": parsed.get("thought", ""),
        "tool": tool_name,
        "args": args,
        "json_parse_error": json_parse_error,
        "validation_result": validation_result,
    }
