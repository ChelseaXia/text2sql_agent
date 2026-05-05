"""Tool-calling Text2SQL agent controller."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from text2sql.agents.tools import AgentTools, build_sample_context
from text2sql.agents.trace import write_trace_examples
from text2sql.config import PROJECT_ROOT, RESULTS_DIR
from text2sql.data import load_bird_dev
from text2sql.db import run_sql, same_result
from text2sql.eval import compute_metrics
from text2sql.prompts.planning import SYSTEM_PROMPT
from text2sql.schema.linker import SchemaLinker

DEFAULT_MODEL = "deepseek-v4-flash"
FALLBACK_MODEL = "deepseek-chat"
DEFAULT_DB_ID = "california_schools"
DEFAULT_LIMIT = 50
DEFAULT_TOP_K_SCHEMA = 30
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_WORKERS = 2
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 120

DEFAULT_TRACES_PATH = RESULTS_DIR / "day7_agent_traces.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day7_agent_metrics.json"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "agent_trace_examples.md"


def require_api_key(api_key=None):
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in the environment.")
    return key


def make_client(api_key=None):
    return OpenAI(api_key=require_api_key(api_key), base_url="https://api.deepseek.com")


class Text2SQLAgent:
    def __init__(self, context, model=DEFAULT_MODEL, top_k_schema=DEFAULT_TOP_K_SCHEMA, max_steps=DEFAULT_MAX_STEPS, timeout_seconds=DEFAULT_SAMPLE_TIMEOUT_SECONDS, api_key=None):
        self.context = context
        self.model = model
        self.top_k_schema = top_k_schema
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.client = make_client(api_key)
        self.tools = AgentTools(context, top_k_schema=top_k_schema)
        self.final_sql = ""
        self.final_sql_source = "unfinished"
        self.has_successful_execute = False
        self.execute_call_count = 0
        self.last_successful_sql = ""
        self.last_execute_error = None

    @staticmethod
    def tool_specs():
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_schema",
                    "description": "Retrieve linked schema text and selected tables for the current database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "evidence": {"type": "string"},
                            "db_id": {"type": "string"},
                            "db_path": {"type": "string"},
                        },
                        "required": ["question", "evidence", "db_id", "db_path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_table",
                    "description": "Inspect one table's columns, types, descriptions, and sample values.",
                    "parameters": {
                        "type": "object",
                        "properties": {"table_name": {"type": "string"}},
                        "required": ["table_name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sample_rows",
                    "description": "Sample a few rows from one table.",
                    "parameters": {
                        "type": "object",
                        "properties": {"table_name": {"type": "string"}, "n": {"type": "integer", "minimum": 1, "maximum": 10}},
                        "required": ["table_name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_column_values",
                    "description": "Search a text-like column for distinct values matching a keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string"},
                            "column_name": {"type": "string"},
                            "keyword": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                        },
                        "required": ["table_name", "column_name", "keyword"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a candidate SQLite SQL query against the current database.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Finish the task and return the final SQL after at least one successful execute_sql.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _tool_choice(self, step_index):
        if not self.has_successful_execute and step_index >= self.max_steps - 1:
            return {"type": "function", "function": {"name": "execute_sql"}}
        if self.has_successful_execute and step_index >= self.max_steps:
            return {"type": "function", "function": {"name": "finish"}}
        return "auto"

    def _call_model(self, messages, tool_choice):
        last_error = None
        model_candidates = [self.model]
        if self.model != FALLBACK_MODEL:
            model_candidates.append(FALLBACK_MODEL)
        for model_name in model_candidates:
            try:
                return self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=self.tool_specs(),
                    tool_choice=tool_choice,
                    temperature=0.0,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Agent model call failed: {last_error}")

    @staticmethod
    def _parse_tool_arguments(raw_arguments):
        parsed = json.loads(raw_arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed

    @staticmethod
    def _validate_required(args, required, allowed):
        missing = [key for key in required if key not in args]
        extra = sorted(set(args) - set(allowed))
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        if extra:
            raise ValueError(f"Unexpected fields: {extra}")

    def _validate_tool_arguments(self, tool_name, args):
        if tool_name == "retrieve_schema":
            self._validate_required(args, ["question", "evidence", "db_id", "db_path"], ["question", "evidence", "db_id", "db_path"])
        elif tool_name == "inspect_table":
            self._validate_required(args, ["table_name"], ["table_name"])
        elif tool_name == "sample_rows":
            self._validate_required(args, ["table_name"], ["table_name", "n"])
        elif tool_name == "search_column_values":
            self._validate_required(args, ["table_name", "column_name", "keyword"], ["table_name", "column_name", "keyword", "limit"])
        elif tool_name in {"execute_sql", "finish"}:
            self._validate_required(args, ["sql"], ["sql"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        return args

    def _run_tool(self, tool_name, args):
        if tool_name == "retrieve_schema":
            return self.tools.retrieve_schema(**args)
        if tool_name == "inspect_table":
            return self.tools.inspect_table(**args)
        if tool_name == "sample_rows":
            return self.tools.sample_rows(**args)
        if tool_name == "search_column_values":
            return self.tools.search_column_values(**args)
        if tool_name == "execute_sql":
            self.execute_call_count += 1
            result = self.tools.execute_sql(**args)
            if result["success"]:
                self.has_successful_execute = True
                self.last_successful_sql = args["sql"]
                self.last_execute_error = None
            else:
                self.last_execute_error = result["error"]
            return result
        if tool_name == "finish":
            if not self.has_successful_execute:
                raise ValueError("finish cannot be used before a successful execute_sql.")
            return self.tools.finish(**args)
        raise ValueError(f"Unknown tool: {tool_name}")

    def run(self):
        sample = self.context.sample
        started_at = time.monotonic()
        traces = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"sample_id: {sample['sample_id']}\n"
                    f"db_id: {sample['db_id']}\n"
                    f"db_path: {sample['db_path']}\n"
                    f"question: {sample['question']}\n"
                    f"evidence: {sample.get('evidence') or 'None'}\n"
                    "Solve this by using tools and finish with SQL."
                ),
            },
        ]

        forced_schema_observation = self.tools.retrieve_schema(
            question=sample["question"],
            evidence=sample.get("evidence", ""),
            db_id=sample["db_id"],
            db_path=sample["db_path"],
        )
        traces.append(
            {
                "step": 1,
                "action": "retrieve_schema",
                "input": {
                    "question": sample["question"],
                    "evidence": sample.get("evidence", ""),
                    "db_id": sample["db_id"],
                    "db_path": sample["db_path"],
                },
                "observation": forced_schema_observation,
                "forced": True,
            }
        )
        messages.append({"role": "user", "content": "Initial schema retrieval result from the controller:\n" + json.dumps(forced_schema_observation, ensure_ascii=False)})

        for step_index in range(2, self.max_steps + 1):
            if time.monotonic() - started_at > self.timeout_seconds:
                traces.append({"step": step_index, "action": "timeout", "input": {}, "observation": {"error": f"Sample timed out after {self.timeout_seconds}s"}})
                break

            response = self._call_model(messages, tool_choice=self._tool_choice(step_index))
            message = response.choices[0].message
            assistant_entry = {"role": "assistant", "content": message.content or ""}
            if getattr(message, "tool_calls", None):
                assistant_entry["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                    }
                    for tool_call in message.tool_calls
                ]
            messages.append(assistant_entry)

            if not getattr(message, "tool_calls", None):
                observation = {"error": "Assistant did not call a tool. Use tools and finish with finish(sql)."}
                traces.append({"step": step_index, "action": "assistant_no_tool_call", "input": {"content": message.content or ""}, "observation": observation})
                messages.append({"role": "user", "content": observation["error"]})
                continue

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments
                validated_args = None
                try:
                    validated_args = self._validate_tool_arguments(tool_name, self._parse_tool_arguments(raw_arguments))
                    observation = self._run_tool(tool_name, validated_args)
                except Exception as exc:
                    observation = {"error": str(exc)}

                traces.append({"step": step_index, "action": tool_name, "input": validated_args if validated_args is not None else raw_arguments, "observation": observation})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": json.dumps(observation, ensure_ascii=False)})

                if tool_name == "finish" and "final_sql" in observation:
                    self.final_sql = observation["final_sql"]
                    self.final_sql_source = "finish_tool"
                    break
            if self.final_sql:
                break

        if not self.final_sql and self.last_successful_sql:
            self.final_sql = self.last_successful_sql
            self.final_sql_source = "last_successful_execute"
            traces.append({"step": len(traces) + 1, "action": "controller_fallback_finish", "input": {"sql": self.last_successful_sql}, "observation": {"final_sql": self.last_successful_sql}})

        final_result = run_sql(self.final_sql, sample["db_path"]) if self.final_sql else {"success": False, "rows": [], "error": "Agent did not produce final SQL"}
        gold_result = run_sql(sample["gold_sql"], sample["db_path"])
        ex = bool(final_result["success"] and gold_result["success"] and same_result(final_result["rows"], gold_result["rows"]))
        return {
            "sample_id": sample["sample_id"],
            "db_id": sample["db_id"],
            "difficulty": sample["difficulty"],
            "question": sample["question"],
            "evidence": sample.get("evidence", ""),
            "gold_sql": sample["gold_sql"],
            "final_sql": self.final_sql,
            "final_sql_source": self.final_sql_source,
            "pred_success": final_result["success"],
            "gold_success": gold_result["success"],
            "pred_error": final_result["error"],
            "gold_error": gold_result["error"],
            "pred_row_count": len(final_result["rows"]),
            "gold_row_count": len(gold_result["rows"]),
            "pred_rows_preview": final_result["rows"][:5],
            "gold_rows_preview": gold_result["rows"][:5],
            "ex": ex,
            "execute_call_count": self.execute_call_count,
            "has_successful_execute": self.has_successful_execute,
            "trace": traces,
        }


def run_agent(samples, traces_path, metrics_path, doc_path, model, top_k_schema, max_steps, max_workers, timeout_seconds, api_key=None):
    require_api_key(api_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = list(samples)
    linker_cache = {sample["db_path"]: SchemaLinker(sample["db_path"]) for sample in samples}

    def _run_one(sample):
        context = build_sample_context(sample, linker_cache[sample["db_path"]])
        agent = Text2SQLAgent(context=context, model=model, top_k_schema=top_k_schema, max_steps=max_steps, timeout_seconds=timeout_seconds, api_key=api_key)
        return agent.run()

    results_by_id = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(_run_one, sample): sample for sample in samples}
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "sample_id": sample["sample_id"],
                    "db_id": sample["db_id"],
                    "difficulty": sample["difficulty"],
                    "question": sample["question"],
                    "evidence": sample.get("evidence", ""),
                    "gold_sql": sample["gold_sql"],
                    "final_sql": "",
                    "final_sql_source": "controller_error",
                    "pred_success": False,
                    "gold_success": True,
                    "pred_error": str(exc),
                    "gold_error": None,
                    "pred_row_count": 0,
                    "gold_row_count": 0,
                    "pred_rows_preview": [],
                    "gold_rows_preview": [],
                    "ex": False,
                    "execute_call_count": 0,
                    "has_successful_execute": False,
                    "trace": [{"step": 1, "action": "controller_error", "input": {}, "observation": {"error": str(exc)}}],
                }
            results_by_id[sample["sample_id"]] = result
            print(f"sample_id={sample['sample_id']} pred_success={result['pred_success']} ex={result['ex']} execute_calls={result.get('execute_call_count', 0)}")

    ordered_results = [results_by_id[sample["sample_id"]] for sample in samples]
    with traces_path.open("w", encoding="utf-8") as traces_file:
        for result in ordered_results:
            traces_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    metrics = compute_metrics(ordered_results)
    metrics["avg_execute_call_count"] = (
        sum(result.get("execute_call_count", 0) for result in ordered_results) / len(ordered_results)
        if ordered_results else 0.0
    )
    metrics["finish_count"] = sum(1 for result in ordered_results if result.get("final_sql"))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trace_examples(traces_path, doc_path)
    return {
        "metrics": metrics,
        "traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
        "doc_path": str(doc_path),
    }


run_agent_batch = run_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Run Text2SQL agent controller.")
    parser.add_argument("--db-id", default=DEFAULT_DB_ID)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k-schema", type=int, default=DEFAULT_TOP_K_SCHEMA)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_SAMPLE_TIMEOUT_SECONDS)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_bird_dev(limit=args.limit, db_id=args.db_id)
    summary = run_agent(
        samples=samples,
        traces_path=args.traces_output,
        metrics_path=args.metrics_output,
        doc_path=args.doc_output,
        model=args.model,
        top_k_schema=args.top_k_schema,
        max_steps=args.max_steps,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
