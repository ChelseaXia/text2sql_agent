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
DEFAULT_MAX_CALL_TIMEOUT_SECONDS = 30

DEFAULT_TRACES_PATH = RESULTS_DIR / "react_agent_traces.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "react_agent_metrics.json"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "agent_trace_examples.md"


def require_api_key(api_key=None):
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in the environment.")
    return key


def make_client(api_key=None):
    return OpenAI(api_key=require_api_key(api_key), base_url="https://api.deepseek.com")


class Text2SQLAgent:
    def __init__(self, context, model=DEFAULT_MODEL, top_k_schema=DEFAULT_TOP_K_SCHEMA, max_steps=DEFAULT_MAX_STEPS, timeout_seconds=DEFAULT_SAMPLE_TIMEOUT_SECONDS, api_key=None, allow_model_fallback=False):
        self.context = context
        self.model = model
        self.top_k_schema = top_k_schema
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.allow_model_fallback = allow_model_fallback
        self.client = make_client(api_key)
        self.tools = AgentTools(context, top_k_schema=top_k_schema)
        self.final_sql = ""
        self.final_sql_source = "unfinished"
        self.has_successful_execute = False
        self.execute_call_count = 0
        self.last_successful_sql = ""
        self.last_execute_error = None
        self.actual_model = None
        self.successful_executed_sqls = []
        self.strict_final_sql = ""
        self.strict_final_sql_source = "unfinished"
        self.relaxed_final_sql = ""
        self.relaxed_final_sql_source = "unfinished"

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

    def _call_model(self, messages, tool_choice, remaining_seconds):
        call_timeout = min(DEFAULT_MAX_CALL_TIMEOUT_SECONDS, max(1, int(remaining_seconds)))
        last_error = None
        model_candidates = [self.model]
        if self.allow_model_fallback and self.model != FALLBACK_MODEL:
            model_candidates.append(FALLBACK_MODEL)
        for model_name in model_candidates:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=self.tool_specs(),
                    tool_choice=tool_choice,
                    temperature=0.0,
                    extra_body={"thinking": {"type": "disabled"}},
                    timeout=call_timeout,
                )
                self.actual_model = model_name
                return response
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
    def _normalize_sql(sql):
        if not sql:
            return ""
        sql = sql.strip()
        sql = sql.rstrip(";")
        # 压缩空白
        import re
        return re.sub(r"\s+", " ", sql)

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
                normalized_sql = self._normalize_sql(args["sql"])
                self.successful_executed_sqls.append({
                    "normalized_sql": normalized_sql,
                    "row_count": result.get("row_count", len(result["rows"])),
                    "rows_preview": result["rows"][:5],
                    "error": None
                })
            else:
                self.last_execute_error = result["error"]
            return result
        if tool_name == "finish":
            if not self.has_successful_execute:
                raise ValueError("finish cannot be used before a successful execute_sql.")
            normalized_finish_sql = self._normalize_sql(args["sql"])
            matched = any(
                executed["normalized_sql"] == normalized_finish_sql
                for executed in self.successful_executed_sqls
            )
            if not matched:
                return {
                    "error": "finish SQL must be successfully executed by execute_sql before finish.",
                    "finish_rejected": True
                }
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
                "model": self.model,
                "actual_model": self.actual_model,
            }
        )
        messages.append({"role": "user", "content": "Initial schema retrieval result from the controller:\n" + json.dumps(forced_schema_observation, ensure_ascii=False)})

        for step_index in range(2, self.max_steps + 1):
            remaining_seconds = self.timeout_seconds - (time.monotonic() - started_at)
            if remaining_seconds <= 0:
                traces.append(
                    {
                        "step": step_index,
                        "action": "timeout",
                        "input": {},
                        "observation": {"error": f"Sample timed out after {self.timeout_seconds}s"},
                        "model": self.model,
                        "actual_model": self.actual_model,
                    }
                )
                break

            response = self._call_model(messages, tool_choice=self._tool_choice(step_index), remaining_seconds=remaining_seconds)
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
                traces.append(
                    {
                        "step": step_index,
                        "action": "assistant_no_tool_call",
                        "input": {"content": message.content or ""},
                        "observation": observation,
                        "model": self.model,
                        "actual_model": self.actual_model,
                    }
                )
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

                trace_entry = {
                    "step": step_index,
                    "action": tool_name,
                    "input": validated_args if validated_args is not None else raw_arguments,
                    "observation": observation,
                    "model": self.model,
                    "actual_model": self.actual_model,
                }
                if observation.get("finish_rejected"):
                    trace_entry["finish_rejected"] = True
                traces.append(trace_entry)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": json.dumps(observation, ensure_ascii=False)})

                if tool_name == "finish" and "final_sql" in observation:
                    self.final_sql = observation["final_sql"]
                    self.final_sql_source = "finish_tool"
                    break
            if self.final_sql:
                break

        if self.final_sql:
            self.strict_final_sql = self.final_sql
            self.strict_final_sql_source = self.final_sql_source
        else:
            self.strict_final_sql = ""
            self.strict_final_sql_source = "no_finish"
            traces.append(
                {
                    "step": len(traces) + 1,
                    "action": "no_finish",
                    "input": {},
                    "observation": {"error": "Agent did not call finish with a successfully executed final SQL."},
                    "model": self.model,
                    "actual_model": self.actual_model,
                }
            )

        if self.strict_final_sql:
            self.relaxed_final_sql = self.strict_final_sql
            self.relaxed_final_sql_source = self.strict_final_sql_source
        elif self.last_successful_sql:
            self.relaxed_final_sql = self.last_successful_sql
            self.relaxed_final_sql_source = "last_successful_execute"
        else:
            self.relaxed_final_sql = ""
            self.relaxed_final_sql_source = "no_finish"

        strict_result = (
            run_sql(self.strict_final_sql, sample["db_path"])
            if self.strict_final_sql
            else {"success": False, "rows": [], "error": "Agent did not produce final SQL"}
        )
        relaxed_result = (
            run_sql(self.relaxed_final_sql, sample["db_path"])
            if self.relaxed_final_sql
            else {"success": False, "rows": [], "error": "Agent did not produce final SQL"}
        )
        gold_result = run_sql(sample["gold_sql"], sample["db_path"])
        strict_ex = bool(
            strict_result["success"] and gold_result["success"] and same_result(strict_result["rows"], gold_result["rows"])
        )
        relaxed_ex = bool(
            relaxed_result["success"] and gold_result["success"] and same_result(relaxed_result["rows"], gold_result["rows"])
        )
        return {
            "sample_id": sample["sample_id"],
            "db_id": sample["db_id"],
            "difficulty": sample["difficulty"],
            "question": sample["question"],
            "evidence": sample.get("evidence", ""),
            "gold_sql": sample["gold_sql"],
            "strict_final_sql": self.strict_final_sql,
            "strict_final_sql_source": self.strict_final_sql_source,
            "strict_pred_success": strict_result["success"],
            "strict_ex": strict_ex,
            "relaxed_final_sql": self.relaxed_final_sql,
            "relaxed_final_sql_source": self.relaxed_final_sql_source,
            "relaxed_pred_success": relaxed_result["success"],
            "relaxed_ex": relaxed_ex,
            "final_sql": self.strict_final_sql,
            "final_sql_source": self.strict_final_sql_source,
            "pred_success": strict_result["success"],
            "gold_success": gold_result["success"],
            "pred_error": strict_result["error"],
            "gold_error": gold_result["error"],
            "pred_row_count": len(strict_result["rows"]),
            "gold_row_count": len(gold_result["rows"]),
            "pred_rows_preview": strict_result["rows"][:5],
            "gold_rows_preview": gold_result["rows"][:5],
            "ex": strict_ex,
            "execute_call_count": self.execute_call_count,
            "has_successful_execute": self.has_successful_execute,
            "last_successful_sql": self.last_successful_sql,
            "successful_executed_sqls": self.successful_executed_sqls,
            "model": self.model,
            "actual_model": self.actual_model,
            "trace": traces,
        }


def run_agent(samples, traces_path, metrics_path, doc_path, model, top_k_schema, max_steps, max_workers, timeout_seconds, api_key=None, allow_model_fallback=False):
    require_api_key(api_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = list(samples)
    linker_cache = {sample["db_path"]: SchemaLinker(sample["db_path"]) for sample in samples}

    def _run_one(sample):
        context = build_sample_context(sample, linker_cache[sample["db_path"]])
        agent = Text2SQLAgent(
            context=context,
            model=model,
            top_k_schema=top_k_schema,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
            allow_model_fallback=allow_model_fallback,
        )
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
                    "model": model,
                    "actual_model": None,
                    "trace": [{"step": 1, "action": "controller_error", "input": {}, "observation": {"error": str(exc)}, "model": model, "actual_model": None}],
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
    
    strict_finish_count = 0
    no_finish_count = 0
    finish_rejected_count = 0
    fallback_finish_count = 0
    executed_but_not_finished_count = 0

    for result in ordered_results:
        if result.get("strict_final_sql_source", result.get("final_sql_source")) == "finish_tool":
            strict_finish_count += 1
        elif result.get("strict_final_sql_source", result.get("final_sql_source")) == "no_finish":
            no_finish_count += 1
            if result.get("has_successful_execute"):
                executed_but_not_finished_count += 1
        if result.get("relaxed_final_sql_source") == "last_successful_execute":
            fallback_finish_count += 1

        trace = result.get("trace", [])
        for step in trace:
            if step.get("finish_rejected"):
                finish_rejected_count += 1

    total = len(ordered_results)
    strict_vsr_count = sum(1 for result in ordered_results if result.get("strict_pred_success", result.get("pred_success")))
    strict_ex_count = sum(1 for result in ordered_results if result.get("strict_ex", result.get("ex")))
    relaxed_vsr_count = sum(1 for result in ordered_results if result.get("relaxed_pred_success", result.get("pred_success")))
    relaxed_ex_count = sum(1 for result in ordered_results if result.get("relaxed_ex", result.get("ex")))

    metrics["strict_finish_count"] = strict_finish_count
    metrics["finish_count"] = strict_finish_count
    metrics["no_finish_count"] = no_finish_count
    metrics["fallback_finish_count"] = fallback_finish_count
    metrics["finish_rejected_count"] = finish_rejected_count
    metrics["executed_but_not_finished_count"] = executed_but_not_finished_count
    metrics["strict_VSR"] = strict_vsr_count / total if total else 0.0
    metrics["strict_EX"] = strict_ex_count / total if total else 0.0
    metrics["relaxed_VSR"] = relaxed_vsr_count / total if total else 0.0
    metrics["relaxed_EX"] = relaxed_ex_count / total if total else 0.0
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
    parser.add_argument("--allow-model-fallback", action="store_true")
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
        allow_model_fallback=args.allow_model_fallback,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
