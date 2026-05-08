"""Compare local and MCP ToolExecutor outputs on fixed diagnostic tool calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from text2sql.agents.tool_executor import LocalToolExecutor, MCPToolExecutor
from text2sql.config import RESULTS_DIR
from text2sql.data import resolve_eval_samples
from text2sql.pipelines.schema_linked import build_schema_linker
from text2sql.schema.linker import DEFAULT_TOP_K


DEFAULT_REPORT_PATH = RESULTS_DIR / "iterative_agent" / "backend_parity_report.json"
LARGE_FIELDS = {"linked_schema_text"}


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, tuple):
        return [canonical(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    text = json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def missing_vs_none_diffs(local_value: Any, mcp_value: Any, prefix: str = "") -> list[str]:
    diffs = []
    if isinstance(local_value, dict) and isinstance(mcp_value, dict):
        for key in sorted(set(local_value) | set(mcp_value)):
            path = f"{prefix}.{key}" if prefix else key
            local_has = key in local_value
            mcp_has = key in mcp_value
            if local_has != mcp_has:
                present_value = local_value.get(key) if local_has else mcp_value.get(key)
                if present_value is None:
                    diffs.append(f"{path}: missing vs None")
                else:
                    diffs.append(f"{path}: missing key")
                continue
            diffs.extend(missing_vs_none_diffs(local_value[key], mcp_value[key], path))
    elif isinstance(local_value, list) and isinstance(mcp_value, list):
        for index, (left, right) in enumerate(zip(local_value, mcp_value)):
            diffs.extend(missing_vs_none_diffs(left, right, f"{prefix}[{index}]"))
    return diffs


def semantic_match(tool: str, local_output: dict[str, Any], mcp_output: dict[str, Any]) -> tuple[bool, str | None]:
    if tool == "execute_sql" and local_output.get("safety_blocked") or mcp_output.get("safety_blocked"):
        if local_output.get("safety_blocked") and mcp_output.get("safety_blocked"):
            return True, None
        return False, "safety_blocked differs"
    if canonical(local_output) == canonical(mcp_output):
        return True, None
    none_diffs = missing_vs_none_diffs(local_output, mcp_output)
    if none_diffs:
        return False, "; ".join(none_diffs[:5])
    return False, "normalized outputs differ"


def call_tool(executor, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool == "retrieve_schema":
        return executor.retrieve_schema(question=args["question"], db_id=args["db_id"])
    if tool == "inspect_table":
        return executor.inspect_table(table_name=args["table_name"], db_id=args["db_id"])
    if tool == "sample_rows":
        return executor.sample_rows(table_name=args["table_name"], n=args["limit"], db_id=args["db_id"])
    if tool == "search_column_values":
        return executor.search_column_values(
            table_name=args["table_name"],
            column_name=args["column_name"],
            query=args["search_value"],
            db_id=args["db_id"],
        )
    if tool == "execute_sql":
        return executor.execute_sql(sql=args["sql"], db_id=args["db_id"])
    raise ValueError(f"Unsupported tool: {tool}")


def build_cases() -> list[dict[str, Any]]:
    return [
        {
            "tool": "retrieve_schema",
            "args": {"db_id": "california_schools", "question": "How many schools are there?"},
        },
        {
            "tool": "inspect_table",
            "args": {"db_id": "california_schools", "table_name": "schools"},
        },
        {
            "tool": "sample_rows",
            "args": {"db_id": "california_schools", "table_name": "schools", "limit": 3},
        },
        {
            "tool": "search_column_values",
            "args": {
                "db_id": "california_schools",
                "table_name": "schools",
                "column_name": "School",
                "search_value": "Charter",
            },
        },
        {
            "tool": "execute_sql",
            "args": {"db_id": "california_schools", "sql": "SELECT 1 AS one"},
        },
        {
            "tool": "execute_sql",
            "args": {"db_id": "california_schools", "sql": "DROP TABLE schools"},
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed local-vs-MCP tool backend parity diagnostics.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--top-k-schema", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = resolve_eval_samples(limit=1, db_id="california_schools")[0]
    schema_linker = build_schema_linker(
        db_path=sample["db_path"],
        top_k=args.top_k_schema,
        schema_linker_mode="bm25",
        embedding_model_path=args.embedding_model_path,
    )
    local_executor = LocalToolExecutor(sample, schema_linker, top_k_schema=args.top_k_schema)
    mcp_executor = MCPToolExecutor(sample, schema_linker=schema_linker, top_k_schema=args.top_k_schema)

    case_reports = []
    for case in build_cases():
        tool = case["tool"]
        local_output = call_tool(local_executor, tool, case["args"])
        mcp_output = call_tool(mcp_executor, tool, case["args"])
        exact_match = canonical(local_output) == canonical(mcp_output)
        sem_match, mismatch_reason = semantic_match(tool, local_output, mcp_output)
        case_report = {
            "tool": tool,
            "args": case["args"],
            "local_hash": stable_hash(local_output),
            "mcp_hash": stable_hash(mcp_output),
            "exact_match": exact_match,
            "semantic_match": sem_match,
            "mismatch_reason": mismatch_reason,
        }
        for field in LARGE_FIELDS:
            if field in local_output or field in mcp_output:
                case_report[f"{field}_exact_match"] = local_output.get(field) == mcp_output.get(field)
                case_report[f"{field}_local_hash"] = stable_hash(local_output.get(field))
                case_report[f"{field}_mcp_hash"] = stable_hash(mcp_output.get(field))
        if tool == "execute_sql" and case["args"]["sql"].upper().startswith("DROP"):
            case_report["safety_guard_passed"] = bool(
                local_output.get("safety_blocked") and mcp_output.get("safety_blocked")
            )
            if not case_report["safety_guard_passed"]:
                case_report["semantic_match"] = False
                case_report["mismatch_reason"] = "safety guard did not block both backends"
        case_reports.append(case_report)

    report = {
        "backend_pair": "local_vs_mcp",
        "sample_count": 1,
        "tool_call_count": len(case_reports),
        "all_passed": all(case["semantic_match"] for case in case_reports),
        "cases": case_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(args.output), "all_passed": report["all_passed"]}, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
