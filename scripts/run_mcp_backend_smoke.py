"""Small MCP backend smoke run for the iterative agent runtime adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.agents.iterative_agent import run_iterative_agent
from text2sql.agents.tool_executor import MCPToolExecutor
from text2sql.config import RESULTS_DIR
from text2sql.data import resolve_eval_samples
from text2sql.pipelines.schema_linked import build_schema_linker
from text2sql.schema.linker import DEFAULT_TOP_K


DEFAULT_OUTPUT_DIR = RESULTS_DIR / "iterative_agent" / "mcp_backend_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny MCP backend smoke test, not a benchmark.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--memory-mode", choices=["working", "episodic"], default="working")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--top-k-schema", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model-path", type=Path)
    return parser.parse_args()


def trace_has_mcp_tool_calls(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("tool_backend") == "mcp" and event.get("action") in {
            "retrieve_schema",
            "inspect_table",
            "sample_rows",
            "search_column_values",
            "execute_sql",
        }:
            return True
    return False


def memory_has_mcp_observations(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for event in record.get("trace", []):
            memory_summary = event.get("working_memory_summary") or {}
            if event.get("tool_backend") == "mcp" and memory_summary.get("tool_observation_count", 0) > 0:
                return True
    return False


def main() -> int:
    args = parse_args()
    samples = resolve_eval_samples(limit=args.limit, db_id="california_schools")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    traces_path = args.output_dir / "traces.jsonl"

    summary = run_iterative_agent(
        samples=samples,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        traces_path=traces_path,
        max_steps=args.max_steps,
        top_k_schema=args.top_k_schema,
        embedding_model_path=args.embedding_model_path,
        tool_use_mode="rule_based",
        memory_mode=args.memory_mode,
        tool_backend="mcp",
    )

    schema_linker = build_schema_linker(
        db_path=samples[0]["db_path"],
        top_k=args.top_k_schema,
        schema_linker_mode="bm25",
        embedding_model_path=args.embedding_model_path,
    )
    safety_observation = MCPToolExecutor(samples[0], schema_linker=schema_linker).execute_sql(
        "DROP TABLE schools",
        db_id="california_schools",
    )
    smoke_report = {
        "purpose": "mcp_backend_smoke_validation_not_benchmark",
        "sample_count": len(samples),
        "summary": summary,
        "trace_has_mcp_tool_calls": trace_has_mcp_tool_calls(traces_path),
        "memory_has_mcp_tool_observations": memory_has_mcp_observations(predictions_path),
        "safety_guard_observation": safety_observation,
        "safety_guard_passed": bool(safety_observation.get("safety_blocked")),
    }
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(smoke_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"smoke_report_path": str(report_path), **smoke_report}, ensure_ascii=False, indent=2))
    return 0 if all(
        [
            smoke_report["trace_has_mcp_tool_calls"],
            smoke_report["memory_has_mcp_tool_observations"],
            smoke_report["safety_guard_passed"],
        ]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
