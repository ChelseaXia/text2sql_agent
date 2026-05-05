"""Controlled execution-repair agent built from existing Day3 and Day5.5 logic."""

import argparse
import json
from pathlib import Path

from text2sql.config import PROJECT_ROOT, RESULTS_DIR
from text2sql.data import load_bird_dev_by_sample_ids
from text2sql.db import run_sql, same_result
from text2sql.eval import compute_metrics, load_jsonl
from text2sql.llm import call_llm
from text2sql.pipelines.naive import extract_sql
from text2sql.prompts.generation import build_linked_prompt
from text2sql.prompts.repair import build_repair_prompt
from text2sql.schema.linker import SchemaLinker

DEFAULT_DAY5_INPUT_PATH = RESULTS_DIR / "day5_repair_from_day3_predictions.jsonl"
DEFAULT_TRACES_PATH = RESULTS_DIR / "controlled_agent_traces.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "controlled_agent_metrics.json"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "controlled_agent_trace_examples.md"
DEFAULT_DB_ID = "california_schools"
DEFAULT_LIMIT = 50
DEFAULT_TOP_K_SCHEMA = 30
DEFAULT_MAX_REPAIR_ROUNDS = 1


def retrieved_column_names(items):
    return [f"{item['table']}.{item['column']}" for item in items]


def load_day5_sample_ids(day5_predictions_path, db_id=None, limit=None):
    rows = load_jsonl(day5_predictions_path)
    if db_id is not None:
        rows = [row for row in rows if row["db_id"] == db_id]
    sample_ids = [row["sample_id"] for row in rows]
    if limit is not None:
        sample_ids = sample_ids[:limit]
    return sample_ids


def load_controlled_samples(day5_predictions_path, db_id=None, limit=None):
    sample_ids = load_day5_sample_ids(day5_predictions_path, db_id=db_id, limit=limit)
    return load_bird_dev_by_sample_ids(sample_ids, db_id=db_id)


def generate_sql(sample, linked_schema_text):
    raw_response = ""
    try:
        raw_response = call_llm(build_linked_prompt(sample, linked_schema_text))
        return extract_sql(raw_response), raw_response, None
    except Exception as exc:
        return "", raw_response, str(exc)


def repair_sql(sample, linked_schema_text, previous_sql, previous_error):
    raw_response = ""
    try:
        raw_response = call_llm(
            build_repair_prompt(
                question=sample["question"],
                evidence=sample.get("evidence", ""),
                linked_schema_text=linked_schema_text,
                previous_sql=previous_sql,
                sqlite_error=previous_error,
            )
        )
        return extract_sql(raw_response), raw_response, None
    except Exception as exc:
        return "", raw_response, str(exc)


def execute_sql(sql, db_path, llm_error=None):
    if not sql:
        return {"success": False, "rows": [], "error": llm_error or "Empty SQL prediction"}
    return run_sql(sql, db_path)


def make_trace(step, action, reasoning, tool_input, tool_output, observation):
    return {
        "step": step,
        "action": action,
        "reasoning": reasoning,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "observation": observation,
    }


def write_trace_examples(records, output_path):
    lines = [
        "# Controlled Agent Trace Examples",
        "",
        "Stepwise traces for the controlled execution-repair agent.",
        "",
    ]

    if not records:
        lines.append("No traces available yet.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    examples = []
    for record in records:
        if record.get("ex"):
            examples.append(record)
            break
    for record in records:
        if record.get("repair_attempted"):
            examples.append(record)
            break
    if not examples:
        examples = records[:1]

    seen = set()
    for record in examples:
        if record["sample_id"] in seen:
            continue
        seen.add(record["sample_id"])
        lines.extend(
            [
                f"## Sample {record['sample_id']}",
                "",
                f"- Difficulty: `{record['difficulty']}`",
                f"- EX: `{record['ex']}`",
                f"- Repair attempted: `{record['repair_attempted']}`",
                f"- Repair success: `{record['repair_success']}`",
                "",
                "Final SQL:",
                "```sql",
                record.get("final_sql") or "-- no final sql --",
                "```",
                "",
                "Trace summary:",
            ]
        )
        for step in record.get("trace", [])[:6]:
            lines.append(f"- Step {step['step']} `{step['action']}`: {step['observation']}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_controlled_agent(samples, traces_path, metrics_path, doc_path, top_k_schema, max_repair_rounds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = list(samples)
    linker_cache = {}
    records = []

    with traces_path.open("w", encoding="utf-8") as traces_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)

            trace = []
            linked_items, linked_schema_text = linker_cache[db_path].retrieve(
                sample["question"],
                sample.get("evidence", ""),
                top_k=top_k_schema,
            )
            selected_tables = []
            for item in linked_items:
                if item["table"] not in selected_tables:
                    selected_tables.append(item["table"])
            trace.append(
                make_trace(
                    1,
                    "retrieve_schema",
                    "Use schema linker to select task-relevant tables and columns before SQL generation.",
                    {
                        "question": sample["question"],
                        "evidence": sample.get("evidence", ""),
                        "top_k_schema": top_k_schema,
                    },
                    {
                        "selected_tables": selected_tables,
                        "retrieved_columns": retrieved_column_names(linked_items),
                        "linked_schema_text": linked_schema_text,
                    },
                    f"Selected tables: {', '.join(selected_tables)}.",
                )
            )

            initial_sql, initial_raw_response, initial_llm_error = generate_sql(sample, linked_schema_text)
            trace.append(
                make_trace(
                    2,
                    "generate_sql",
                    "Generate an initial SQL query using the schema-linked promptfix prompt.",
                    {"question": sample["question"], "evidence": sample.get("evidence", "")},
                    {"initial_sql": initial_sql, "raw_response": initial_raw_response, "llm_error": initial_llm_error},
                    "Generated initial SQL candidate." if initial_sql else f"Initial generation failed: {initial_llm_error}",
                )
            )

            initial_result = execute_sql(initial_sql, db_path, initial_llm_error)
            trace.append(
                make_trace(
                    3,
                    "execute_sql",
                    "Execute the initial SQL to validate syntax and semantics before any repair.",
                    {"sql": initial_sql},
                    {
                        "success": initial_result["success"],
                        "row_count": len(initial_result["rows"]),
                        "rows_preview": initial_result["rows"][:5],
                        "error": initial_result["error"],
                    },
                    "Initial SQL executed successfully." if initial_result["success"] else f"Initial SQL failed: {initial_result['error']}",
                )
            )

            repair_attempted = False
            repair_success = False
            repaired_sql = ""
            final_sql = initial_sql
            final_result = initial_result
            final_error = initial_result["error"]

            if not initial_result["success"]:
                repair_attempted = True
                previous_sql = initial_sql
                previous_error = initial_result["error"] or "Unknown SQLite error"

                for repair_round in range(1, max_repair_rounds + 1):
                    repaired_sql, repair_raw_response, repair_llm_error = repair_sql(
                        sample,
                        linked_schema_text,
                        previous_sql,
                        previous_error,
                    )
                    trace.append(
                        make_trace(
                            4,
                            "repair_sql",
                            "Repair the failed SQL using the strict execution-repair prompt.",
                            {
                                "previous_sql": previous_sql,
                                "previous_error": previous_error,
                                "repair_round": repair_round,
                            },
                            {
                                "repaired_sql": repaired_sql,
                                "raw_response": repair_raw_response,
                                "llm_error": repair_llm_error,
                            },
                            "Generated repaired SQL candidate." if repaired_sql else f"Repair generation failed: {repair_llm_error}",
                        )
                    )

                    repaired_result = execute_sql(repaired_sql, db_path, repair_llm_error)
                    trace.append(
                        make_trace(
                            5,
                            "execute_sql",
                            "Execute the repaired SQL and keep it only if the execution succeeds.",
                            {"sql": repaired_sql},
                            {
                                "success": repaired_result["success"],
                                "row_count": len(repaired_result["rows"]),
                                "rows_preview": repaired_result["rows"][:5],
                                "error": repaired_result["error"],
                            },
                            "Repaired SQL executed successfully." if repaired_result["success"] else f"Repaired SQL failed: {repaired_result['error']}",
                        )
                    )

                    final_error = repaired_result["error"]
                    if repaired_result["success"]:
                        repair_success = True
                        final_sql = repaired_sql
                        final_result = repaired_result
                        break

                if not repair_success:
                    final_sql = initial_sql
                    final_result = initial_result

            trace.append(
                make_trace(
                    6,
                    "finish",
                    "Finish with the repaired SQL if repair succeeded; otherwise keep the initial SQL as the final candidate.",
                    {"initial_sql": initial_sql, "repaired_sql": repaired_sql},
                    {"final_sql": final_sql, "repair_attempted": repair_attempted, "repair_success": repair_success},
                    "Final SQL selected for evaluation.",
                )
            )

            gold_result = run_sql(sample["gold_sql"], db_path)
            ex = bool(final_result["success"] and gold_result["success"] and same_result(final_result["rows"], gold_result["rows"]))
            record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "evidence": sample.get("evidence", ""),
                "gold_sql": sample["gold_sql"],
                "initial_sql": initial_sql,
                "final_sql": final_sql,
                "pred_success": final_result["success"],
                "gold_success": gold_result["success"],
                "ex": ex,
                "pred_error": final_error,
                "repair_attempted": repair_attempted,
                "repair_success": repair_success,
                "execute_call_count": 2 if repair_attempted else 1,
                "trace": trace,
            }
            traces_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            traces_file.flush()
            records.append(record)
            print(f"{index}: sample_id={sample['sample_id']} pred_success={record['pred_success']} ex={record['ex']} repair_attempted={repair_attempted}")

    metrics = compute_metrics(records)
    metrics["repair_attempt_count"] = sum(1 for record in records if record["repair_attempted"])
    metrics["repair_success_count"] = sum(1 for record in records if record["repair_success"])
    metrics["avg_execute_call_count"] = (
        sum(record["execute_call_count"] for record in records) / len(records) if records else 0.0
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trace_examples(records, doc_path)
    return {
        "metrics": metrics,
        "traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
        "doc_path": str(doc_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the controlled execution-repair agent.")
    parser.add_argument("--db-id", default=DEFAULT_DB_ID)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--day5-input", type=Path, default=DEFAULT_DAY5_INPUT_PATH)
    parser.add_argument("--top-k-schema", type=int, default=DEFAULT_TOP_K_SCHEMA)
    parser.add_argument("--max-repair-rounds", type=int, default=DEFAULT_MAX_REPAIR_ROUNDS)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_controlled_samples(args.day5_input, db_id=args.db_id, limit=args.limit)
    summary = run_controlled_agent(
        samples=samples,
        traces_path=args.traces_output,
        metrics_path=args.metrics_output,
        doc_path=args.doc_output,
        top_k_schema=args.top_k_schema,
        max_repair_rounds=args.max_repair_rounds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
