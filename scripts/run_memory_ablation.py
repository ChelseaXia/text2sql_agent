"""Run a small memory ablation on stratified iterative-agent samples."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from text2sql.agents.iterative_agent import run_iterative_agent
from text2sql.data import load_eval_manifest


DEFAULT_MANIFEST = Path("results/iterative_agent/stratified_300_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/iterative_agent")
DEFAULT_METRICS_OUTPUT = DEFAULT_OUTPUT_DIR / "memory_ablation_metrics.json"
DEFAULT_CASES_OUTPUT = DEFAULT_OUTPUT_DIR / "memory_ablation_cases.csv"
DEFAULT_TRACES_OUTPUT = DEFAULT_OUTPUT_DIR / "memory_ablation_traces.jsonl"
PREFERRED_DBS = ["financial", "formula_1", "california_schools"]


def select_databases(rows, min_questions=10):
    counts = Counter(row["db_id"] for row in rows)
    selected = [db_id for db_id in PREFERRED_DBS if counts[db_id] >= min_questions]
    if len(selected) < 3:
        for db_id, count in counts.most_common():
            if count >= min_questions and db_id not in selected:
                selected.append(db_id)
            if len(selected) >= 3:
                break
    return selected[:3]


def selected_samples(rows, selected_dbs, seed=42, per_db_limit=20):
    rng = random.Random(seed)
    output = []
    for db_id in selected_dbs:
        db_rows = [row for row in rows if row["db_id"] == db_id]
        rng.shuffle(db_rows)
        for index, row in enumerate(db_rows[:per_db_limit], start=1):
            copied = dict(row)
            copied["memory_ablation_order"] = index
            output.append(copied)
    return output


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def pct(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def finished(row):
    return row.get("final_sql_source") == "finish_tool"


def summarize_records(records):
    total = len(records)
    return {
        "sample_count": total,
        "EX": pct(sum(bool(row.get("ex")) for row in records), total),
        "VSR": pct(sum(bool(row.get("pred_success")) for row in records), total),
        "finish_rate": pct(sum(finished(row) for row in records), total),
        "avg_tool_calls": pct(sum(row.get("tool_call_count", 0) for row in records), total),
        "avg_execute_calls": pct(sum(row.get("execute_call_count", 0) for row in records), total),
        "inspect_table_call_count": sum(
            1 for row in records for event in row.get("trace", []) if event.get("action") == "inspect_table"
        ),
        "search_column_values_call_count": sum(row.get("search_column_values_count", 0) for row in records),
        "memory_hit_count": sum(row.get("memory_hit_count", 0) for row in records),
        "memory_access_count": sum(row.get("memory_access_count", 0) for row in records),
        "memory_hit_rate": pct(
            sum(row.get("memory_hit_count", 0) for row in records),
            sum(row.get("memory_access_count", 0) for row in records),
        ),
        "memory_write_count": sum(row.get("memory_write_count", 0) for row in records),
    }


def first_vs_later(records):
    first = [row for row in records if row.get("memory_ablation_order") == 1]
    later = [row for row in records if row.get("memory_ablation_order", 1) > 1]
    return {
        "first_question": summarize_records(first),
        "second_and_later": summarize_records(later),
    }


def compute_ablation_metrics(off_records, on_records, selected_dbs):
    by_db = {}
    for db_id in selected_dbs:
        off_db = [row for row in off_records if row["db_id"] == db_id]
        on_db = [row for row in on_records if row["db_id"] == db_id]
        off_summary = summarize_records(off_db)
        on_summary = summarize_records(on_db)
        by_db[db_id] = {
            "memory_off": off_summary,
            "memory_on": on_summary,
            "delta_EX": on_summary["EX"] - off_summary["EX"],
            "delta_VSR": on_summary["VSR"] - off_summary["VSR"],
            "delta_finish_rate": on_summary["finish_rate"] - off_summary["finish_rate"],
            "avg_tool_calls_reduction": off_summary["avg_tool_calls"] - on_summary["avg_tool_calls"],
            "avg_execute_calls_reduction": off_summary["avg_execute_calls"] - on_summary["avg_execute_calls"],
            "inspect_table_call_reduction": off_summary["inspect_table_call_count"] - on_summary["inspect_table_call_count"],
            "search_column_values_call_reduction": (
                off_summary["search_column_values_call_count"] - on_summary["search_column_values_call_count"]
            ),
        }

    off_summary = summarize_records(off_records)
    on_summary = summarize_records(on_records)
    return {
        "selected_dbs": selected_dbs,
        "memory_off": off_summary,
        "memory_on": on_summary,
        "overall_delta": {
            "delta_EX": on_summary["EX"] - off_summary["EX"],
            "delta_VSR": on_summary["VSR"] - off_summary["VSR"],
            "delta_finish_rate": on_summary["finish_rate"] - off_summary["finish_rate"],
            "avg_tool_calls_reduction": off_summary["avg_tool_calls"] - on_summary["avg_tool_calls"],
            "avg_execute_calls_reduction": off_summary["avg_execute_calls"] - on_summary["avg_execute_calls"],
            "inspect_table_call_reduction": off_summary["inspect_table_call_count"] - on_summary["inspect_table_call_count"],
            "search_column_values_call_reduction": (
                off_summary["search_column_values_call_count"] - on_summary["search_column_values_call_count"]
            ),
        },
        "first_vs_second_and_later": {
            "memory_off": first_vs_later(off_records),
            "memory_on": first_vs_later(on_records),
        },
        "by_db": by_db,
    }


def write_cases_csv(off_records, on_records, path):
    off_by_id = {row["sample_id"]: row for row in off_records}
    rows = []
    for on_row in on_records:
        off_row = off_by_id[on_row["sample_id"]]
        rows.append(
            {
                "sample_id": on_row["sample_id"],
                "db_id": on_row["db_id"],
                "difficulty": on_row.get("difficulty", ""),
                "memory_ablation_order": on_row.get("memory_ablation_order", ""),
                "question": on_row["question"],
                "memory_off_correct": bool(off_row.get("ex")),
                "memory_on_correct": bool(on_row.get("ex")),
                "memory_off_vsr": bool(off_row.get("pred_success")),
                "memory_on_vsr": bool(on_row.get("pred_success")),
                "memory_off_tool_calls": off_row.get("tool_call_count", 0),
                "memory_on_tool_calls": on_row.get("tool_call_count", 0),
                "memory_off_execute_calls": off_row.get("execute_call_count", 0),
                "memory_on_execute_calls": on_row.get("execute_call_count", 0),
                "memory_hit_count": on_row.get("memory_hit_count", 0),
                "memory_write_count": on_row.get("memory_write_count", 0),
                "memory_on_final_sql": on_row.get("final_sql", ""),
                "memory_off_final_sql": off_row.get("final_sql", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_traces(off_records, on_records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for mode, records in (("off", off_records), ("episodic", on_records)):
            for record in records:
                for event in record.get("trace", []):
                    event = dict(event)
                    event["memory_ablation_mode"] = mode
                    event["db_id"] = record["db_id"]
                    event["difficulty"] = record.get("difficulty")
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_mode(samples, mode, output_dir):
    predictions_path = output_dir / f"memory_ablation_{mode}_predictions.jsonl"
    metrics_path = output_dir / f"memory_ablation_{mode}_raw_metrics.json"
    traces_path = output_dir / f"memory_ablation_{mode}_raw_traces.jsonl"
    run_iterative_agent(
        samples=samples,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        traces_path=traces_path,
        memory_mode="off" if mode == "off" else "episodic",
    )
    return load_jsonl(predictions_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small working/episodic memory ablation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASES_OUTPUT)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-db-limit", type=int, default=20)
    parser.add_argument("--min-questions", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_eval_manifest(args.manifest)
    selected_dbs = select_databases(rows, min_questions=args.min_questions)
    samples = selected_samples(rows, selected_dbs, seed=args.seed, per_db_limit=args.per_db_limit)
    output_dir = args.metrics_output.parent

    off_records = run_mode(samples, "off", output_dir)
    on_records = run_mode(samples, "episodic", output_dir)
    metrics = compute_ablation_metrics(off_records, on_records, selected_dbs)
    metrics["seed"] = args.seed
    metrics["per_db_limit"] = args.per_db_limit
    metrics["manifest"] = str(args.manifest)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_cases_csv(off_records, on_records, args.cases_output)
    write_combined_traces(off_records, on_records, args.traces_output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
