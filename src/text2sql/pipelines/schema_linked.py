"""Schema-linked baseline."""

import argparse
import json
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import load_bird_dev
from text2sql.db import run_sql, same_result
from text2sql.eval import compute_metrics
from text2sql.llm import call_llm
from text2sql.pipelines.naive import extract_sql
from text2sql.prompts.generation import build_linked_prompt
from text2sql.schema.linker import DEFAULT_TOP_K, SchemaLinker

METHOD_NAME = "schema_linked_promptfix"
DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day3_schema_table_linked_predictions.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day3_schema_table_linked_metrics.json"
DEFAULT_DEBUG_PATH = RESULTS_DIR / "day3_schema_table_linking_debug.jsonl"
DEFAULT_PROMPTFIX_PREDICTIONS_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_predictions.jsonl"
DEFAULT_PROMPTFIX_METRICS_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_metrics.json"
DEFAULT_PROMPTFIX_DEBUG_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_debug.jsonl"
DAY2_METRICS_PATH = RESULTS_DIR / "day2_naive_metrics.json"


def retrieved_column_names(items):
    return [f"{item['table']}.{item['column']}" for item in items]


def load_day2_comparison(day3_metrics):
    if not DAY2_METRICS_PATH.exists():
        return {}
    day2_metrics = json.loads(DAY2_METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "naive_full_schema": {
            "total": day2_metrics.get("total"),
            "VSR": day2_metrics.get("VSR"),
            "EX": day2_metrics.get("EX"),
            "difficulty_breakdown": day2_metrics.get("difficulty_breakdown", {}),
        },
        "schema_linked_promptfix": {
            "total": day3_metrics.get("total"),
            "VSR": day3_metrics.get("VSR"),
            "EX": day3_metrics.get("EX"),
            "difficulty_breakdown": day3_metrics.get("difficulty_breakdown", {}),
        },
        "delta": {
            "VSR": day3_metrics.get("VSR", 0.0) - day2_metrics.get("VSR", 0.0),
            "EX": day3_metrics.get("EX", 0.0) - day2_metrics.get("EX", 0.0),
        },
    }


def run_schema_linked(samples, predictions_path, metrics_path, debug_path, top_k=DEFAULT_TOP_K):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    records = []

    with predictions_path.open("w", encoding="utf-8") as predictions_file, debug_path.open("w", encoding="utf-8") as debug_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)

            linked_items, linked_schema_text = linker_cache[db_path].retrieve(sample["question"], sample["evidence"], top_k=top_k)
            retrieved_columns = retrieved_column_names(linked_items)
            debug_record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "evidence": sample["evidence"],
                "retrieved_columns": retrieved_columns,
                "linked_schema_text": linked_schema_text,
            }
            debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
            debug_file.flush()

            raw_response = ""
            pred_sql = ""
            llm_error = None
            try:
                raw_response = call_llm(build_linked_prompt(sample, linked_schema_text))
                pred_sql = extract_sql(raw_response)
            except Exception as exc:
                llm_error = str(exc)

            pred_result = run_sql(pred_sql, db_path) if pred_sql else {
                "success": False,
                "rows": [],
                "error": llm_error or "Empty SQL prediction",
            }
            gold_result = run_sql(sample["gold_sql"], db_path)
            ex = bool(pred_result["success"] and gold_result["success"] and same_result(pred_result["rows"], gold_result["rows"]))

            record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "evidence": sample["evidence"],
                "gold_sql": sample["gold_sql"],
                "pred_sql": pred_sql,
                "pred_success": pred_result["success"],
                "gold_success": gold_result["success"],
                "pred_error": llm_error or pred_result["error"],
                "gold_error": gold_result["error"],
                "pred_row_count": len(pred_result["rows"]),
                "gold_row_count": len(gold_result["rows"]),
                "pred_rows_preview": pred_result["rows"][:5],
                "gold_rows_preview": gold_result["rows"][:5],
                "ex": ex,
                "raw_response": raw_response,
                "retrieved_columns": retrieved_columns,
                "method": METHOD_NAME,
            }
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            records.append(record)
            print(f"{index}: sample_id={sample['sample_id']} pred_success={record['pred_success']} ex={ex}")

    metrics = compute_metrics(records)
    metrics["comparison_with_day2"] = load_day2_comparison(metrics)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "debug_path": str(debug_path),
    }


run_schema_linked_baseline = run_schema_linked
_retrieved_column_names = retrieved_column_names
_load_day2_comparison = load_day2_comparison


def print_debug_preview(db_id, limit, top_k):
    samples = load_bird_dev(limit=limit, db_id=db_id)
    if not samples:
        print("No samples found.")
        return
    linker = SchemaLinker(samples[0]["db_path"])
    for sample in samples:
        linked_items, _ = linker.retrieve(sample["question"], sample["evidence"], top_k=top_k)
        print(f"\nsample_id={sample['sample_id']} difficulty={sample['difficulty']}")
        print(sample["question"])
        for column in retrieved_column_names(linked_items):
            print(f"- {column}")


def write_linking_debug(samples, debug_path, top_k):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    total = 0
    with debug_path.open("w", encoding="utf-8") as debug_file:
        for sample in samples:
            total += 1
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)
            linked_items, linked_schema_text = linker_cache[db_path].retrieve(sample["question"], sample["evidence"], top_k=top_k)
            debug_file.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "db_id": sample["db_id"],
                        "difficulty": sample["difficulty"],
                        "question": sample["question"],
                        "evidence": sample["evidence"],
                        "retrieved_columns": retrieved_column_names(linked_items),
                        "linked_schema_text": linked_schema_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {"total": total, "debug_path": str(debug_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Run schema-linked baseline.")
    parser.add_argument("--db-id", default="california_schools")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--debug-output", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--debug-preview", type=int, default=0)
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument("--promptfix", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.promptfix:
        if args.predictions_output == DEFAULT_PREDICTIONS_PATH:
            args.predictions_output = DEFAULT_PROMPTFIX_PREDICTIONS_PATH
        if args.metrics_output == DEFAULT_METRICS_PATH:
            args.metrics_output = DEFAULT_PROMPTFIX_METRICS_PATH
        if args.debug_output == DEFAULT_DEBUG_PATH:
            args.debug_output = DEFAULT_PROMPTFIX_DEBUG_PATH
    if args.debug_preview:
        print_debug_preview(args.db_id, args.debug_preview, args.top_k)
        return
    samples = load_bird_dev(limit=args.limit, db_id=args.db_id)
    if args.debug_only:
        summary = write_linking_debug(samples=samples, debug_path=args.debug_output, top_k=args.top_k)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    summary = run_schema_linked(samples, args.predictions_output, args.metrics_output, args.debug_output, args.top_k)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
