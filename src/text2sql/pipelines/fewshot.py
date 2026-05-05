"""Schema-linked + few-shot baseline."""

import argparse
import json
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import load_bird_dev
from text2sql.db import run_sql, same_result
from text2sql.eval import compute_metrics
from text2sql.llm import call_llm, get_api_key
from text2sql.pipelines.fewshot_retriever import FewShotRetriever
from text2sql.pipelines.naive import extract_sql
from text2sql.pipelines.schema_linked import load_day2_comparison, retrieved_column_names
from text2sql.prompts.generation import build_fewshot_prompt
from text2sql.schema.linker import SchemaLinker

METHOD_NAME = "fewshot_retrieval"
DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day4_fewshot_predictions.jsonl"
DEFAULT_DEBUG_PATH = RESULTS_DIR / "day4_fewshot_debug.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day4_fewshot_metrics.json"
DAY3_PROMPTFIX_METRICS_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_metrics.json"


def load_day3_promptfix_metrics():
    if not DAY3_PROMPTFIX_METRICS_PATH.exists():
        return {}
    return json.loads(DAY3_PROMPTFIX_METRICS_PATH.read_text(encoding="utf-8"))


def build_comparison(metrics):
    day2_comparison = load_day2_comparison(metrics)
    day3_metrics = load_day3_promptfix_metrics()
    return {
        "naive_full_schema": day2_comparison.get("naive_full_schema", {}),
        "schema_linked_promptfix": {
            "total": day3_metrics.get("total"),
            "VSR": day3_metrics.get("VSR"),
            "EX": day3_metrics.get("EX"),
            "difficulty_breakdown": day3_metrics.get("difficulty_breakdown", {}),
        },
        "fewshot_retrieval": {
            "total": metrics.get("total"),
            "VSR": metrics.get("VSR"),
            "EX": metrics.get("EX"),
            "difficulty_breakdown": metrics.get("difficulty_breakdown", {}),
        },
    }


def run_fewshot_retrieval(samples, predictions_path, debug_path, metrics_path, top_k_schema, top_k_examples):
    get_api_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    retriever = FewShotRetriever()
    records = []

    with predictions_path.open("w", encoding="utf-8") as predictions_file, debug_path.open("w", encoding="utf-8") as debug_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)

            linked_items, linked_schema_text = linker_cache[db_path].retrieve(sample["question"], sample["evidence"], top_k=top_k_schema)
            examples = retriever.get_top_k_examples(sample["sample_id"], sample["db_id"], sample["question"], k=top_k_examples)
            debug_file.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "db_id": sample["db_id"],
                        "question": sample["question"],
                        "linked_schema_text": linked_schema_text,
                        "retrieved_columns": retrieved_column_names(linked_items),
                        "retrieved_examples": examples,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            debug_file.flush()

            raw_response = ""
            pred_sql = ""
            llm_error = None
            try:
                raw_response = call_llm(build_fewshot_prompt(sample, linked_schema_text, examples))
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
                "raw_response": raw_response,
                "pred_success": pred_result["success"],
                "gold_success": gold_result["success"],
                "ex": ex,
                "pred_error": llm_error or pred_result["error"],
                "gold_error": gold_result["error"],
                "pred_row_count": len(pred_result["rows"]),
                "gold_row_count": len(gold_result["rows"]),
                "pred_rows_preview": pred_result["rows"][:5],
                "gold_rows_preview": gold_result["rows"][:5],
                "method": METHOD_NAME,
            }
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            records.append(record)
            print(f"{index}: sample_id={sample['sample_id']} pred_success={record['pred_success']} ex={ex}")

    metrics = compute_metrics(records)
    metrics["comparison_with_day2_day3"] = build_comparison(metrics)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "debug_path": str(debug_path),
        "metrics_path": str(metrics_path),
        "example_source": retriever.example_source,
    }


run_fewshot_baseline = run_fewshot_retrieval


def parse_args():
    parser = argparse.ArgumentParser(description="Run schema-linked promptfix + few-shot baseline.")
    parser.add_argument("--db-id", default="california_schools")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k-schema", type=int, default=30)
    parser.add_argument("--top-k-examples", type=int, default=3)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--debug-output", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_bird_dev(limit=args.limit, db_id=args.db_id)
    summary = run_fewshot_retrieval(
        samples=samples,
        predictions_path=args.predictions_output,
        debug_path=args.debug_output,
        metrics_path=args.metrics_output,
        top_k_schema=args.top_k_schema,
        top_k_examples=args.top_k_examples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
