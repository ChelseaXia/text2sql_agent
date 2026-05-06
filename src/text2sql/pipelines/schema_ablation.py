"""Schema format ablation with DDL-style serialization."""

import argparse
import json
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import resolve_eval_samples
from text2sql.db import run_sql, same_result
from text2sql.eval import compute_metrics
from text2sql.llm import call_llm
from text2sql.pipelines.naive import extract_sql
from text2sql.pipelines.schema_linked import load_day2_comparison, retrieved_column_names
from text2sql.prompts.generation import build_linked_prompt
from text2sql.schema.formatters import format_table_linked_schema_ddl
from text2sql.schema.linker import DEFAULT_LINKER_MODE, SchemaLinker

METHOD_NAME = "schema_format_ddl_ablation"
DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day6_schema_ddl_promptfix_predictions.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day6_schema_ddl_promptfix_metrics.json"
DEFAULT_DEBUG_PATH = RESULTS_DIR / "day6_schema_ddl_promptfix_debug.jsonl"


def run_schema_format_ddl_ablation(
    samples,
    predictions_path,
    metrics_path,
    debug_path,
    top_k_schema,
    schema_linker_mode=DEFAULT_LINKER_MODE,
    embedding_model_path=None,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    records = []
    with predictions_path.open("w", encoding="utf-8") as predictions_file, debug_path.open("w", encoding="utf-8") as debug_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(
                    db_path,
                    mode=schema_linker_mode,
                    dense_model_path=embedding_model_path,
                )

            linked_items, _ = linker_cache[db_path].retrieve(sample["question"], sample["evidence"], top_k=top_k_schema)
            linked_schema_text, selected_tables = format_table_linked_schema_ddl(linked_items, linker_cache[db_path].items)
            retrieved_columns = retrieved_column_names(linked_items)
            debug_file.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "question": sample["question"],
                        "retrieved_columns": retrieved_columns,
                        "selected_tables": selected_tables,
                        "linked_schema_text": linked_schema_text,
                        "schema_format": "ddl",
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
            failure_reason = None if pred_result["success"] else (llm_error or pred_result["error"] or "Unknown prediction failure")

            record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "db_path": db_path,
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "evidence": sample["evidence"],
                "gold_sql": sample["gold_sql"],
                "pred_sql": pred_sql,
                "predicted_sql": pred_sql,
                "final_sql": pred_sql,
                "pred_success": pred_result["success"],
                "is_executable": pred_result["success"],
                "gold_success": gold_result["success"],
                "error": failure_reason,
                "failure_reason": failure_reason,
                "llm_error": llm_error,
                "pred_error": llm_error or pred_result["error"],
                "gold_error": gold_result["error"],
                "pred_row_count": len(pred_result["rows"]),
                "gold_row_count": len(gold_result["rows"]),
                "pred_rows_preview": pred_result["rows"][:5],
                "gold_rows_preview": gold_result["rows"][:5],
                "ex": ex,
                "is_correct": ex,
                "raw_response": raw_response,
                "retrieved_columns": retrieved_columns,
                "selected_tables": selected_tables,
                "schema_format": "ddl",
                "schema_linker_mode": schema_linker_mode,
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


run_schema_ddl_baseline = run_schema_format_ddl_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="Run schema DDL promptfix baseline.")
    parser.add_argument("--db-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--top-k-schema", type=int, default=30)
    parser.add_argument("--schema-linker-mode", choices=("hybrid", "bm25"), default=DEFAULT_LINKER_MODE)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--debug-output", type=Path, default=DEFAULT_DEBUG_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.manifest is None and args.db_id is None:
        args.db_id = "california_schools"
    if args.manifest is None and args.limit is None:
        args.limit = 50
    samples = resolve_eval_samples(limit=args.limit, db_id=args.db_id, manifest_path=args.manifest)
    summary = run_schema_format_ddl_ablation(
        samples=samples,
        predictions_path=args.predictions_output,
        metrics_path=args.metrics_output,
        debug_path=args.debug_output,
        top_k_schema=args.top_k_schema,
        schema_linker_mode=args.schema_linker_mode,
        embedding_model_path=args.embedding_model_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
