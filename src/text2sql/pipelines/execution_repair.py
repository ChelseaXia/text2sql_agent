"""Execution self-repair baseline."""

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
from text2sql.prompts.repair import build_repair_prompt
from text2sql.schema.linker import SchemaLinker

DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day5_repair_predictions.jsonl"
DEFAULT_TRACES_PATH = RESULTS_DIR / "day5_repair_traces.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day5_repair_metrics.json"


def generate_initial_sql(sample, linked_schema_text):
    raw_response = ""
    try:
        raw_response = call_llm(build_linked_prompt(sample, linked_schema_text))
        return extract_sql(raw_response), raw_response
    except Exception as exc:
        return "", str(exc)


def generate_repaired_sql(sample, linked_schema_text, previous_sql, previous_error):
    raw_response = ""
    try:
        raw_response = call_llm(
            build_repair_prompt(
                question=sample["question"],
                evidence=sample["evidence"],
                linked_schema_text=linked_schema_text,
                previous_sql=previous_sql,
                sqlite_error=previous_error,
            )
        )
        return extract_sql(raw_response), raw_response
    except Exception as exc:
        return "", str(exc)


def execute_with_trace(sql, db_path, llm_error=None):
    if not sql:
        return {"success": False, "rows": [], "error": llm_error or "Empty SQL prediction"}
    return run_sql(sql, db_path)


def run_execution_repair_baseline(samples, predictions_path, traces_path, metrics_path, top_k_schema, max_repair_rounds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    records = []
    repair_attempt_count = 0
    repair_success_count = 0
    repaired_sample_count = 0
    total_repair_rounds = 0

    with predictions_path.open("w", encoding="utf-8") as predictions_file, traces_path.open("w", encoding="utf-8") as traces_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)
            _, linked_schema_text = linker_cache[db_path].retrieve(sample["question"], sample["evidence"], top_k=top_k_schema)

            attempts = []
            initial_pred_sql, initial_response_or_error = generate_initial_sql(sample, linked_schema_text)
            initial_result = execute_with_trace(initial_pred_sql, db_path, None if initial_pred_sql else initial_response_or_error)
            attempts.append({"round": 0, "sql": initial_pred_sql, "success": initial_result["success"], "error": initial_result["error"]})

            final_pred_sql = initial_pred_sql
            final_result = initial_result
            repaired = False
            repair_rounds = 0
            if not initial_result["success"]:
                repaired = True
                repaired_sample_count += 1
                previous_sql = initial_pred_sql
                previous_error = initial_result["error"] or "Unknown SQLite error"
                for repair_round in range(1, max_repair_rounds + 1):
                    repair_attempt_count += 1
                    repair_rounds += 1
                    total_repair_rounds += 1
                    repaired_sql, repair_response_or_error = generate_repaired_sql(sample, linked_schema_text, previous_sql, previous_error)
                    repaired_result = execute_with_trace(repaired_sql, db_path, None if repaired_sql else repair_response_or_error)
                    attempts.append({"round": repair_round, "sql": repaired_sql, "success": repaired_result["success"], "error": repaired_result["error"]})
                    final_pred_sql = repaired_sql
                    final_result = repaired_result
                    previous_sql = repaired_sql
                    previous_error = repaired_result["error"] or "Unknown SQLite error"
                    if repaired_result["success"]:
                        repair_success_count += 1
                        break

            gold_result = run_sql(sample["gold_sql"], db_path)
            ex = bool(final_result["success"] and gold_result["success"] and same_result(final_result["rows"], gold_result["rows"]))
            prediction_record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "evidence": sample["evidence"],
                "gold_sql": sample["gold_sql"],
                "initial_pred_sql": initial_pred_sql,
                "final_pred_sql": final_pred_sql,
                "initial_success": initial_result["success"],
                "final_success": final_result["success"],
                "pred_success": final_result["success"],
                "initial_error": initial_result["error"],
                "final_error": final_result["error"],
                "repaired": repaired,
                "repair_rounds": repair_rounds,
                "gold_success": gold_result["success"],
                "ex": ex,
                "pred_row_count": len(final_result["rows"]),
                "gold_row_count": len(gold_result["rows"]),
                "pred_rows_preview": final_result["rows"][:5],
                "gold_rows_preview": gold_result["rows"][:5],
            }
            trace_record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "question": sample["question"],
                "difficulty": sample["difficulty"],
                "linked_schema_text": linked_schema_text,
                "attempts": attempts,
            }
            predictions_file.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            traces_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            traces_file.flush()
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "db_id": sample["db_id"],
                    "difficulty": sample["difficulty"],
                    "pred_success": final_result["success"],
                    "gold_success": gold_result["success"],
                    "ex": ex,
                }
            )
            print(f"{index}: sample_id={sample['sample_id']} initial_success={initial_result['success']} final_success={final_result['success']} ex={ex}")

    metrics = compute_metrics(records)
    metrics["repair_attempt_count"] = repair_attempt_count
    metrics["repair_success_count"] = repair_success_count
    metrics["avg_repair_rounds"] = total_repair_rounds / repaired_sample_count if repaired_sample_count else 0.0
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run execution self-repair baseline.")
    parser.add_argument("--db-id", default="california_schools")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k-schema", type=int, default=30)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_bird_dev(limit=args.limit, db_id=args.db_id)
    summary = run_execution_repair_baseline(
        samples=samples,
        predictions_path=args.predictions_output,
        traces_path=args.traces_output,
        metrics_path=args.metrics_output,
        top_k_schema=args.top_k_schema,
        max_repair_rounds=args.max_repair_rounds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
