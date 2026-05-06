"""Strict repair protocol that starts from saved schema-linked predictions."""

import argparse
import csv
import json
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import get_db_path, load_eval_manifest
from text2sql.db import run_sql, same_result
from text2sql.eval import compute_metrics, load_jsonl
from text2sql.llm import call_llm
from text2sql.pipelines.naive import extract_sql
from text2sql.prompts.repair import build_repair_prompt
from text2sql.schema.linker import DEFAULT_LINKER_MODE, SchemaLinker

METHOD_NAME = "strict_execution_repair"
DEFAULT_DAY3_INPUT_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_predictions.jsonl"
DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day5_repair_from_day3_predictions.jsonl"
DEFAULT_TRACES_PATH = RESULTS_DIR / "day5_repair_from_day3_traces.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day5_repair_from_day3_metrics.json"
DEFAULT_PAIRWISE_PATH = RESULTS_DIR / "day5_repair_from_day3_pairwise_compare.csv"


def generate_repaired_sql(question, evidence, linked_schema_text, previous_sql, sqlite_error):
    raw_response = ""
    try:
        raw_response = call_llm(
            build_repair_prompt(
                question=question,
                evidence=evidence,
                linked_schema_text=linked_schema_text,
                previous_sql=previous_sql,
                sqlite_error=sqlite_error,
            )
        )
        return extract_sql(raw_response), raw_response, None
    except Exception as exc:
        return "", "", str(exc)


def execute_with_trace(sql, db_path, llm_error=None):
    if not sql:
        return {"success": False, "rows": [], "error": llm_error or "Empty SQL prediction"}
    return run_sql(sql, db_path)


def determine_change_type(day3_ex, strict_ex):
    if day3_ex and strict_ex:
        return "same_correct"
    if (not day3_ex) and (not strict_ex):
        return "same_wrong"
    if (not day3_ex) and strict_ex:
        return "improved"
    return "regressed"


def write_pairwise_compare(path, rows):
    fieldnames = [
        "sample_id",
        "difficulty",
        "day3_ex",
        "strict_repair_ex",
        "day3_pred_success",
        "strict_final_success",
        "change_type",
        "question",
        "initial_pred_sql",
        "final_pred_sql",
        "initial_error",
        "final_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def resolve_day3_records(day3_predictions_path, manifest_path=None, db_id=None, limit=None):
    day3_records = load_jsonl(day3_predictions_path)
    if manifest_path is None:
        return day3_records

    manifest_samples = load_eval_manifest(manifest_path, db_id=db_id, limit=limit)
    day3_by_sample_id = {row["sample_id"]: row for row in day3_records}
    missing = [sample["sample_id"] for sample in manifest_samples if sample["sample_id"] not in day3_by_sample_id]
    if missing:
        raise KeyError(f"Missing sample_ids in Day 3 input for manifest: {missing[:10]}")

    aligned_records = []
    for sample in manifest_samples:
        row = dict(day3_by_sample_id[sample["sample_id"]])
        row["db_id"] = sample["db_id"]
        row["db_path"] = sample["db_path"]
        row["difficulty"] = sample["difficulty"]
        row["question"] = sample["question"]
        row["gold_sql"] = sample["gold_sql"]
        row["evidence"] = sample["evidence"]
        aligned_records.append(row)
    return aligned_records


def run_strict_execution_repair(
    day3_predictions_path,
    predictions_path,
    traces_path,
    metrics_path,
    pairwise_path,
    top_k_schema,
    max_repair_rounds,
    manifest_path=None,
    db_id=None,
    limit=None,
    schema_linker_mode=DEFAULT_LINKER_MODE,
    embedding_model_path=None,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    day3_records = resolve_day3_records(day3_predictions_path, manifest_path=manifest_path, db_id=db_id, limit=limit)
    linker_cache = {}
    metric_rows = []
    pairwise_rows = []
    repair_attempt_count = 0
    repair_success_count = 0
    repaired_sample_count = 0
    total_repair_rounds = 0

    with predictions_path.open("w", encoding="utf-8") as predictions_file, traces_path.open("w", encoding="utf-8") as traces_file:
        for index, day3_row in enumerate(day3_records, start=1):
            db_path = day3_row.get("db_path") or str(get_db_path(day3_row["db_id"]))
            initial_pred_sql = day3_row.get("pred_sql", "")
            initial_success = bool(day3_row.get("pred_success"))
            initial_error = day3_row.get("pred_error")
            day3_ex = bool(day3_row.get("ex"))

            attempts = [
                {
                    "round": 0,
                    "sql": initial_pred_sql,
                    "success": initial_success,
                    "error": initial_error,
                    "raw_response": day3_row.get("raw_response", ""),
                    "source": "day3_promptfix",
                }
            ]

            final_pred_sql = initial_pred_sql
            final_success = initial_success
            final_error = initial_error
            final_rows_preview = day3_row.get("pred_rows_preview", [])
            final_row_count = day3_row.get("pred_row_count", 0)
            repair_rounds = 0
            repaired = False
            linked_schema_text = None

            if not initial_success:
                repaired = True
                repaired_sample_count += 1
                if db_path not in linker_cache:
                    linker_cache[db_path] = SchemaLinker(
                        db_path,
                        mode=schema_linker_mode,
                        dense_model_path=embedding_model_path,
                    )
                _, linked_schema_text = linker_cache[db_path].retrieve(day3_row["question"], day3_row.get("evidence", ""), top_k=top_k_schema)

                previous_sql = initial_pred_sql
                previous_error = initial_error or "Unknown SQLite error"
                for repair_round in range(1, max_repair_rounds + 1):
                    repair_attempt_count += 1
                    repair_rounds += 1
                    total_repair_rounds += 1
                    repaired_sql, raw_response, llm_error = generate_repaired_sql(
                        day3_row["question"],
                        day3_row.get("evidence", ""),
                        linked_schema_text,
                        previous_sql,
                        previous_error,
                    )
                    repaired_result = execute_with_trace(repaired_sql, db_path, llm_error)
                    attempts.append(
                        {
                            "round": repair_round,
                            "sql": repaired_sql,
                            "success": repaired_result["success"],
                            "error": repaired_result["error"],
                            "raw_response": raw_response,
                            "llm_error": llm_error,
                            "source": METHOD_NAME,
                        }
                    )
                    final_pred_sql = repaired_sql
                    final_success = repaired_result["success"]
                    final_error = repaired_result["error"]
                    final_rows_preview = repaired_result["rows"][:5]
                    final_row_count = len(repaired_result["rows"])
                    previous_sql = repaired_sql
                    previous_error = repaired_result["error"] or "Unknown SQLite error"
                    if repaired_result["success"]:
                        repair_success_count += 1
                        break

            gold_sql = day3_row["gold_sql"]
            gold_result = run_sql(gold_sql, db_path)
            final_rows = run_sql(final_pred_sql, db_path)["rows"] if final_success else []
            strict_ex = bool(final_success and gold_result["success"] and same_result(final_rows, gold_result["rows"]))

            prediction_record = {
                "sample_id": day3_row["sample_id"],
                "db_id": day3_row["db_id"],
                "db_path": db_path,
                "difficulty": day3_row["difficulty"],
                "question": day3_row["question"],
                "evidence": day3_row.get("evidence", ""),
                "gold_sql": gold_sql,
                "initial_pred_sql": initial_pred_sql,
                "final_pred_sql": final_pred_sql,
                "predicted_sql": final_pred_sql,
                "final_sql": final_pred_sql,
                "initial_success": initial_success,
                "final_success": final_success,
                "pred_success": final_success,
                "is_executable": final_success,
                "day3_pred_success": initial_success,
                "error": None if final_success else (final_error or initial_error or "Unknown prediction failure"),
                "failure_reason": None if final_success else (final_error or initial_error or "Unknown prediction failure"),
                "initial_error": initial_error,
                "final_error": final_error,
                "repaired": repaired,
                "repair_rounds": repair_rounds,
                "gold_success": gold_result["success"],
                "ex": strict_ex,
                "is_correct": strict_ex,
                "day3_ex": day3_ex,
                "pred_row_count": final_row_count,
                "gold_row_count": len(gold_result["rows"]),
                "pred_rows_preview": final_rows_preview,
                "gold_rows_preview": gold_result["rows"][:5],
                "schema_linker_mode": schema_linker_mode,
                "method": METHOD_NAME,
            }
            trace_record = {
                "sample_id": day3_row["sample_id"],
                "db_id": day3_row["db_id"],
                "difficulty": day3_row["difficulty"],
                "question": day3_row["question"],
                "evidence": day3_row.get("evidence", ""),
                "linked_schema_text": linked_schema_text,
                "attempts": attempts,
            }
            pairwise_rows.append(
                {
                    "sample_id": day3_row["sample_id"],
                    "difficulty": day3_row["difficulty"],
                    "day3_ex": day3_ex,
                    "strict_repair_ex": strict_ex,
                    "day3_pred_success": initial_success,
                    "strict_final_success": final_success,
                    "change_type": determine_change_type(day3_ex, strict_ex),
                    "question": day3_row["question"],
                    "initial_pred_sql": initial_pred_sql,
                    "final_pred_sql": final_pred_sql,
                    "initial_error": initial_error,
                    "final_error": final_error,
                }
            )
            predictions_file.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            traces_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            traces_file.flush()
            metric_rows.append(prediction_record)
            print(f"{index}: sample_id={day3_row['sample_id']} day3_pred_success={initial_success} final_success={final_success} ex={strict_ex}")

    metrics = compute_metrics(metric_rows)
    metrics["repair_attempt_count"] = repair_attempt_count
    metrics["repair_success_count"] = repair_success_count
    metrics["avg_repair_rounds"] = total_repair_rounds / repaired_sample_count if repaired_sample_count else 0.0
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_pairwise_compare(pairwise_path, pairwise_rows)
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
        "pairwise_path": str(pairwise_path),
    }


run_day5_5_from_day3 = run_strict_execution_repair


def parse_args():
    parser = argparse.ArgumentParser(description="Run strict repair from Day 3 predictions.")
    parser.add_argument("--day3-input", type=Path, default=DEFAULT_DAY3_INPUT_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--db-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--schema-linker-mode", choices=("hybrid", "bm25"), default=DEFAULT_LINKER_MODE)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--pairwise-output", type=Path, default=DEFAULT_PAIRWISE_PATH)
    parser.add_argument("--top-k-schema", type=int, default=30)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_strict_execution_repair(
        day3_predictions_path=args.day3_input,
        predictions_path=args.predictions_output,
        traces_path=args.traces_output,
        metrics_path=args.metrics_output,
        pairwise_path=args.pairwise_output,
        top_k_schema=args.top_k_schema,
        max_repair_rounds=args.max_repair_rounds,
        manifest_path=args.manifest,
        db_id=args.db_id,
        limit=args.limit,
        schema_linker_mode=args.schema_linker_mode,
        embedding_model_path=args.embedding_model_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
