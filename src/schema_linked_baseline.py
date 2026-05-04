"""Run the schema-linked Text2SQL baseline on BIRD dev samples."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

try:
    from config import RESULTS_DIR
    from db_utils import run_sql, same_result
    from evaluate import compute_metrics
    from llm_client import call_llm, get_api_key
    from load_bird import load_bird_dev
    from naive_baseline import extract_sql
    from schema_linker import DEFAULT_TOP_K, SchemaLinker
except ModuleNotFoundError:
    from src.config import RESULTS_DIR
    from src.db_utils import run_sql, same_result
    from src.evaluate import compute_metrics
    from src.llm_client import call_llm, get_api_key
    from src.load_bird import load_bird_dev
    from src.naive_baseline import extract_sql
    from src.schema_linker import DEFAULT_TOP_K, SchemaLinker


DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day3_schema_table_linked_predictions.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day3_schema_table_linked_metrics.json"
DEFAULT_DEBUG_PATH = RESULTS_DIR / "day3_schema_table_linking_debug.jsonl"
DEFAULT_PROMPTFIX_PREDICTIONS_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_predictions.jsonl"
DEFAULT_PROMPTFIX_METRICS_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_metrics.json"
DEFAULT_PROMPTFIX_DEBUG_PATH = RESULTS_DIR / "day3_schema_table_linked_promptfix_debug.jsonl"
DAY2_METRICS_PATH = RESULTS_DIR / "day2_naive_metrics.json"


def build_linked_prompt(sample: Dict, linked_schema_text: str) -> str:
    """Construct the schema-linked prompt."""
    evidence = sample.get("evidence") or "None"
    return f"""Given the relevant SQLite schema tables, evidence, and question, write the correct SQLite SQL query.

SQL generation instruction:
- Use exact table and column names from the provided schema.
- Never invent normalized column names such as Enrollment, FRPM_Count, or Percent_Eligible_FRPM if the schema provides columns like `Enrollment (K-12)` or `Percent (%) Eligible FRPM (K-12)`.
- Wrap column names containing spaces, parentheses, hyphens, percent signs, or slashes with backticks.
- Be careful about table-column association: only use a column under the table where it appears in the schema.

{linked_schema_text}

Evidence:
{evidence}

Question:
{sample["question"]}

Return only the SQL query."""


def _retrieved_column_names(items):
    return [f"{item['table']}.{item['column']}" for item in items]


def _load_day2_comparison(day3_metrics: Dict) -> Dict:
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
        "schema_table_linked": {
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


def run_schema_linked_baseline(
    samples: Iterable[Dict],
    predictions_path: Path,
    metrics_path: Path,
    debug_path: Path,
    top_k: int = DEFAULT_TOP_K,
) -> Dict:
    """Run schema-linked prompting, save predictions/debug, and write metrics."""
    get_api_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    records = []

    with predictions_path.open("w", encoding="utf-8") as predictions_file, debug_path.open(
        "w", encoding="utf-8"
    ) as debug_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)

            linked_items, linked_schema_text = linker_cache[db_path].retrieve(
                sample["question"], sample["evidence"], top_k=top_k
            )
            retrieved_columns = _retrieved_column_names(linked_items)

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

            prompt = build_linked_prompt(sample, linked_schema_text)
            raw_response = ""
            pred_sql = ""
            llm_error = None

            try:
                raw_response = call_llm(prompt)
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
            }
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            records.append(record)
            print(f"{index}: sample_id={sample['sample_id']} pred_success={record['pred_success']} ex={ex}")

    metrics = compute_metrics(records)
    metrics["comparison_with_day2"] = _load_day2_comparison(metrics)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "debug_path": str(debug_path),
    }


def print_debug_preview(db_id: str, limit: int, top_k: int) -> None:
    """Print retrieved columns for a small manual inspection batch."""
    samples = load_bird_dev(limit=limit, db_id=db_id)
    if not samples:
        print("No samples found.")
        return

    linker = SchemaLinker(samples[0]["db_path"])
    for sample in samples:
        linked_items, _ = linker.retrieve(sample["question"], sample["evidence"], top_k=top_k)
        print(f"\nsample_id={sample['sample_id']} difficulty={sample['difficulty']}")
        print(sample["question"])
        for column in _retrieved_column_names(linked_items):
            print(f"- {column}")


def write_linking_debug(samples: Iterable[Dict], debug_path: Path, top_k: int) -> Dict:
    """Write retrieved columns and linked schema text without calling the LLM."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    total = 0

    with debug_path.open("w", encoding="utf-8") as debug_file:
        for sample in samples:
            total += 1
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = SchemaLinker(db_path)

            linked_items, linked_schema_text = linker_cache[db_path].retrieve(
                sample["question"], sample["evidence"], top_k=top_k
            )
            debug_record = {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "evidence": sample["evidence"],
                "retrieved_columns": _retrieved_column_names(linked_items),
                "linked_schema_text": linked_schema_text,
            }
            debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

    return {"total": total, "debug_path": str(debug_path)}


def parse_args() -> argparse.Namespace:
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


def main() -> None:
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

    summary = run_schema_linked_baseline(
        samples=samples,
        predictions_path=args.predictions_output,
        metrics_path=args.metrics_output,
        debug_path=args.debug_output,
        top_k=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
