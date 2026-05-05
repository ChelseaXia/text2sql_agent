"""Naive full-schema baseline."""

import argparse
import json
import re
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import load_bird_dev
from text2sql.db import run_sql, same_result
from text2sql.llm import call_llm
from text2sql.prompts.generation import build_naive_prompt
from text2sql.schema.parser import get_full_schema_text

METHOD_NAME = "naive_full_schema"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "day2_naive_predictions.jsonl"


def extract_sql(raw_response):
    text = raw_response.strip()
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE)
    match = re.search(r"\b(WITH|SELECT)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():].strip()
    return text.rstrip("`").strip()


def iter_target_samples(db_id, limit):
    return load_bird_dev(limit=limit, db_id=db_id)


def run_naive_full_schema(samples, output_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schema_cache = {}
    total = 0
    ex_count = 0
    pred_success_count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            total += 1
            db_path = sample["db_path"]
            if db_path not in schema_cache:
                schema_cache[db_path] = get_full_schema_text(db_path)

            prompt = build_naive_prompt(sample, schema_cache[db_path])
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

            if pred_result["success"]:
                pred_success_count += 1
            if ex:
                ex_count += 1

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
                "method": METHOD_NAME,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{total}: sample_id={sample['sample_id']} pred_success={record['pred_success']} ex={ex}")

    return {
        "total": total,
        "pred_success_count": pred_success_count,
        "ex_count": ex_count,
        "output_path": str(output_path),
    }


run_naive_baseline = run_naive_full_schema


def parse_args():
    parser = argparse.ArgumentParser(description="Run naive full-schema baseline.")
    parser.add_argument("--db-id", default="california_schools")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = iter_target_samples(db_id=args.db_id, limit=args.limit)
    summary = run_naive_full_schema(samples=samples, output_path=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
