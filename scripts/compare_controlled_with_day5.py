"""Compare controlled agent outputs against Day5.5 strict repair results."""

import argparse
import csv
import json
import re
from pathlib import Path

from text2sql.config import PROJECT_ROOT, RESULTS_DIR
from text2sql.data import get_db_path
from text2sql.db import run_sql, same_result
from text2sql.eval import load_jsonl

DEFAULT_DAY5_PATH = RESULTS_DIR / "day5_repair_from_day3_predictions.jsonl"
DEFAULT_CONTROLLED_PATH = RESULTS_DIR / "controlled_agent_traces.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "controlled_agent_metrics.json"
DEFAULT_CSV_PATH = RESULTS_DIR / "controlled_vs_day5_diff.csv"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "controlled_vs_day5_diff.md"


def normalize_sql(sql):
    text = (sql or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s*;\s*$", "", text)
    return text.strip()


def sql_same(left, right):
    return normalize_sql(left) == normalize_sql(right)


def load_records_by_sample_id(path):
    rows = load_jsonl(path)
    return {row["sample_id"]: row for row in rows}


def get_day5_final_sql(row):
    return row.get("final_sql") or row.get("final_pred_sql") or row.get("initial_pred_sql") or ""


def get_controlled_trace_step(record, action, step_number=None):
    for step in record.get("trace", []):
        if step.get("action") != action:
            continue
        if step_number is not None and step.get("step") != step_number:
            continue
        return step
    return None


def get_controlled_repair_sql(record):
    step = get_controlled_trace_step(record, "repair_sql", 4)
    if not step:
        return ""
    return step.get("tool_output", {}).get("repaired_sql", "")


def get_controlled_generate_raw_response(record):
    step = get_controlled_trace_step(record, "generate_sql", 2)
    if not step:
        return ""
    return step.get("tool_output", {}).get("raw_response", "")


def get_controlled_repair_raw_response(record):
    step = get_controlled_trace_step(record, "repair_sql", 4)
    if not step:
        return ""
    return step.get("tool_output", {}).get("raw_response", "")


def get_status(row):
    if row["day5_pred_success"] and not row["controlled_pred_success"]:
        return "controlled_execution_failed"
    if row["day5_ex"] and row["controlled_ex"]:
        return "both_correct"
    if row["day5_ex"] and not row["controlled_ex"]:
        return "day5_only_correct"
    if row["controlled_ex"] and not row["day5_ex"]:
        return "controlled_only_correct"
    if not row["sql_same_initial"] or not row["sql_same_final"]:
        return "sql_mismatch"
    return "both_wrong"


def infer_difference_causes(day5_row, controlled_row):
    causes = []

    day5_initial_sql = day5_row.get("initial_pred_sql", "")
    controlled_initial_sql = controlled_row.get("initial_sql", "")
    day5_final_sql = get_day5_final_sql(day5_row)
    controlled_final_sql = controlled_row.get("final_sql", "")

    day5_repair_sql = day5_row.get("final_pred_sql", "")
    controlled_repair_sql = get_controlled_repair_sql(controlled_row)

    controlled_initial_raw = get_controlled_generate_raw_response(controlled_row)
    controlled_repair_raw = get_controlled_repair_raw_response(controlled_row)

    if not sql_same(day5_initial_sql, controlled_initial_sql):
        causes.append("initial SQL mismatch")

    repair_signal = bool(day5_row.get("repaired")) or bool(controlled_row.get("repair_attempted"))
    if repair_signal and not sql_same(day5_repair_sql, controlled_repair_sql):
        causes.append("repair SQL mismatch")

    if controlled_initial_raw and sql_same(controlled_initial_raw, day5_initial_sql) and not sql_same(controlled_initial_sql, day5_initial_sql):
        causes.append("parsing / extract_sql mismatch")
    elif controlled_repair_raw and sql_same(controlled_repair_raw, day5_repair_sql) and not sql_same(controlled_repair_sql, day5_repair_sql):
        causes.append("parsing / extract_sql mismatch")

    if not sql_same(day5_final_sql, controlled_final_sql):
        final_selection_pattern = (
            sql_same(day5_initial_sql, controlled_initial_sql)
            and bool(controlled_row.get("repair_attempted"))
            and not bool(controlled_row.get("repair_success"))
            and sql_same(controlled_final_sql, controlled_initial_sql)
            and not sql_same(day5_final_sql, day5_initial_sql)
        )
        if final_selection_pattern:
            causes.append("final SQL field selection mismatch")

    if day5_row.get("pred_success") and controlled_row.get("pred_success"):
        day5_execution = run_sql(day5_final_sql, get_db_path(day5_row["db_id"]))
        controlled_execution = run_sql(controlled_final_sql, get_db_path(controlled_row["db_id"]))
        if day5_execution["success"] and controlled_execution["success"]:
            if not same_result(day5_execution["rows"], controlled_execution["rows"]):
                causes.append("execution result mismatch")
        elif day5_execution["success"] != controlled_execution["success"]:
            causes.append("execution result mismatch")
    elif day5_row.get("pred_success") != controlled_row.get("pred_success"):
        causes.append("execution result mismatch")

    return causes


def build_diff_rows(day5_rows, controlled_rows):
    sample_ids = sorted(set(day5_rows) | set(controlled_rows))
    rows = []

    for sample_id in sample_ids:
        day5_row = day5_rows[sample_id]
        controlled_row = controlled_rows[sample_id]
        day5_final_sql = get_day5_final_sql(day5_row)
        controlled_final_sql = controlled_row.get("final_sql", "")
        row = {
            "sample_id": sample_id,
            "difficulty": day5_row.get("difficulty", controlled_row.get("difficulty", "")),
            "question": day5_row.get("question", controlled_row.get("question", "")),
            "day5_pred_sql": day5_row.get("initial_pred_sql", ""),
            "controlled_initial_sql": controlled_row.get("initial_sql", ""),
            "day5_final_sql": day5_final_sql,
            "controlled_final_sql": controlled_final_sql,
            "day5_pred_success": bool(day5_row.get("pred_success")),
            "controlled_pred_success": bool(controlled_row.get("pred_success")),
            "day5_ex": bool(day5_row.get("ex")),
            "controlled_ex": bool(controlled_row.get("ex")),
            "day5_pred_error": day5_row.get("final_error") or day5_row.get("initial_error") or "",
            "controlled_pred_error": controlled_row.get("pred_error") or "",
            "repair_attempted": bool(controlled_row.get("repair_attempted")),
            "repair_success": bool(controlled_row.get("repair_success")),
            "sql_same_initial": sql_same(day5_row.get("initial_pred_sql", ""), controlled_row.get("initial_sql", "")),
            "sql_same_final": sql_same(day5_final_sql, controlled_final_sql),
        }
        row["status"] = get_status(row)
        row["difference_causes"] = "; ".join(infer_difference_causes(day5_row, controlled_row))
        rows.append(row)

    return rows


def summarize_rows(rows):
    summary = {
        "total": len(rows),
        "same_ex_count": sum(1 for row in rows if row["day5_ex"] == row["controlled_ex"]),
        "day5_only_correct_count": sum(1 for row in rows if row["status"] == "day5_only_correct"),
        "controlled_only_correct_count": sum(1 for row in rows if row["status"] == "controlled_only_correct"),
        "controlled_execution_failed_count": sum(1 for row in rows if row["status"] == "controlled_execution_failed"),
        "initial_sql_mismatch_count": sum(1 for row in rows if not row["sql_same_initial"]),
        "final_sql_mismatch_count": sum(1 for row in rows if not row["sql_same_final"]),
        "repair_success_mismatch_count": 0,
        "difference_causes": {
            "initial SQL mismatch": sum("initial SQL mismatch" in row["difference_causes"] for row in rows),
            "repair SQL mismatch": sum("repair SQL mismatch" in row["difference_causes"] for row in rows),
            "execution result mismatch": sum("execution result mismatch" in row["difference_causes"] for row in rows),
            "parsing / extract_sql mismatch": sum("parsing / extract_sql mismatch" in row["difference_causes"] for row in rows),
            "final SQL field selection mismatch": sum("final SQL field selection mismatch" in row["difference_causes"] for row in rows),
        },
    }
    return summary


def collect_repair_success_mismatch_count(day5_rows, controlled_rows):
    mismatches = 0
    for sample_id, day5_row in day5_rows.items():
        controlled_row = controlled_rows[sample_id]
        day5_repair_success = bool(day5_row.get("repaired")) and bool(day5_row.get("pred_success")) and not bool(day5_row.get("initial_success"))
        if day5_repair_success != bool(controlled_row.get("repair_success")):
            mismatches += 1
    return mismatches


def write_csv(path, rows):
    fieldnames = [
        "sample_id",
        "difficulty",
        "question",
        "day5_pred_sql",
        "controlled_initial_sql",
        "day5_final_sql",
        "controlled_final_sql",
        "day5_pred_success",
        "controlled_pred_success",
        "day5_ex",
        "controlled_ex",
        "day5_pred_error",
        "controlled_pred_error",
        "repair_attempted",
        "repair_success",
        "sql_same_initial",
        "sql_same_final",
        "status",
        "difference_causes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize_case(row):
    return [
        f"### Sample {row['sample_id']}",
        "",
        f"- Difficulty: `{row['difficulty']}`",
        f"- Status: `{row['status']}`",
        f"- Causes: `{row['difference_causes'] or 'none identified'}`",
        f"- Day5 pred_success / EX: `{row['day5_pred_success']}` / `{row['day5_ex']}`",
        f"- Controlled pred_success / EX: `{row['controlled_pred_success']}` / `{row['controlled_ex']}`",
        "",
        f"Question: {row['question']}",
        "",
        "Day5 final SQL:",
        "```sql",
        row["day5_final_sql"] or "-- empty --",
        "```",
        "",
        "Controlled final SQL:",
        "```sql",
        row["controlled_final_sql"] or "-- empty --",
        "```",
        "",
    ]


def write_markdown(path, rows, summary, controlled_metrics):
    day5_only = [row for row in rows if row["status"] == "day5_only_correct"]
    controlled_failures = [row for row in rows if row["status"] == "controlled_execution_failed"]
    repair_mismatches = [row for row in rows if "repair SQL mismatch" in row["difference_causes"] or row["repair_success"]]

    lines = [
        "# Controlled Agent vs Day5.5 Diff",
        "",
        "This report compares saved Day5.5 strict-repair predictions with saved controlled-agent traces on the same sample set.",
        "",
        "## Summary",
        "",
        f"- Total: `{summary['total']}`",
        f"- same_ex_count: `{summary['same_ex_count']}`",
        f"- day5_only_correct_count: `{summary['day5_only_correct_count']}`",
        f"- controlled_only_correct_count: `{summary['controlled_only_correct_count']}`",
        f"- controlled_execution_failed_count: `{summary['controlled_execution_failed_count']}`",
        f"- initial_sql_mismatch_count: `{summary['initial_sql_mismatch_count']}`",
        f"- final_sql_mismatch_count: `{summary['final_sql_mismatch_count']}`",
        f"- repair_success_mismatch_count: `{summary['repair_success_mismatch_count']}`",
        "",
        "Difference cause counts:",
        f"- initial SQL mismatch: `{summary['difference_causes']['initial SQL mismatch']}`",
        f"- repair SQL mismatch: `{summary['difference_causes']['repair SQL mismatch']}`",
        f"- execution result mismatch: `{summary['difference_causes']['execution result mismatch']}`",
        f"- parsing / extract_sql mismatch: `{summary['difference_causes']['parsing / extract_sql mismatch']}`",
        f"- final SQL field selection mismatch: `{summary['difference_causes']['final SQL field selection mismatch']}`",
        "",
        "Controlled metrics snapshot:",
        f"- EX: `{controlled_metrics.get('EX')}`",
        f"- VSR: `{controlled_metrics.get('VSR')}`",
        f"- repair_attempt_count: `{controlled_metrics.get('repair_attempt_count')}`",
        f"- repair_success_count: `{controlled_metrics.get('repair_success_count')}`",
        "",
    ]

    if day5_only:
        lines.extend(["## Day5-Only Correct Cases", ""])
        for row in day5_only[:5]:
            lines.extend(summarize_case(row))

    if controlled_failures:
        lines.extend(["## Controlled Execution Failures", ""])
        for row in controlled_failures[:3]:
            lines.extend(summarize_case(row))

    if repair_mismatches:
        lines.extend(["## Repair-Related Cases", ""])
        picked = []
        for row in repair_mismatches:
            if row["sample_id"] in picked:
                continue
            picked.append(row["sample_id"])
            lines.extend(summarize_case(row))
            if len(picked) >= 3:
                break

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare controlled agent outputs with Day5.5 strict repair results.")
    parser.add_argument("--day5-input", type=Path, default=DEFAULT_DAY5_PATH)
    parser.add_argument("--controlled-input", type=Path, default=DEFAULT_CONTROLLED_PATH)
    parser.add_argument("--controlled-metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    day5_rows = load_records_by_sample_id(args.day5_input)
    controlled_rows = load_records_by_sample_id(args.controlled_input)
    controlled_metrics = json.loads(args.controlled_metrics.read_text(encoding="utf-8"))

    if set(day5_rows) != set(controlled_rows):
        missing_in_controlled = sorted(set(day5_rows) - set(controlled_rows))
        missing_in_day5 = sorted(set(controlled_rows) - set(day5_rows))
        raise ValueError(
            f"Sample sets do not match. Missing in controlled: {missing_in_controlled[:10]}; "
            f"missing in day5: {missing_in_day5[:10]}"
        )

    rows = build_diff_rows(day5_rows, controlled_rows)
    summary = summarize_rows(rows)
    summary["repair_success_mismatch_count"] = collect_repair_success_mismatch_count(day5_rows, controlled_rows)

    write_csv(args.csv_output, rows)
    write_markdown(args.doc_output, rows, summary, controlled_metrics)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
