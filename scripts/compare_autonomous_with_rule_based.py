"""Compare autonomous tool selection against rule-based iterative-agent predictions."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path("results/iterative_agent/stratified_300_manifest.jsonl")
DEFAULT_AUTONOMOUS_PREDICTIONS = Path("results/iterative_agent/autonomous_stratified300_predictions.jsonl")
DEFAULT_RULE_BASED_PREDICTIONS = Path("results/iterative_agent/v1_1_stratified300_predictions.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("results/iterative_agent/autonomous_vs_rule_based_summary.json")
DEFAULT_CASES_OUTPUT = Path("results/iterative_agent/autonomous_vs_rule_based_cases.csv")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def pct(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def prediction_sql(row):
    return row.get("final_sql") or row.get("predicted_sql") or row.get("pred_sql") or ""


def is_correct(row):
    return bool(row.get("ex") or row.get("is_correct"))


def is_executable(row):
    return bool(row.get("pred_success") or row.get("is_executable"))


def finished(row):
    return row.get("final_sql_source") == "finish_tool"


def empty_bucket():
    return {
        "sample_count": 0,
        "autonomous_ex_count": 0,
        "autonomous_vsr_count": 0,
        "autonomous_finish_count": 0,
        "rule_based_ex_count": 0,
        "rule_based_vsr_count": 0,
        "rule_based_finish_count": 0,
        "autonomous_tool_call_count": 0,
        "rule_based_tool_call_count": 0,
        "autonomous_execute_call_count": 0,
        "rule_based_execute_call_count": 0,
        "autonomous_only_correct_count": 0,
        "rule_based_only_correct_count": 0,
        "both_correct_count": 0,
        "both_wrong_count": 0,
    }


def update_bucket(bucket, autonomous, rule_based):
    autonomous_correct = is_correct(autonomous)
    rule_based_correct = is_correct(rule_based)
    bucket["sample_count"] += 1
    bucket["autonomous_ex_count"] += int(autonomous_correct)
    bucket["autonomous_vsr_count"] += int(is_executable(autonomous))
    bucket["autonomous_finish_count"] += int(finished(autonomous))
    bucket["rule_based_ex_count"] += int(rule_based_correct)
    bucket["rule_based_vsr_count"] += int(is_executable(rule_based))
    bucket["rule_based_finish_count"] += int(finished(rule_based))
    bucket["autonomous_tool_call_count"] += autonomous.get("tool_call_count", 0)
    bucket["rule_based_tool_call_count"] += rule_based.get("tool_call_count", 0)
    bucket["autonomous_execute_call_count"] += autonomous.get("execute_call_count", 0)
    bucket["rule_based_execute_call_count"] += rule_based.get("execute_call_count", 0)
    bucket["autonomous_only_correct_count"] += int(autonomous_correct and not rule_based_correct)
    bucket["rule_based_only_correct_count"] += int(rule_based_correct and not autonomous_correct)
    bucket["both_correct_count"] += int(autonomous_correct and rule_based_correct)
    bucket["both_wrong_count"] += int((not autonomous_correct) and (not rule_based_correct))


def finalize_bucket(bucket):
    total = bucket["sample_count"]
    return {
        **bucket,
        "autonomous_EX": pct(bucket["autonomous_ex_count"], total),
        "autonomous_VSR": pct(bucket["autonomous_vsr_count"], total),
        "autonomous_finish_rate": pct(bucket["autonomous_finish_count"], total),
        "rule_based_EX": pct(bucket["rule_based_ex_count"], total),
        "rule_based_VSR": pct(bucket["rule_based_vsr_count"], total),
        "rule_based_finish_rate": pct(bucket["rule_based_finish_count"], total),
        "autonomous_avg_tool_calls": pct(bucket["autonomous_tool_call_count"], total),
        "rule_based_avg_tool_calls": pct(bucket["rule_based_tool_call_count"], total),
        "autonomous_avg_execute_calls": pct(bucket["autonomous_execute_call_count"], total),
        "rule_based_avg_execute_calls": pct(bucket["rule_based_execute_call_count"], total),
    }


def index_by_sample_id(rows, wanted_ids, label):
    indexed = {}
    duplicates = []
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id not in wanted_ids:
            continue
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = row
    if duplicates:
        raise ValueError(f"{label} has duplicate sample_id values: {sorted(set(duplicates))[:20]}")
    return indexed


def validate_coverage(manifest_rows, autonomous_by_id, rule_based_by_id):
    wanted_ids = {row["sample_id"] for row in manifest_rows}
    missing_autonomous = sorted(wanted_ids - set(autonomous_by_id))
    missing_rule_based = sorted(wanted_ids - set(rule_based_by_id))
    if missing_autonomous or missing_rule_based:
        errors = []
        if missing_autonomous:
            errors.append(f"autonomous missing {len(missing_autonomous)} ids, first={missing_autonomous[:20]}")
        if missing_rule_based:
            errors.append(f"rule_based missing {len(missing_rule_based)} ids, first={missing_rule_based[:20]}")
        raise KeyError("; ".join(errors))


def autonomous_failure_modes(row):
    modes = Counter()
    modes["over_exploration"] += row.get("over_exploration_count", 0)
    modes["premature_finish"] += row.get("premature_finish_count", 0)
    modes["tool_selection_error"] += row.get("tool_selection_error_count", 0)
    modes["argument_error"] += row.get("argument_error_count", 0)
    modes["json_parse_error"] += row.get("json_parse_error_count", 0)
    modes["budget_exceeded"] += row.get("budget_exceeded_count", 0)
    return modes


def compare(manifest_path, autonomous_predictions_path, rule_based_predictions_path, summary_output, cases_output):
    manifest_rows = load_jsonl(manifest_path)
    wanted_ids = {row["sample_id"] for row in manifest_rows}
    autonomous_by_id = index_by_sample_id(load_jsonl(autonomous_predictions_path), wanted_ids, "autonomous")
    rule_based_by_id = index_by_sample_id(load_jsonl(rule_based_predictions_path), wanted_ids, "rule_based")
    validate_coverage(manifest_rows, autonomous_by_id, rule_based_by_id)

    overall = empty_bucket()
    by_database = defaultdict(empty_bucket)
    by_difficulty = defaultdict(empty_bucket)
    failure_modes = Counter()
    cases = []

    for sample in manifest_rows:
        sample_id = sample["sample_id"]
        autonomous = autonomous_by_id[sample_id]
        rule_based = rule_based_by_id[sample_id]
        for bucket in (overall, by_database[sample["db_id"]], by_difficulty[sample.get("difficulty", "unknown")]):
            update_bucket(bucket, autonomous, rule_based)
        failure_modes.update(autonomous_failure_modes(autonomous))

        autonomous_correct = is_correct(autonomous)
        rule_based_correct = is_correct(rule_based)
        if autonomous_correct and not rule_based_correct:
            outcome = "autonomous_only_correct"
        elif rule_based_correct and not autonomous_correct:
            outcome = "rule_based_only_correct"
        elif autonomous_correct and rule_based_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"

        cases.append(
            {
                "sample_id": sample_id,
                "db_id": sample["db_id"],
                "difficulty": sample.get("difficulty", "unknown"),
                "outcome": outcome,
                "autonomous_correct": autonomous_correct,
                "rule_based_correct": rule_based_correct,
                "autonomous_vsr": is_executable(autonomous),
                "rule_based_vsr": is_executable(rule_based),
                "autonomous_finish_reason": autonomous.get("finish_reason", ""),
                "autonomous_tool_calls": autonomous.get("tool_call_count", 0),
                "rule_based_tool_calls": rule_based.get("tool_call_count", 0),
                "autonomous_execute_calls": autonomous.get("execute_call_count", 0),
                "rule_based_execute_calls": rule_based.get("execute_call_count", 0),
                "question": sample["question"],
                "autonomous_final_sql": prediction_sql(autonomous),
                "rule_based_final_sql": prediction_sql(rule_based),
                "gold_sql": sample.get("gold_sql", ""),
            }
        )

    summary = {
        "manifest_path": str(manifest_path),
        "autonomous_predictions_path": str(autonomous_predictions_path),
        "rule_based_predictions_path": str(rule_based_predictions_path),
        "sample_count": overall["sample_count"],
        "overall": finalize_bucket(overall),
        "by_database": {key: finalize_bucket(value) for key, value in sorted(by_database.items())},
        "by_difficulty": {key: finalize_bucket(value) for key, value in sorted(by_difficulty.items())},
        "autonomous_failure_modes": dict(failure_modes),
    }

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cases_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(cases[0].keys()) if cases else []
    with cases_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare autonomous and rule-based iterative Text2SQL runs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--autonomous-predictions", type=Path, default=DEFAULT_AUTONOMOUS_PREDICTIONS)
    parser.add_argument("--rule-based-predictions", type=Path, default=DEFAULT_RULE_BASED_PREDICTIONS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASES_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = compare(
        manifest_path=args.manifest,
        autonomous_predictions_path=args.autonomous_predictions,
        rule_based_predictions_path=args.rule_based_predictions,
        summary_output=args.summary_output,
        cases_output=args.cases_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
