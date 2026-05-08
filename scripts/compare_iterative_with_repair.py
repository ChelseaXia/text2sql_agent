"""Compare v1.1 iterative-agent predictions with strict repair predictions.

The strict repair file may contain the full dev set. This script filters both
prediction files to the sample ids in the supplied manifest, validates exact
coverage, and writes an aggregate JSON summary plus a per-sample CSV.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path("results/iterative_agent/stratified_300_manifest.jsonl")
DEFAULT_AGENT_PREDICTIONS = Path("results/iterative_agent/v1_1_stratified300_predictions.jsonl")
DEFAULT_REPAIR_PREDICTIONS = Path("results/expanded/day5_5_strict_repair_predictions.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("results/iterative_agent/v1_1_stratified300_vs_repair_summary.json")
DEFAULT_CASES_OUTPUT = Path("results/iterative_agent/v1_1_stratified300_vs_repair_cases.csv")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def pct(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def prediction_sql(row):
    return (
        row.get("final_sql")
        or row.get("final_pred_sql")
        or row.get("predicted_sql")
        or row.get("pred_sql")
        or ""
    )


def is_correct(row):
    return bool(row.get("ex") or row.get("is_correct"))


def is_executable(row):
    return bool(
        row.get("pred_success")
        or row.get("is_executable")
        or row.get("final_success")
        or row.get("initial_success")
    )


def empty_bucket():
    return {
        "sample_count": 0,
        "agent_ex_count": 0,
        "agent_vsr_count": 0,
        "repair_ex_count": 0,
        "repair_vsr_count": 0,
        "both_correct_count": 0,
        "both_wrong_count": 0,
        "agent_only_correct_count": 0,
        "repair_only_correct_count": 0,
    }


def update_bucket(bucket, agent_correct, agent_vsr, repair_correct, repair_vsr):
    bucket["sample_count"] += 1
    bucket["agent_ex_count"] += int(agent_correct)
    bucket["agent_vsr_count"] += int(agent_vsr)
    bucket["repair_ex_count"] += int(repair_correct)
    bucket["repair_vsr_count"] += int(repair_vsr)
    bucket["both_correct_count"] += int(agent_correct and repair_correct)
    bucket["both_wrong_count"] += int((not agent_correct) and (not repair_correct))
    bucket["agent_only_correct_count"] += int(agent_correct and not repair_correct)
    bucket["repair_only_correct_count"] += int(repair_correct and not agent_correct)


def finalize_bucket(bucket):
    total = bucket["sample_count"]
    agent_ex = pct(bucket["agent_ex_count"], total)
    repair_ex = pct(bucket["repair_ex_count"], total)
    agent_vsr = pct(bucket["agent_vsr_count"], total)
    repair_vsr = pct(bucket["repair_vsr_count"], total)
    return {
        **bucket,
        "agent_EX": agent_ex,
        "repair_EX": repair_ex,
        "agent_VSR": agent_vsr,
        "repair_VSR": repair_vsr,
        "agent_minus_repair_EX": agent_ex - repair_ex,
        "agent_minus_repair_VSR": agent_vsr - repair_vsr,
    }


def index_by_sample_id(rows, wanted_ids, label):
    by_id = {}
    duplicates = []
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id not in wanted_ids:
            continue
        if sample_id in by_id:
            duplicates.append(sample_id)
        by_id[sample_id] = row
    if duplicates:
        raise ValueError(f"{label} has duplicate sample_id values: {sorted(set(duplicates))[:20]}")
    return by_id


def validate_coverage(manifest_rows, agent_by_id, repair_by_id):
    wanted_ids = {row["sample_id"] for row in manifest_rows}
    missing_agent = sorted(wanted_ids - set(agent_by_id))
    missing_repair = sorted(wanted_ids - set(repair_by_id))
    if missing_agent or missing_repair:
        message = []
        if missing_agent:
            message.append(f"agent predictions missing {len(missing_agent)} ids, first={missing_agent[:20]}")
        if missing_repair:
            message.append(f"repair predictions missing {len(missing_repair)} ids, first={missing_repair[:20]}")
        raise KeyError("; ".join(message))


def compare(manifest_path, agent_predictions_path, repair_predictions_path, summary_output, cases_output):
    manifest_rows = load_jsonl(manifest_path)
    wanted_ids = {row["sample_id"] for row in manifest_rows}
    agent_by_id = index_by_sample_id(load_jsonl(agent_predictions_path), wanted_ids, "agent predictions")
    repair_by_id = index_by_sample_id(load_jsonl(repair_predictions_path), wanted_ids, "repair predictions")
    validate_coverage(manifest_rows, agent_by_id, repair_by_id)

    overall = empty_bucket()
    by_database = defaultdict(empty_bucket)
    by_difficulty = defaultdict(empty_bucket)
    by_database_difficulty = defaultdict(empty_bucket)
    cases = []
    agent_only_ids = []
    repair_only_ids = []
    both_wrong_ids = []

    for sample in manifest_rows:
        sample_id = sample["sample_id"]
        agent = agent_by_id[sample_id]
        repair = repair_by_id[sample_id]

        agent_correct = is_correct(agent)
        repair_correct = is_correct(repair)
        agent_vsr = is_executable(agent)
        repair_vsr = is_executable(repair)

        for bucket in (
            overall,
            by_database[sample["db_id"]],
            by_difficulty[sample.get("difficulty", "unknown")],
            by_database_difficulty[f"{sample['db_id']}::{sample.get('difficulty', 'unknown')}"],
        ):
            update_bucket(bucket, agent_correct, agent_vsr, repair_correct, repair_vsr)

        if agent_correct and not repair_correct:
            outcome = "agent_only_correct"
            agent_only_ids.append(sample_id)
        elif repair_correct and not agent_correct:
            outcome = "repair_only_correct"
            repair_only_ids.append(sample_id)
        elif agent_correct and repair_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
            both_wrong_ids.append(sample_id)

        cases.append(
            {
                "sample_id": sample_id,
                "db_id": sample["db_id"],
                "difficulty": sample.get("difficulty", "unknown"),
                "outcome": outcome,
                "agent_correct": agent_correct,
                "repair_correct": repair_correct,
                "agent_vsr": agent_vsr,
                "repair_vsr": repair_vsr,
                "agent_tool_call_count": agent.get("tool_call_count", ""),
                "agent_execute_call_count": agent.get("execute_call_count", ""),
                "agent_suspicious_trigger_count": agent.get("suspicious_trigger_count", ""),
                "agent_repair_attempt_count": agent.get("repair_attempt_count", ""),
                "strict_repair_rounds": repair.get("repair_rounds", ""),
                "question": sample["question"],
                "agent_final_sql": prediction_sql(agent),
                "repair_final_sql": prediction_sql(repair),
                "gold_sql": sample.get("gold_sql", ""),
            }
        )

    summary = {
        "manifest_path": str(manifest_path),
        "agent_predictions_path": str(agent_predictions_path),
        "repair_predictions_path": str(repair_predictions_path),
        "sample_count": overall["sample_count"],
        "overall": finalize_bucket(overall),
        "by_database": {key: finalize_bucket(value) for key, value in sorted(by_database.items())},
        "by_difficulty": {key: finalize_bucket(value) for key, value in sorted(by_difficulty.items())},
        "by_database_difficulty": {
            key: finalize_bucket(value) for key, value in sorted(by_database_difficulty.items())
        },
        "agent_only_correct_sample_ids": agent_only_ids,
        "repair_only_correct_sample_ids": repair_only_ids,
        "both_wrong_sample_ids": both_wrong_ids,
    }

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cases_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "db_id",
        "difficulty",
        "outcome",
        "agent_correct",
        "repair_correct",
        "agent_vsr",
        "repair_vsr",
        "agent_tool_call_count",
        "agent_execute_call_count",
        "agent_suspicious_trigger_count",
        "agent_repair_attempt_count",
        "strict_repair_rounds",
        "question",
        "agent_final_sql",
        "repair_final_sql",
        "gold_sql",
    ]
    with cases_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare iterative agent v1.1 with strict repair.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--agent-predictions", type=Path, default=DEFAULT_AGENT_PREDICTIONS)
    parser.add_argument("--repair-predictions", type=Path, default=DEFAULT_REPAIR_PREDICTIONS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASES_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = compare(
        manifest_path=args.manifest,
        agent_predictions_path=args.agent_predictions,
        repair_predictions_path=args.repair_predictions,
        summary_output=args.summary_output,
        cases_output=args.cases_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
