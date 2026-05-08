"""Compare self-consistency voting with compact strict-repair baseline."""

import argparse
import csv
import json
from pathlib import Path

from text2sql.eval import load_jsonl


DEFAULT_SELF_CONSISTENCY_PATH = Path("results/self_consistency/california_schools_50_predictions.jsonl")
DEFAULT_BASELINE_PATH = Path("results/controlled_vs_day5_diff.csv")
DEFAULT_OUTPUT_CSV = Path("results/self_consistency/california_schools_50_pairwise_vs_repair.csv")
DEFAULT_OUTPUT_JSON = Path("results/self_consistency/california_schools_50_pairwise_summary.json")

STAT_KEYS = [
    "both_correct_count",
    "both_wrong_count",
    "self_consistency_only_correct_count",
    "repair_only_correct_count",
    "oracle_true_selected_false_count",
    "oracle_false_count",
]


def empty_stats():
    stats = {key: 0 for key in STAT_KEYS}
    stats["sample_count"] = 0
    return stats


def load_baseline_rows(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {str(row["sample_id"]): row for row in rows}


def case_tags(baseline_correct, selected_correct, oracle_correct, difficulty):
    tags = []
    if (not baseline_correct) and selected_correct:
        tags.append("baseline_wrong_self_consistency_correct")
    if baseline_correct and (not selected_correct):
        tags.append("baseline_correct_self_consistency_wrong")
    if oracle_correct and (not selected_correct):
        tags.append("oracle_true_selected_false")
    if difficulty == "challenging" and (not oracle_correct):
        tags.append("challenging_generation_bottleneck")
    return tags


def update_stats(stats, baseline_correct, selected_correct, oracle_correct):
    stats["sample_count"] += 1
    if baseline_correct and selected_correct:
        stats["both_correct_count"] += 1
    elif (not baseline_correct) and (not selected_correct):
        stats["both_wrong_count"] += 1
    elif (not baseline_correct) and selected_correct:
        stats["self_consistency_only_correct_count"] += 1
    elif baseline_correct and (not selected_correct):
        stats["repair_only_correct_count"] += 1

    if oracle_correct and (not selected_correct):
        stats["oracle_true_selected_false_count"] += 1
    if not oracle_correct:
        stats["oracle_false_count"] += 1


def build_pairwise_rows(self_consistency_rows, baseline_by_sample_id):
    pairwise_rows = []
    overall = empty_stats()
    by_difficulty = {}

    baseline_wrong_self_correct = []
    baseline_correct_self_wrong = []
    oracle_true_selected_false = []
    challenging_generation_bottleneck = []

    missing_baseline_ids = []

    for row in self_consistency_rows:
        sample_id = str(row["sample_id"])
        baseline_row = baseline_by_sample_id.get(sample_id)
        if baseline_row is None:
            missing_baseline_ids.append(sample_id)
            continue

        difficulty = row["difficulty"]
        baseline_correct = str(baseline_row.get("day5_ex", "")).strip().lower() == "true"
        selected_correct = bool(row.get("selected_correct"))
        oracle_correct = bool(row.get("oracle_correct"))
        selected_executable = bool(row.get("selected_executable"))

        tags = case_tags(
            baseline_correct=baseline_correct,
            selected_correct=selected_correct,
            oracle_correct=oracle_correct,
            difficulty=difficulty,
        )
        pairwise_row = {
            "sample_id": sample_id,
            "difficulty": difficulty,
            "question": row.get("question", ""),
            "repair_correct": baseline_correct,
            "self_consistency_selected_correct": selected_correct,
            "self_consistency_selected_executable": selected_executable,
            "oracle_correct": oracle_correct,
            "selected_candidate_id": row.get("selected_candidate_id"),
            "selected_cluster_id": row.get("selected_cluster_id"),
            "selected_cluster_size": row.get("selected_cluster_size"),
            "valid_candidate_count": row.get("valid_candidate_count"),
            "cluster_count": row.get("cluster_count"),
            "cluster_confidence": row.get("cluster_confidence"),
            "selected_sql": row.get("selected_sql", ""),
            "case_tags": "|".join(tags),
        }
        pairwise_rows.append(pairwise_row)

        update_stats(overall, baseline_correct, selected_correct, oracle_correct)
        difficulty_stats = by_difficulty.setdefault(difficulty, empty_stats())
        update_stats(difficulty_stats, baseline_correct, selected_correct, oracle_correct)

        compact_case = {
            "sample_id": sample_id,
            "difficulty": difficulty,
            "question": row.get("question", ""),
            "repair_correct": baseline_correct,
            "selected_correct": selected_correct,
            "oracle_correct": oracle_correct,
            "selected_sql": row.get("selected_sql", ""),
            "cluster_confidence": row.get("cluster_confidence"),
            "valid_candidate_count": row.get("valid_candidate_count"),
            "cluster_count": row.get("cluster_count"),
        }
        if "baseline_wrong_self_consistency_correct" in tags:
            baseline_wrong_self_correct.append(compact_case)
        if "baseline_correct_self_consistency_wrong" in tags:
            baseline_correct_self_wrong.append(compact_case)
        if "oracle_true_selected_false" in tags:
            oracle_true_selected_false.append(compact_case)
        if "challenging_generation_bottleneck" in tags:
            challenging_generation_bottleneck.append(compact_case)

    if missing_baseline_ids:
        raise KeyError(f"Missing baseline rows for sample_ids: {missing_baseline_ids[:10]}")

    summary = dict(overall)
    summary["by_difficulty"] = {name: by_difficulty[name] for name in sorted(by_difficulty)}
    summary["baseline_wrong_self_consistency_correct_cases"] = baseline_wrong_self_correct
    summary["baseline_correct_self_consistency_wrong_cases"] = baseline_correct_self_wrong
    summary["oracle_true_selected_false_cases"] = oracle_true_selected_false
    summary["challenging_generation_bottleneck_cases"] = challenging_generation_bottleneck
    return pairwise_rows, summary


def write_pairwise_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "difficulty",
        "question",
        "repair_correct",
        "self_consistency_selected_correct",
        "self_consistency_selected_executable",
        "oracle_correct",
        "selected_candidate_id",
        "selected_cluster_id",
        "selected_cluster_size",
        "valid_candidate_count",
        "cluster_count",
        "cluster_confidence",
        "selected_sql",
        "case_tags",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Compare self-consistency voting with strict-repair baseline.")
    parser.add_argument("--self-consistency-input", type=Path, default=DEFAULT_SELF_CONSISTENCY_PATH)
    parser.add_argument("--baseline-input", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main():
    args = parse_args()
    self_consistency_rows = load_jsonl(args.self_consistency_input)
    baseline_by_sample_id = load_baseline_rows(args.baseline_input)
    pairwise_rows, summary = build_pairwise_rows(self_consistency_rows, baseline_by_sample_id)
    write_pairwise_csv(args.csv_output, pairwise_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pairwise_row_count": len(pairwise_rows),
                "csv_output": str(args.csv_output),
                "summary_output": str(args.summary_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
