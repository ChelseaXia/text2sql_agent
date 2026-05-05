"""Check whether Day5.5 and ReAct Agent use the same sample_id set."""

import json
from collections import Counter

from text2sql.config import RESULTS_DIR
from text2sql.eval import load_jsonl

DAY5_METRICS_PATH = RESULTS_DIR / "day5_repair_from_day3_metrics.json"
DAY5_PREDICTIONS_PATH = RESULTS_DIR / "day5_repair_from_day3_predictions.jsonl"
REACT_AGENT_TRACES_PATH = RESULTS_DIR / "react_agent_traces.jsonl"
OUTPUT_PATH = RESULTS_DIR / "sample_alignment_report.json"


def difficulty_distribution(records):
    counts = Counter(record.get("difficulty", "unknown") for record in records)
    return dict(sorted(counts.items()))


def main():
    day5_records = load_jsonl(DAY5_PREDICTIONS_PATH)
    react_records = load_jsonl(REACT_AGENT_TRACES_PATH)

    day5_ids = [record["sample_id"] for record in day5_records]
    react_ids = [record["sample_id"] for record in react_records]
    day5_set = set(day5_ids)
    react_set = set(react_ids)

    report = {
        "day5_5_metrics_path": str(DAY5_METRICS_PATH),
        "day5_5_predictions_path": str(DAY5_PREDICTIONS_PATH),
        "react_agent_traces_path": str(REACT_AGENT_TRACES_PATH),
        "day5_5_sample_count": len(day5_ids),
        "react_agent_sample_count": len(react_ids),
        "intersection_count": len(day5_set & react_set),
        "only_in_day5_5": sorted(day5_set - react_set),
        "only_in_react_agent": sorted(react_set - day5_set),
        "day5_5_difficulty_distribution": difficulty_distribution(day5_records),
        "react_agent_difficulty_distribution": difficulty_distribution(react_records),
        "is_apples_to_apples": day5_ids == react_ids,
    }

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"day5_5_sample_count: {report['day5_5_sample_count']}")
    print(f"react_agent_sample_count: {report['react_agent_sample_count']}")
    print(f"intersection_count: {report['intersection_count']}")
    print(f"only_in_day5_5: {report['only_in_day5_5']}")
    print(f"only_in_react_agent: {report['only_in_react_agent']}")
    print(f"day5_5_difficulty_distribution: {report['day5_5_difficulty_distribution']}")
    print(f"react_agent_difficulty_distribution: {report['react_agent_difficulty_distribution']}")

    if not report["is_apples_to_apples"]:
        print("ReAct Agent result is not apples-to-apples comparable with Day5.5.")


if __name__ == "__main__":
    main()
