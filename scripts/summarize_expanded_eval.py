import json
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.eval import compute_metrics, load_jsonl


EXPANDED_DIR = RESULTS_DIR / "expanded"
SUMMARY_PATH = EXPANDED_DIR / "expanded_core_eval_summary.json"
METHODS = {
    "day2": {
        "label": "Day2 naive full schema",
        "predictions": EXPANDED_DIR / "day2_naive_predictions.jsonl",
        "metrics": EXPANDED_DIR / "day2_naive_metrics.json",
    },
    "day3": {
        "label": "Day3 schema-linked + promptfix",
        "predictions": EXPANDED_DIR / "day3_schema_linked_predictions.jsonl",
        "metrics": EXPANDED_DIR / "day3_schema_linked_metrics.json",
    },
    "day5_5": {
        "label": "Day5.5 strict execution repair",
        "predictions": EXPANDED_DIR / "day5_5_strict_repair_predictions.jsonl",
        "metrics": EXPANDED_DIR / "day5_5_strict_repair_metrics.json",
    },
    "day6": {
        "label": "Day6 DDL/schema serialization ablation",
        "predictions": EXPANDED_DIR / "day6_ddl_predictions.jsonl",
        "metrics": EXPANDED_DIR / "day6_ddl_metrics.json",
    },
}


def load_saved_metrics(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def group_metrics(records, key):
    grouped = {}
    for record in records:
        value = record.get(key)
        if value in (None, ""):
            continue
        grouped.setdefault(str(value), []).append(record)
    return {
        value: compute_metrics(rows)
        for value, rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def method_summary(predictions_path, metrics_path):
    records = load_jsonl(predictions_path) if predictions_path.exists() else []
    metrics = compute_metrics(records) if records else (load_saved_metrics(metrics_path) or {})
    summary = {
        "sample_count": len(records) if records else metrics.get("total", 0),
        "EX": metrics.get("EX", 0.0),
        "VSR": metrics.get("VSR", 0.0),
    }
    if records:
        summary["by_database"] = group_metrics(records, "db_id")
        summary["by_difficulty"] = group_metrics(records, "difficulty")
    else:
        summary["by_database"] = {}
        summary["by_difficulty"] = metrics.get("difficulty_breakdown", {})
    return summary


def gain_summary(base, target):
    base_ex = base.get("EX", 0.0)
    base_vsr = base.get("VSR", 0.0)
    target_ex = target.get("EX", 0.0)
    target_vsr = target.get("VSR", 0.0)
    return {
        "EX_absolute": target_ex - base_ex,
        "EX_relative": ((target_ex - base_ex) / base_ex) if base_ex else None,
        "VSR_absolute": target_vsr - base_vsr,
        "VSR_relative": ((target_vsr - base_vsr) / base_vsr) if base_vsr else None,
    }


def main():
    methods = {name: method_summary(paths["predictions"], paths["metrics"]) for name, paths in METHODS.items()}
    total_sample_count = max((summary["sample_count"] for summary in methods.values()), default=0)
    summary = {
        "total_sample_count": total_sample_count,
        "methods": methods,
        "gains": {
            "day3_vs_day2": gain_summary(methods["day2"], methods["day3"]),
            "day5_5_vs_day3": gain_summary(methods["day3"], methods["day5_5"]),
            "day6_vs_day3": gain_summary(methods["day3"], methods["day6"]),
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
