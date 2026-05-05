"""Metrics helpers for saved prediction files."""

import json


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(records):
    rows = list(records)
    total = len(rows)
    pred_success_count = sum(1 for row in rows if row.get("pred_success"))
    ex_count = sum(1 for row in rows if row.get("ex"))
    gold_success_count = sum(1 for row in rows if row.get("gold_success"))
    difficulty_breakdown = {}

    for difficulty in ("simple", "moderate", "challenging"):
        subset = [row for row in rows if row.get("difficulty") == difficulty]
        sample_count = len(subset)
        subset_vsr_count = sum(1 for row in subset if row.get("pred_success"))
        subset_ex_count = sum(1 for row in subset if row.get("ex"))
        difficulty_breakdown[difficulty] = {
            "sample_count": sample_count,
            "vsr_count": subset_vsr_count,
            "ex_count": subset_ex_count,
            "VSR": subset_vsr_count / sample_count if sample_count else 0.0,
            "EX": subset_ex_count / sample_count if sample_count else 0.0,
        }

    return {
        "total": total,
        "ex_count": ex_count,
        "vsr_count": pred_success_count,
        "gold_success_count": gold_success_count,
        "EX": ex_count / total if total else 0.0,
        "VSR": pred_success_count / total if total else 0.0,
        "difficulty_breakdown": difficulty_breakdown,
    }


def save_metrics(records, output_path):
    metrics = compute_metrics(records)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
