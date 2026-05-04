"""Evaluate naive baseline predictions with execution accuracy and validity."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from config import RESULTS_DIR
except ModuleNotFoundError:
    from src.config import RESULTS_DIR


DEFAULT_INPUT_PATH = RESULTS_DIR / "day2_naive_predictions.jsonl"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "day2_naive_metrics.json"


def load_jsonl(path: Path) -> List[Dict]:
    """Load prediction records from a jsonl file."""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(records: Iterable[Dict]) -> Dict:
    """Compute EX and VSR from saved prediction records."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate naive baseline predictions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    metrics = compute_metrics(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
