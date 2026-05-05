"""Read old and new result files and print a compact summary."""

import argparse
import json
from pathlib import Path

from text2sql.config import RESULTS_DIR

KNOWN_METRICS = [
    ("naive_full_schema", RESULTS_DIR / "day2_naive_metrics.json"),
    ("schema_linked_promptfix", RESULTS_DIR / "day3_schema_table_linked_promptfix_metrics.json"),
    ("fewshot_retrieval", RESULTS_DIR / "day4_fewshot_metrics.json"),
    ("strict_execution_repair", RESULTS_DIR / "day5_repair_from_day3_metrics.json"),
    ("schema_format_ddl_ablation", RESULTS_DIR / "day6_schema_ddl_promptfix_metrics.json"),
    ("react_agent", RESULTS_DIR / "day7_agent_metrics.json"),
]


def load_metrics(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Summarize available result metrics.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    rows = []
    for method, default_path in KNOWN_METRICS:
        path = args.results_dir / default_path.name
        metrics = load_metrics(path)
        if metrics is None:
            continue
        rows.append((method, metrics.get("total"), metrics.get("EX"), metrics.get("VSR"), path.name))

    if not rows:
        print("No metric files found.")
        return

    print(f"{'method':<28} {'total':>5} {'EX':>6} {'VSR':>6} file")
    for method, total, ex, vsr, filename in rows:
        print(f"{method:<28} {total:>5} {ex:>6.2f} {vsr:>6.2f} {filename}")


if __name__ == "__main__":
    main()
