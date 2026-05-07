"""Analyze saved self-consistency voting predictions."""

import argparse
import json
from pathlib import Path

from text2sql.eval import load_jsonl
from text2sql.pipelines.self_consistency import compute_self_consistency_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze saved self-consistency voting predictions.")
    parser.add_argument("--predictions-input", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.predictions_input)
    metrics = compute_self_consistency_metrics(records)
    if args.metrics_output is not None:
        args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
