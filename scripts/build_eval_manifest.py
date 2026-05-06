import argparse
import json
from pathlib import Path

from text2sql.data import load_bird_dev, select_manifest_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Build a frozen evaluation manifest from BIRD dev.")
    parser.add_argument("--db-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_bird_dev(limit=None, db_id=args.db_id)
    selected_samples = select_manifest_samples(samples, limit=args.limit, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for sample in selected_samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "total": len(selected_samples),
        "db_id": args.db_id,
        "limit": args.limit,
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
