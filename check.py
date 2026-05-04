"""Randomly execute 20 BIRD dev gold SQL queries and save the results."""

import json
import random
from pathlib import Path

from src.config import RESULTS_DIR
from src.db_utils import run_sql
from src.load_bird import load_bird_dev


SAMPLE_SIZE = 20
RANDOM_SEED = 42
OUTPUT_PATH = RESULTS_DIR / "day1_gold_sql_check.json"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = load_bird_dev()
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(all_samples, SAMPLE_SIZE)

    results = []
    success_count = 0

    for sample in sampled:
        execution = run_sql(sample["gold_sql"], sample["db_path"])
        if execution["success"]:
            success_count += 1

        results.append(
            {
                "sample_id": sample["sample_id"],
                "db_id": sample["db_id"],
                "db_path": sample["db_path"],
                "question": sample["question"],
                "difficulty": sample["difficulty"],
                "success": execution["success"],
                "error": execution["error"],
                "row_count": len(execution["rows"]),
                "rows_preview": execution["rows"][:5],
            }
        )

    payload = {
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "success_count": success_count,
        "results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{success_count} / {SAMPLE_SIZE}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
