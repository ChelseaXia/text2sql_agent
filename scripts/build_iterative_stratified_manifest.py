"""Build a stratified multi-db manifest for iterative-agent evaluation."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DIFFICULTIES = ("simple", "moderate", "challenging")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocate_database_quotas(rows, target_total, max_per_db):
    by_db = defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)

    total = len(rows)
    quotas = {}
    fractional = []
    for db_id, db_rows in by_db.items():
        cell_count = len({row.get("difficulty", "unknown") for row in db_rows})
        min_quota = min(len(db_rows), cell_count * 3)
        raw = target_total * (len(db_rows) / total)
        quota = min(max_per_db, len(db_rows), max(min_quota, int(raw)))
        quotas[db_id] = quota
        fractional.append((raw - int(raw), db_id))

    while sum(quotas.values()) < target_total:
        changed = False
        for _, db_id in sorted(fractional, reverse=True):
            if quotas[db_id] < min(max_per_db, len(by_db[db_id])):
                quotas[db_id] += 1
                changed = True
                if sum(quotas.values()) >= target_total:
                    break
        if not changed:
            break

    while sum(quotas.values()) > target_total:
        for _, db_id in sorted(fractional):
            cell_count = len({row.get("difficulty", "unknown") for row in by_db[db_id]})
            min_quota = min(len(by_db[db_id]), cell_count * 3)
            if quotas[db_id] > min_quota:
                quotas[db_id] -= 1
                if sum(quotas.values()) <= target_total:
                    break

    return quotas


def allocate_difficulty_quotas(db_rows, db_quota):
    by_diff = defaultdict(list)
    for row in db_rows:
        by_diff[row.get("difficulty", "unknown")].append(row)

    quotas = {}
    remaining = db_quota
    for difficulty, diff_rows in by_diff.items():
        quota = min(len(diff_rows), 3)
        quotas[difficulty] = quota
        remaining -= quota

    total = len(db_rows)
    fractional = []
    for difficulty, diff_rows in by_diff.items():
        raw_extra = max(0.0, db_quota * (len(diff_rows) / total) - quotas[difficulty])
        fractional.append((raw_extra - int(raw_extra), difficulty))
        add = min(len(diff_rows) - quotas[difficulty], int(raw_extra), remaining)
        quotas[difficulty] += add
        remaining -= add

    while remaining > 0:
        changed = False
        for _, difficulty in sorted(fractional, reverse=True):
            if quotas[difficulty] < len(by_diff[difficulty]):
                quotas[difficulty] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break

    return quotas


def build_summary(rows):
    by_database = Counter(row["db_id"] for row in rows)
    by_difficulty = Counter(row.get("difficulty", "unknown") for row in rows)
    by_database_difficulty = defaultdict(Counter)
    for row in rows:
        by_database_difficulty[row["db_id"]][row.get("difficulty", "unknown")] += 1

    return {
        "total_sample_count": len(rows),
        "by_database_count": dict(sorted(by_database.items())),
        "by_difficulty_count": dict(sorted(by_difficulty.items())),
        "by_database_difficulty_count": {
            db_id: {difficulty: counts.get(difficulty, 0) for difficulty in DIFFICULTIES}
            for db_id, counts in sorted(by_database_difficulty.items())
        },
        "first_10_sample_id": [row["sample_id"] for row in rows[:10]],
    }


def build_manifest(input_path, output_path, summary_path, target_total, max_per_db, seed):
    rows = load_jsonl(input_path)
    rng = random.Random(seed)

    by_db = defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)

    db_quotas = allocate_database_quotas(rows, target_total=target_total, max_per_db=max_per_db)
    selected = []
    for db_id in sorted(by_db):
        db_rows = by_db[db_id]
        diff_quotas = allocate_difficulty_quotas(db_rows, db_quotas[db_id])
        by_diff = defaultdict(list)
        for row in db_rows:
            by_diff[row.get("difficulty", "unknown")].append(row)
        for difficulty in sorted(by_diff):
            candidates = list(by_diff[difficulty])
            rng.shuffle(candidates)
            selected.extend(candidates[: diff_quotas[difficulty]])

    selected.sort(key=lambda row: (row["db_id"], DIFFICULTIES.index(row.get("difficulty", "unknown")) if row.get("difficulty", "unknown") in DIFFICULTIES else 99, row["sample_id"]))
    write_jsonl(selected, output_path)
    summary = build_summary(selected)
    summary["seed"] = seed
    summary["target_total"] = target_total
    summary["max_per_db"] = max_per_db
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build iterative-agent stratified manifest.")
    parser.add_argument("--input", type=Path, default=Path("results/expanded/eval_manifest_full_dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/iterative_agent/stratified_300_manifest.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/iterative_agent/stratified_300_manifest_summary.json"))
    parser.add_argument("--target-total", type=int, default=300)
    parser.add_argument("--max-per-db", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_manifest(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        target_total=args.target_total,
        max_per_db=args.max_per_db,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
