"""Load and normalize BIRD samples for local evaluation."""

import json

from text2sql.config import BIRD_DEV_HF_DIR, DEV_DATABASE_DIR, DEV_JSON_PATH, get_dev_dataset_source


def get_db_path(db_id):
    return DEV_DATABASE_DIR / db_id / f"{db_id}.sqlite"


def _iter_raw_samples():
    source = get_dev_dataset_source()

    if source == "json":
        with DEV_JSON_PATH.open("r", encoding="utf-8") as file:
            yield from json.load(file)
        return

    if source == "hf_disk":
        from datasets import load_from_disk

        dataset = load_from_disk(str(BIRD_DEV_HF_DIR))
        yield from dataset
        return

    raise FileNotFoundError(
        "BIRD dev data not found. Expected either "
        f"{DEV_JSON_PATH} or a Hugging Face dataset at {BIRD_DEV_HF_DIR}."
    )


def normalize_bird_sample(item, fallback_index):
    db_id = item["db_id"]
    sample_id = item.get("sample_id", item.get("question_id", fallback_index))
    return {
        "sample_id": sample_id,
        "question": item["question"],
        "gold_sql": item["SQL"],
        "db_id": db_id,
        "db_path": str(get_db_path(db_id)),
        "evidence": item.get("evidence") or "",
        "difficulty": item.get("difficulty", "unknown"),
    }


def load_bird_dev(limit=None, db_id=None):
    samples = []
    for fallback_index, item in enumerate(_iter_raw_samples()):
        if db_id is not None and item["db_id"] != db_id:
            continue
        samples.append(normalize_bird_sample(item, fallback_index))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_bird_dev_by_sample_ids(sample_ids, db_id=None):
    wanted_ids = list(sample_ids)
    wanted_set = set(wanted_ids)
    found = {}

    for fallback_index, item in enumerate(_iter_raw_samples()):
        if db_id is not None and item["db_id"] != db_id:
            continue
        sample = normalize_bird_sample(item, fallback_index)
        if sample["sample_id"] in wanted_set:
            found[sample["sample_id"]] = sample

    missing = [sample_id for sample_id in wanted_ids if sample_id not in found]
    if missing:
        raise KeyError(f"Missing sample_ids in BIRD dev split: {missing[:10]}")

    return [found[sample_id] for sample_id in wanted_ids]
