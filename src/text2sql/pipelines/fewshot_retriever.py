"""Embedding-based few-shot retrieval for Text-to-SQL examples."""

import json
from dataclasses import dataclass

import numpy as np

from text2sql.config import DATA_DIR
from text2sql.data import load_bird_dev

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ExampleRecord:
    sample_id: int
    db_id: str
    question: str
    gold_sql: str
    difficulty: str


def _normalize_train_item(item, fallback_index):
    sample_id = item.get("sample_id", item.get("question_id", fallback_index))
    return {
        "sample_id": sample_id,
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["SQL"],
        "difficulty": item.get("difficulty", "unknown"),
    }


def _load_json_split(path):
    with path.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)
    return [_normalize_train_item(item, index) for index, item in enumerate(raw_items)]


def _load_hf_split(path):
    from datasets import Dataset, DatasetDict, load_from_disk

    loaded = load_from_disk(str(path))
    if isinstance(loaded, Dataset):
        dataset = loaded
    elif isinstance(loaded, DatasetDict):
        train_like_keys = [key for key in loaded.keys() if "train" in key.lower()]
        if not train_like_keys:
            return []
        dataset = loaded[train_like_keys[0]]
    else:
        return []
    return [_normalize_train_item(item, index) for index, item in enumerate(dataset)]


def load_train_pool_if_available():
    for path in [
        DATA_DIR / "bird_train" / "train.json",
        DATA_DIR / "bird_train.json",
        DATA_DIR / "train.json",
    ]:
        if path.exists():
            return _load_json_split(path)
    for path in [DATA_DIR / "bird_train", DATA_DIR / "bird_train_hf", DATA_DIR / "train"]:
        if path.exists():
            loaded = _load_hf_split(path)
            if loaded:
                return loaded
    return []


def load_example_pool():
    train_pool = load_train_pool_if_available()
    if train_pool:
        return [ExampleRecord(**item) for item in train_pool], "train"

    dev_pool = load_bird_dev()
    examples = [
        ExampleRecord(
            sample_id=item["sample_id"],
            db_id=item["db_id"],
            question=item["question"],
            gold_sql=item["gold_sql"],
            difficulty=item["difficulty"],
        )
        for item in dev_pool
    ]
    return examples, "dev_leave_one_out"


class FewShotRetriever:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL, examples=None, example_source=None):
        from sentence_transformers import SentenceTransformer

        if examples is None or example_source is None:
            loaded_examples, loaded_source = load_example_pool()
            examples = loaded_examples if examples is None else list(examples)
            example_source = loaded_source if example_source is None else example_source

        self.model_name = model_name
        self.examples = list(examples)
        self.example_source = example_source
        self.model = SentenceTransformer(model_name)
        self.examples_by_db = {}
        self.embeddings_by_db = {}
        self._build_indexes()

    def _build_indexes(self):
        for example in self.examples:
            self.examples_by_db.setdefault(example.db_id, []).append(example)
        for db_id, examples in self.examples_by_db.items():
            questions = [example.question for example in examples]
            self.embeddings_by_db[db_id] = np.asarray(
                self.model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
            )

    def get_top_k_examples(self, current_sample_id, current_db_id, current_question, k=3):
        examples = self.examples_by_db.get(current_db_id, [])
        if not examples:
            return []
        candidate_indexes = [index for index, example in enumerate(examples) if example.sample_id != current_sample_id]
        if not candidate_indexes:
            return []

        query_embedding = np.asarray(
            self.model.encode([current_question], normalize_embeddings=True, show_progress_bar=False)
        )[0]
        db_embeddings = self.embeddings_by_db[current_db_id]
        scored = []
        for index in candidate_indexes:
            scored.append((float(np.dot(db_embeddings[index], query_embedding)), examples[index]))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {
                "sample_id": example.sample_id,
                "db_id": example.db_id,
                "question": example.question,
                "gold_sql": example.gold_sql,
                "difficulty": example.difficulty,
                "similarity": similarity,
                "example_source": self.example_source,
            }
            for similarity, example in scored[:k]
        ]
