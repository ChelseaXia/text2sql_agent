"""Hybrid BM25 + dense schema linking for column retrieval."""

import math
import re
from collections import Counter

import numpy as np

from text2sql.schema.items import build_schema_items, format_table_linked_schema_text

DEFAULT_TOP_K = 30
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def identifier_terms(name):
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_/()%.-]+", " ", spaced)


def semantic_hints(item):
    column = item["column"].lower()
    hints = []
    if any(term in column for term in ("street", "city", "zip", "state", "latitude", "longitude")):
        hints.append("location address details")
    if "phone" in column:
        hints.append("phone number contact")
    if "open" in column:
        hints.append("opened open date year after before")
    if "closed" in column or "status" in column:
        hints.append("active closed status currently")
    if "charter" in column:
        hints.append("charter school direct funded funding")
    if "enrollment" in column or "enroll" in column:
        hints.append("enrollment students total")
    if "frpm" in column:
        hints.append("frpm free reduced price meal eligible eligibility")
    if "free meal" in column:
        hints.append("free meal eligible rate")
    if "avg" in column or "score" in column or "sat" in item["table"].lower():
        hints.append("sat performance score reading math writing")
    return " ".join(hints)


def item_to_search_text(item):
    sample_values = " ".join(str(value) for value in item.get("sample_values") or [])
    flags = "primary key foreign key" if item.get("is_pk") or item.get("is_fk") else ""
    return " ".join(
        [
            item["table"],
            item["column"],
            identifier_terms(item["table"]),
            identifier_terms(item["column"]),
            item["type"],
            item.get("description") or "",
            sample_values,
            flags,
            semantic_hints(item),
        ]
    )


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(document) for document in documents]
        self.doc_lengths = [len(document) for document in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(document) for document in self.tokenized_docs]
        self.doc_frequencies = Counter()
        for document in self.tokenized_docs:
            self.doc_frequencies.update(set(document))

    def scores(self, query):
        query_terms = tokenize(query)
        doc_count = len(self.tokenized_docs)
        scores = []
        for index, term_frequency in enumerate(self.term_frequencies):
            score = 0.0
            doc_length = self.doc_lengths[index] or 1
            for term in query_terms:
                tf = term_frequency.get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_frequencies.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


class SchemaLinker:
    def __init__(self, db_path, dense_model_name=DEFAULT_DENSE_MODEL, rrf_k=60):
        self.db_path = db_path
        self.items = build_schema_items(db_path)
        self.documents = [item_to_search_text(item) for item in self.items]
        self.bm25 = BM25Index(self.documents)
        self.dense_model_name = dense_model_name
        self.rrf_k = rrf_k
        self._dense_model = None
        self._embeddings = None

    def _load_dense_model(self):
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(self.dense_model_name)
        return self._dense_model

    def _get_embeddings(self):
        if self._embeddings is None:
            model = self._load_dense_model()
            self._embeddings = np.asarray(
                model.encode(self.documents, normalize_embeddings=True, show_progress_bar=False)
            )
        return self._embeddings

    def _dense_scores(self, query):
        model = self._load_dense_model()
        query_embedding = np.asarray(model.encode([query], normalize_embeddings=True, show_progress_bar=False))[0]
        return np.dot(self._get_embeddings(), query_embedding).tolist()

    @staticmethod
    def _rank_from_scores(scores):
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    @staticmethod
    def _query_boost(item, query_tokens):
        column = item["column"].lower()
        boost = 0.0
        if query_tokens & {"location", "locations", "address", "details"}:
            if any(term in column for term in ("street", "city", "zip", "state", "latitude", "longitude")):
                boost += 0.035
        if query_tokens & {"opened", "open", "after", "before", "year"} and "open" in column:
            boost += 0.035
        if query_tokens & {"active", "closed", "currently", "status"} and (
            "closed" in column or "status" in column
        ):
            boost += 0.035
        if query_tokens & {"phone", "numbers", "contact"} and "phone" in column:
            boost += 0.04
        if query_tokens & {"website", "web"} and "website" in column:
            boost += 0.04
        if query_tokens & {"charter"} and "charter" in column:
            boost += 0.02
        if query_tokens & {"enrollment", "students"} and ("enrollment" in column or "enroll" in column):
            boost += 0.02
        if query_tokens & {"sat", "performance", "score", "scores"}:
            if item["table"].lower() == "satscores" or "score" in column or "avg" in column:
                boost += 0.02
        return boost

    def retrieve(self, question, evidence="", top_k=DEFAULT_TOP_K):
        query = f"{question}\n{evidence or ''}"
        query_tokens = set(tokenize(query))
        bm25_rank = self._rank_from_scores(self.bm25.scores(query))
        dense_rank = self._rank_from_scores(self._dense_scores(query))
        fused_scores = Counter()

        for rank, item_index in enumerate(bm25_rank):
            fused_scores[item_index] += 1.0 / (self.rrf_k + rank + 1)
        for rank, item_index in enumerate(dense_rank):
            fused_scores[item_index] += 1.0 / (self.rrf_k + rank + 1)
        for item_index, item in enumerate(self.items):
            fused_scores[item_index] += self._query_boost(item, query_tokens)

        selected_indexes = [
            item_index
            for item_index, _ in sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        ]
        selected_items = [self.items[index] for index in selected_indexes]
        return selected_items, format_table_linked_schema_text(selected_items, self.items)
