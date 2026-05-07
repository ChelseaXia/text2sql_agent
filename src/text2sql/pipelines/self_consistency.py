"""Execution-result self-consistency voting pipeline."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import resolve_eval_samples
from text2sql.db import normalize_rows, run_sql, same_result
from text2sql.llm import DEFAULT_SEED, call_llm
from text2sql.pipelines.naive import extract_sql
from text2sql.pipelines.schema_linked import build_schema_linker, retrieved_column_names
from text2sql.pipelines.strict_repair import generate_repaired_sql
from text2sql.prompts.generation import build_linked_prompt
from text2sql.schema.linker import DEFAULT_LINKER_MODE, DEFAULT_TOP_K

METHOD_NAME = "execution_result_self_consistency_voting"
DEFAULT_PREDICTIONS_PATH = RESULTS_DIR / "day7_self_consistency_predictions.jsonl"
DEFAULT_METRICS_PATH = RESULTS_DIR / "day7_self_consistency_metrics.json"
DEFAULT_DEBUG_PATH = RESULTS_DIR / "day7_self_consistency_debug.jsonl"


def generate_sql_candidate(sample, linked_schema_text, temperature, candidate_id):
    raw_response = ""
    try:
        raw_response = call_llm(
            build_linked_prompt(sample, linked_schema_text),
            temperature=temperature,
            seed=DEFAULT_SEED + candidate_id,
            use_cache=False,
        )
        return extract_sql(raw_response), raw_response, None
    except Exception as exc:
        return "", "", str(exc)


def make_execution_result(sql, db_path, llm_error=None):
    if not sql:
        return {"success": False, "rows": [], "error": llm_error or "Empty SQL prediction"}
    return run_sql(sql, db_path)


def hash_rows(rows):
    normalized = normalize_rows(rows)
    canonical_rows = [
        {"row": list(row), "count": count}
        for row, count in sorted(normalized.items(), key=lambda item: (repr(item[0]), item[1]))
    ]
    payload = json.dumps(canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def cluster_sort_key(cluster):
    representative = cluster["representative"]
    return (
        -cluster["size"],
        1 if representative["was_repaired"] else 0,
        len(representative["final_sql"] or ""),
        representative["candidate_id"],
    )


def candidate_sort_key(candidate):
    return (
        1 if candidate["was_repaired"] else 0,
        len(candidate["final_sql"] or ""),
        candidate["candidate_id"],
    )


def assign_clusters(candidates):
    executable_candidates = [candidate for candidate in candidates if candidate["is_executable"]]
    grouped = defaultdict(list)
    for candidate in executable_candidates:
        grouped[candidate["result_hash"]].append(candidate)

    clusters = []
    for result_hash, members in grouped.items():
        representative = min(members, key=candidate_sort_key)
        clusters.append(
            {
                "result_hash": result_hash,
                "members": members,
                "size": len(members),
                "representative": representative,
            }
        )

    clusters.sort(key=cluster_sort_key)
    cluster_id_by_hash = {}
    for index, cluster in enumerate(clusters):
        cluster_id = index
        cluster_id_by_hash[cluster["result_hash"]] = cluster_id
        for member in cluster["members"]:
            member["cluster_id"] = cluster_id

    for candidate in candidates:
        if not candidate["is_executable"]:
            candidate["cluster_id"] = None

    return clusters


def summarize_selected_sample(sample, candidates, clusters):
    valid_candidates = [candidate for candidate in candidates if candidate["is_executable"]]
    oracle_correct = any(candidate["is_correct"] for candidate in candidates)

    if not clusters:
        return {
            "selected_sql": "",
            "selected_candidate_id": None,
            "selected_cluster_id": None,
            "selected_cluster_size": 0,
            "total_candidate_count": len(candidates),
            "valid_candidate_count": len(valid_candidates),
            "cluster_count": 0,
            "cluster_confidence": 0.0,
            "oracle_correct": oracle_correct,
            "selected_correct": False,
            "selected_executable": False,
            "selected_error": "No executable candidate",
            "selected_failure_reason": "No executable candidate",
        }

    selected_cluster = clusters[0]
    selected_candidate = selected_cluster["representative"]
    valid_candidate_count = len(valid_candidates)
    return {
        "selected_sql": selected_candidate["final_sql"],
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_cluster_id": selected_candidate["cluster_id"],
        "selected_cluster_size": selected_cluster["size"],
        "total_candidate_count": len(candidates),
        "valid_candidate_count": valid_candidate_count,
        "cluster_count": len(clusters),
        "cluster_confidence": (selected_cluster["size"] / valid_candidate_count) if valid_candidate_count else 0.0,
        "oracle_correct": oracle_correct,
        "selected_correct": selected_candidate["is_correct"],
        "selected_executable": selected_candidate["is_executable"],
        "selected_error": selected_candidate["error"],
        "selected_failure_reason": selected_candidate["failure_reason"],
    }


def empty_bucket():
    return {
        "sample_count": 0,
        "selected_ex_count": 0,
        "selected_vsr_count": 0,
        "oracle_ex_count": 0,
        "valid_candidate_total": 0,
        "cluster_total": 0,
        "cluster_confidence_total": 0.0,
        "generation_bottleneck_count": 0,
        "selection_bottleneck_count": 0,
    }


def finalize_bucket(bucket):
    sample_count = bucket["sample_count"]
    return {
        "sample_count": sample_count,
        "selected_EX": bucket["selected_ex_count"] / sample_count if sample_count else 0.0,
        "selected_VSR": bucket["selected_vsr_count"] / sample_count if sample_count else 0.0,
        "oracle_EX_at_K": bucket["oracle_ex_count"] / sample_count if sample_count else 0.0,
        "avg_valid_candidate_count": bucket["valid_candidate_total"] / sample_count if sample_count else 0.0,
        "avg_cluster_count": bucket["cluster_total"] / sample_count if sample_count else 0.0,
        "avg_cluster_confidence": bucket["cluster_confidence_total"] / sample_count if sample_count else 0.0,
        "generation_bottleneck_count": bucket["generation_bottleneck_count"],
        "selection_bottleneck_count": bucket["selection_bottleneck_count"],
    }


def compute_self_consistency_metrics(records):
    overall = empty_bucket()
    by_difficulty = {}
    by_database = {}

    for record in records:
        selected_correct = bool(record["selected_correct"])
        selected_executable = bool(record["selected_executable"])
        oracle_correct = bool(record["oracle_correct"])
        generation_bottleneck = not oracle_correct
        selection_bottleneck = oracle_correct and (not selected_correct)

        for bucket in (
            overall,
            by_difficulty.setdefault(record["difficulty"], empty_bucket()),
            by_database.setdefault(record["db_id"], empty_bucket()),
        ):
            bucket["sample_count"] += 1
            bucket["selected_ex_count"] += 1 if selected_correct else 0
            bucket["selected_vsr_count"] += 1 if selected_executable else 0
            bucket["oracle_ex_count"] += 1 if oracle_correct else 0
            bucket["valid_candidate_total"] += record["valid_candidate_count"]
            bucket["cluster_total"] += record["cluster_count"]
            bucket["cluster_confidence_total"] += record["cluster_confidence"]
            bucket["generation_bottleneck_count"] += 1 if generation_bottleneck else 0
            bucket["selection_bottleneck_count"] += 1 if selection_bottleneck else 0

    return {
        "sample_count": overall["sample_count"],
        "selected_EX": overall["selected_ex_count"] / overall["sample_count"] if overall["sample_count"] else 0.0,
        "selected_VSR": overall["selected_vsr_count"] / overall["sample_count"] if overall["sample_count"] else 0.0,
        "oracle_EX_at_K": overall["oracle_ex_count"] / overall["sample_count"] if overall["sample_count"] else 0.0,
        "avg_valid_candidate_count": overall["valid_candidate_total"] / overall["sample_count"] if overall["sample_count"] else 0.0,
        "avg_cluster_count": overall["cluster_total"] / overall["sample_count"] if overall["sample_count"] else 0.0,
        "avg_cluster_confidence": overall["cluster_confidence_total"] / overall["sample_count"] if overall["sample_count"] else 0.0,
        "generation_bottleneck_count": overall["generation_bottleneck_count"],
        "selection_bottleneck_count": overall["selection_bottleneck_count"],
        "by_difficulty": {name: finalize_bucket(bucket) for name, bucket in sorted(by_difficulty.items())},
        "by_database": {name: finalize_bucket(bucket) for name, bucket in sorted(by_database.items())},
    }


def build_prediction_record(sample, linked_schema_text, retrieved_columns, candidates, selected_summary):
    return {
        "sample_id": sample["sample_id"],
        "db_id": sample["db_id"],
        "db_path": sample["db_path"],
        "difficulty": sample["difficulty"],
        "question": sample["question"],
        "evidence": sample["evidence"],
        "gold_sql": sample["gold_sql"],
        "method": METHOD_NAME,
        "schema_linker_mode": DEFAULT_LINKER_MODE,
        "retrieved_columns": retrieved_columns,
        "linked_schema_text": linked_schema_text,
        "candidates": candidates,
        "selected_sql": selected_summary["selected_sql"],
        "selected_candidate_id": selected_summary["selected_candidate_id"],
        "selected_cluster_id": selected_summary["selected_cluster_id"],
        "selected_cluster_size": selected_summary["selected_cluster_size"],
        "total_candidate_count": selected_summary["total_candidate_count"],
        "valid_candidate_count": selected_summary["valid_candidate_count"],
        "cluster_count": selected_summary["cluster_count"],
        "cluster_confidence": selected_summary["cluster_confidence"],
        "oracle_correct": selected_summary["oracle_correct"],
        "selected_correct": selected_summary["selected_correct"],
        "selected_executable": selected_summary["selected_executable"],
        "pred_sql": selected_summary["selected_sql"],
        "predicted_sql": selected_summary["selected_sql"],
        "final_sql": selected_summary["selected_sql"],
        "pred_success": selected_summary["selected_executable"],
        "is_executable": selected_summary["selected_executable"],
        "ex": selected_summary["selected_correct"],
        "is_correct": selected_summary["selected_correct"],
        "error": selected_summary["selected_error"],
        "failure_reason": selected_summary["selected_failure_reason"],
        "gold_success": True,
    }


def run_self_consistency_voting(
    samples,
    predictions_path,
    metrics_path,
    debug_path=None,
    k=10,
    temperature=0.7,
    enable_repair=True,
    embedding_model_path=None,
    top_k_schema=DEFAULT_TOP_K,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linker_cache = {}
    records = []

    debug_file = debug_path.open("w", encoding="utf-8") if debug_path else None
    try:
        with predictions_path.open("w", encoding="utf-8") as predictions_file:
            for index, sample in enumerate(samples, start=1):
                db_path = sample["db_path"]
                if db_path not in linker_cache:
                    linker_cache[db_path] = build_schema_linker(
                        db_path,
                        top_k=top_k_schema,
                        schema_linker_mode=DEFAULT_LINKER_MODE,
                        embedding_model_path=embedding_model_path,
                    )

                linked_items, linked_schema_text = linker_cache[db_path].retrieve(
                    sample["question"],
                    sample["evidence"],
                    top_k=top_k_schema,
                )
                retrieved_columns = retrieved_column_names(linked_items)
                gold_result = run_sql(sample["gold_sql"], db_path)
                if not gold_result["success"]:
                    raise RuntimeError(f"Gold SQL failed for sample_id={sample['sample_id']}: {gold_result['error']}")

                candidates = []
                debug_candidates = []
                for candidate_index in range(k):
                    candidate_id = candidate_index
                    raw_sql, raw_response, llm_error = generate_sql_candidate(
                        sample=sample,
                        linked_schema_text=linked_schema_text,
                        temperature=temperature,
                        candidate_id=candidate_id,
                    )
                    initial_result = make_execution_result(raw_sql, db_path, llm_error)

                    final_sql = raw_sql
                    final_result = initial_result
                    was_repaired = False
                    repair_error = None
                    repair_raw_response = ""

                    if enable_repair and (not initial_result["success"]):
                        was_repaired = True
                        repaired_sql, repair_raw_response, repair_error = generate_repaired_sql(
                            question=sample["question"],
                            evidence=sample.get("evidence", ""),
                            linked_schema_text=linked_schema_text,
                            previous_sql=raw_sql,
                            sqlite_error=initial_result["error"] or llm_error or "Unknown SQLite error",
                        )
                        repaired_result = make_execution_result(repaired_sql, db_path, repair_error)
                        final_sql = repaired_sql
                        final_result = repaired_result

                    is_executable = bool(final_result["success"])
                    is_correct = bool(is_executable and same_result(final_result["rows"], gold_result["rows"]))
                    failure_reason = None if is_executable else (
                        repair_error
                        or final_result["error"]
                        or llm_error
                        or "Unknown prediction failure"
                    )
                    result_hash = hash_rows(final_result["rows"]) if is_executable else None

                    candidate = {
                        "candidate_id": candidate_id,
                        "raw_sql": raw_sql,
                        "final_sql": final_sql,
                        "was_repaired": was_repaired,
                        "is_executable": is_executable,
                        "is_correct": is_correct,
                        "result_hash": result_hash,
                        "cluster_id": None,
                        "error": failure_reason,
                        "failure_reason": failure_reason,
                    }
                    candidates.append(candidate)
                    debug_candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "raw_response": raw_response,
                            "repair_raw_response": repair_raw_response,
                            "initial_error": initial_result["error"],
                            "final_error": final_result["error"],
                        }
                    )

                clusters = assign_clusters(candidates)
                selected_summary = summarize_selected_sample(sample, candidates, clusters)
                record = build_prediction_record(sample, linked_schema_text, retrieved_columns, candidates, selected_summary)
                records.append(record)
                predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                predictions_file.flush()

                if debug_file is not None:
                    debug_file.write(
                        json.dumps(
                            {
                                "sample_id": sample["sample_id"],
                                "db_id": sample["db_id"],
                                "difficulty": sample["difficulty"],
                                "retrieved_columns": retrieved_columns,
                                "linked_schema_text": linked_schema_text,
                                "debug_candidates": debug_candidates,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    debug_file.flush()

                print(
                    f"{index}: sample_id={sample['sample_id']} "
                    f"selected_executable={record['selected_executable']} "
                    f"selected_correct={record['selected_correct']} "
                    f"oracle_correct={record['oracle_correct']}"
                )
    finally:
        if debug_file is not None:
            debug_file.close()

    metrics = compute_self_consistency_metrics(records)
    metrics["k"] = k
    metrics["temperature"] = temperature
    metrics["enable_repair"] = enable_repair
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "debug_path": str(debug_path) if debug_path else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run execution-result self-consistency voting.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--db-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable-repair", dest="enable_repair", action="store_true")
    parser.add_argument("--disable-repair", dest="enable_repair", action="store_false")
    parser.set_defaults(enable_repair=True)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--debug-output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.manifest is None and args.db_id is None:
        args.db_id = "california_schools"
    if args.manifest is None and args.limit is None:
        args.limit = 50

    samples = resolve_eval_samples(limit=args.limit, db_id=args.db_id, manifest_path=args.manifest)
    summary = run_self_consistency_voting(
        samples=samples,
        predictions_path=args.predictions_output,
        metrics_path=args.metrics_output,
        debug_path=args.debug_output,
        k=args.k,
        temperature=args.temperature,
        enable_repair=args.enable_repair,
        embedding_model_path=args.embedding_model_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
