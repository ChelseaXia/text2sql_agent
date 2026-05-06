import json
from collections import Counter, defaultdict
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.eval import load_jsonl


EXPANDED_DIR = RESULTS_DIR / "expanded"
AUDIT_PATH = EXPANDED_DIR / "full_eval_validity_audit.json"
API_FAILED_MANIFEST_PATH = EXPANDED_DIR / "api_failed_manifest.jsonl"

METHODS = {
    "day2": EXPANDED_DIR / "day2_naive_predictions.jsonl",
    "day3": EXPANDED_DIR / "day3_schema_linked_predictions.jsonl",
    "day5_5": EXPANDED_DIR / "day5_5_strict_repair_predictions.jsonl",
    "day6": EXPANDED_DIR / "day6_ddl_predictions.jsonl",
}


def extract_sql_text(row):
    for key in ("final_sql", "predicted_sql", "pred_sql", "final_pred_sql", "initial_pred_sql"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def normalize_error_text(row):
    for key in ("failure_reason", "error", "llm_error", "pred_error", "final_error", "initial_error"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def is_executable(row):
    if "is_executable" in row:
        return bool(row.get("is_executable"))
    if "pred_success" in row:
        return bool(row.get("pred_success"))
    if "final_success" in row:
        return bool(row.get("final_success"))
    return False


def is_correct(row):
    if "is_correct" in row:
        return bool(row.get("is_correct"))
    return bool(row.get("ex"))


def classify_failure(row):
    sql_text = extract_sql_text(row).strip()
    error_text = normalize_error_text(row).lower()
    llm_error = (row.get("llm_error") or "").lower()

    api_markers = (
        "api.deepseek.com",
        "deepseek api",
        "missing deepseek api key",
        "timed out",
        "timeout",
        "name resolution",
        "failed to resolve",
        "429",
        "rate limit",
        "connection aborted",
        "max retries exceeded",
        "remote disconnected",
    )
    db_markers = (
        "unable to open database",
        "database disk image is malformed",
        "no such table",
        "no such column",
        "ambiguous column name",
        "syntax error",
        "misuse of",
        "datatype mismatch",
        "no such function",
    )

    if any(marker in error_text for marker in api_markers) or any(marker in llm_error for marker in api_markers):
        return "api_or_llm_failure"
    if not sql_text:
        if llm_error:
            return "llm_failure_empty_sql"
        raw_response = (row.get("raw_response") or "").strip()
        if raw_response:
            return "sql_extraction_failure"
        return "empty_sql_no_response"
    if not is_executable(row):
        if any(marker in error_text for marker in db_markers):
            return "execution_error"
        return "non_executable_other"
    if is_executable(row) and not is_correct(row):
        return "executable_but_incorrect"
    return "correct"


def top_counter_dict(counter, top_n=20):
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(top_n)
    ]


def aggregate_group(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)

    result = {}
    for value, subset in sorted(grouped.items(), key=lambda item: item[0]):
        total = len(subset)
        executable_count = sum(1 for row in subset if is_executable(row))
        correct_count = sum(1 for row in subset if is_correct(row))
        nonempty_sql_count = sum(1 for row in subset if extract_sql_text(row).strip())
        result[value] = {
            "total": total,
            "nonempty_sql_count": nonempty_sql_count,
            "is_executable_count": executable_count,
            "is_correct_count": correct_count,
            "VSR": executable_count / total if total else 0.0,
            "EX": correct_count / total if total else 0.0,
        }
    return result


def longest_failure_streak(rows, predicate):
    best = {"length": 0, "line_start": None, "line_end": None, "sample_id_start": None, "sample_id_end": None}
    current_start = None

    for index, row in enumerate(rows, start=1):
        if predicate(row):
            if current_start is None:
                current_start = index
        else:
            if current_start is not None:
                length = index - current_start
                if length > best["length"]:
                    start_row = rows[current_start - 1]
                    end_row = rows[index - 2]
                    best = {
                        "length": length,
                        "line_start": current_start,
                        "line_end": index - 1,
                        "sample_id_start": start_row.get("sample_id"),
                        "sample_id_end": end_row.get("sample_id"),
                    }
                current_start = None

    if current_start is not None:
        length = len(rows) - current_start + 1
        if length > best["length"]:
            start_row = rows[current_start - 1]
            end_row = rows[-1]
            best = {
                "length": length,
                "line_start": current_start,
                "line_end": len(rows),
                "sample_id_start": start_row.get("sample_id"),
                "sample_id_end": end_row.get("sample_id"),
            }
    return best


def sample_failure_windows(rows, window_size=20, min_failures=15):
    windows = []
    for start in range(0, len(rows) - window_size + 1):
        window = rows[start : start + window_size]
        failures = sum(1 for row in window if not is_executable(row))
        if failures >= min_failures:
            windows.append(
                {
                    "line_start": start + 1,
                    "line_end": start + window_size,
                    "sample_id_start": window[0].get("sample_id"),
                    "sample_id_end": window[-1].get("sample_id"),
                    "non_executable_count": failures,
                }
            )
    return windows[:20]


def audit_method(name, path):
    rows = load_jsonl(path)
    error_counter = Counter()
    llm_error_counter = Counter()
    failure_reason_counter = Counter()
    failure_class_counter = Counter()
    api_failed_rows = []

    for row in rows:
        error_text = normalize_error_text(row)
        if error_text:
            error_counter[error_text] += 1
        llm_error = (row.get("llm_error") or "").strip()
        if llm_error:
            llm_error_counter[llm_error] += 1
        failure_reason = (row.get("failure_reason") or "").strip()
        if failure_reason:
            failure_reason_counter[failure_reason] += 1

        failure_class = classify_failure(row)
        failure_class_counter[failure_class] += 1
        if failure_class == "api_or_llm_failure":
            api_failed_rows.append(row)

    total = len(rows)
    nonempty_sql_count = sum(1 for row in rows if extract_sql_text(row).strip())
    executable_count = sum(1 for row in rows if is_executable(row))
    correct_count = sum(1 for row in rows if is_correct(row))

    return {
        "path": str(path),
        "total": total,
        "nonempty_sql_count": nonempty_sql_count,
        "nonempty_sql_rate": nonempty_sql_count / total if total else 0.0,
        "is_executable_count": executable_count,
        "VSR": executable_count / total if total else 0.0,
        "is_correct_count": correct_count,
        "EX": correct_count / total if total else 0.0,
        "top_error_counts": top_counter_dict(error_counter),
        "top_failure_reason_counts": top_counter_dict(failure_reason_counter),
        "top_llm_error_counts": top_counter_dict(llm_error_counter),
        "failure_class_counts": dict(failure_class_counter),
        "by_db_id": aggregate_group(rows, "db_id"),
        "by_difficulty": aggregate_group(rows, "difficulty"),
        "longest_non_executable_streak": longest_failure_streak(rows, lambda row: not is_executable(row)),
        "longest_api_failure_streak": longest_failure_streak(rows, lambda row: classify_failure(row) == "api_or_llm_failure"),
        "high_failure_windows": sample_failure_windows(rows),
        "api_failed_rows": api_failed_rows,
    }


def build_api_failed_manifest(audits):
    merged = {}
    for method_name, audit in audits.items():
        for row in audit["api_failed_rows"]:
            sample_id = row.get("sample_id")
            if sample_id in merged:
                merged[sample_id]["failed_methods"].append(method_name)
                continue
            merged[sample_id] = {
                "sample_id": sample_id,
                "db_id": row.get("db_id"),
                "difficulty": row.get("difficulty"),
                "question": row.get("question"),
                "gold_sql": row.get("gold_sql"),
                "db_path": row.get("db_path"),
                "evidence": row.get("evidence", ""),
                "failed_methods": [method_name],
                "failure_reason": normalize_error_text(row),
            }
    records = sorted(merged.values(), key=lambda row: (row["sample_id"] is None, row["sample_id"]))
    with API_FAILED_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "count": len(records),
        "path": str(API_FAILED_MANIFEST_PATH),
    }


def main():
    audits = {name: audit_method(name, path) for name, path in METHODS.items()}

    summary_consistency = {}
    for name, audit in audits.items():
        summary_consistency[name] = {
            "total_matches_1534": audit["total"] == 1534,
            "low_vsr_not_due_to_missing_rows": audit["total"] == 1534,
            "nonempty_sql_rate": audit["nonempty_sql_rate"],
            "VSR": audit["VSR"],
            "EX": audit["EX"],
        }

    api_failed_manifest = build_api_failed_manifest(audits)

    root_cause_summary = {}
    for name, audit in audits.items():
        counts = audit["failure_class_counts"]
        root_cause_summary[name] = {
            "dominant_failure_class": max(counts.items(), key=lambda item: item[1])[0] if counts else None,
            "api_or_llm_failure_count": counts.get("api_or_llm_failure", 0),
            "sql_extraction_failure_count": counts.get("sql_extraction_failure", 0),
            "execution_error_count": counts.get("execution_error", 0),
            "executable_but_incorrect_count": counts.get("executable_but_incorrect", 0),
        }

    output = {
        "total_expected_samples": 1534,
        "methods": {
            name: {
                key: value
                for key, value in audit.items()
                if key != "api_failed_rows"
            }
            for name, audit in audits.items()
        },
        "summary_consistency": summary_consistency,
        "root_cause_summary": root_cause_summary,
        "api_failed_manifest": api_failed_manifest,
    }
    AUDIT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
