#!/usr/bin/env bash

set -u

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <manifest-path>"
  exit 1
fi

MANIFEST_PATH="$1"
RESULTS_DIR="results/expanded"
LOG_DIR="logs"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-}"
EMBEDDING_ARGS=()

if [ -n "$EMBEDDING_MODEL_PATH" ]; then
  EMBEDDING_ARGS=(--embedding-model-path "$EMBEDDING_MODEL_PATH")
fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

run_with_retry() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"

  echo "Running ${name}..."
  if "$@" >"$log_path" 2>&1; then
    echo "${name} completed."
    return 0
  fi

  echo "${name} failed. Retrying once..."
  if "$@" >>"$log_path" 2>&1; then
    echo "${name} completed on retry."
    return 0
  fi

  echo "${name} failed again. See ${log_path}"
  return 1
}

run_with_retry \
  "expanded_day2_naive" \
  env PYTHONPATH=src python3 scripts/run_naive_baseline.py \
    --manifest "$MANIFEST_PATH" \
    --output "$RESULTS_DIR/day2_naive_predictions.jsonl" \
    --metrics-output "$RESULTS_DIR/day2_naive_metrics.json" || exit 1

run_with_retry \
  "expanded_day3_schema_linked" \
  env PYTHONPATH=src python3 scripts/run_schema_linked.py \
    --manifest "$MANIFEST_PATH" \
    --promptfix \
    "${EMBEDDING_ARGS[@]}" \
    --predictions-output "$RESULTS_DIR/day3_schema_linked_predictions.jsonl" \
    --metrics-output "$RESULTS_DIR/day3_schema_linked_metrics.json" \
    --debug-output "$RESULTS_DIR/day3_schema_linked_debug.jsonl" || exit 1

run_with_retry \
  "expanded_day5_5_strict_repair" \
  env PYTHONPATH=src python3 scripts/run_strict_repair.py \
    --manifest "$MANIFEST_PATH" \
    --day3-input "$RESULTS_DIR/day3_schema_linked_predictions.jsonl" \
    "${EMBEDDING_ARGS[@]}" \
    --predictions-output "$RESULTS_DIR/day5_5_strict_repair_predictions.jsonl" \
    --traces-output "$RESULTS_DIR/day5_5_strict_repair_traces.jsonl" \
    --metrics-output "$RESULTS_DIR/day5_5_strict_repair_metrics.json" \
    --pairwise-output "$RESULTS_DIR/day5_5_strict_repair_pairwise_compare.csv" || exit 1

run_with_retry \
  "expanded_day6_ddl" \
  env PYTHONPATH=src python3 scripts/run_schema_format_ablation.py \
    --manifest "$MANIFEST_PATH" \
    "${EMBEDDING_ARGS[@]}" \
    --predictions-output "$RESULTS_DIR/day6_ddl_predictions.jsonl" \
    --metrics-output "$RESULTS_DIR/day6_ddl_metrics.json" \
    --debug-output "$RESULTS_DIR/day6_ddl_debug.jsonl" || exit 1

env PYTHONPATH=src python3 scripts/summarize_expanded_eval.py >"$LOG_DIR/expanded_summary.log" 2>&1
