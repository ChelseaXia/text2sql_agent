# Suspicion-Triggered Iterative Tool-Using Agent Design

This experiment adds a controlled iterative Text-to-SQL agent for cases where execution-result self-consistency is bottlenecked by generation rather than selection.

## Motivation

Execution-result self-consistency voting on the compact `california_schools` subset reached:

- Selected EX: `38%`
- Oracle EX@10: `40%`
- Selection bottleneck: `1` sample
- Generation bottleneck: `30` samples
- Challenging subset: `19/19` samples had Oracle=False

The next target is therefore candidate generation and semantic planning under complex queries, not more voting.

## Control Policy

This is not a free-form ReAct agent. The controller owns the loop, tool budget, and state transitions.

Allowed tools:

- `retrieve_schema(question, db_id)`
- `inspect_table(db_id, table_name)`
- `sample_rows(db_id, table_name, n=5)`
- `search_column_values(db_id, table_name, column_name, query)`
- `execute_sql(db_id, sql)`
- `finish(final_sql)`

Default budget:

- `max_steps = 5`
- At most one exploration tool after each suspicious successful execution.
- Execution failures trigger repair or re-planning directly.
- The agent never receives `gold_sql` or evaluator correctness during the loop.

## Loop

1. Retrieve linked schema.
2. Generate initial SQL.
3. Execute SQL.
4. If execution failed, add failed SQL and error to the within-turn scratchpad, then repair or re-plan.
5. If execution succeeded but looks suspicious, select exactly one exploration tool.
6. Add the observation to the within-turn scratchpad.
7. Regenerate SQL from schema, current SQL, suspicion reason, and scratchpad.
8. Repeat until no suspicion is detected or `max_steps` is reached.
9. Finish with the final SQL.

## Suspicion Rules

The controller currently triggers on:

- Empty result.
- Scalar result while the question asks for a list, top-k, grouping, or ranking.
- Question mentions an entity or value but SQL has no `WHERE`.
- Query likely requires multiple tables but SQL has no `JOIN`.
- SQL uses literal values that may not exist in the database.
- Execution error.

## Memory

Only within-turn scratchpad memory is implemented:

- Previous failed SQL.
- Execution errors.
- Inspected table schema.
- Sampled rows.
- Searched values.
- Revised hypothesis.

Cross-sample memory is intentionally not implemented.

## Evaluation Plan

Manifests live under `results/iterative_agent/`:

- `california_schools_50_manifest.jsonl`
- `challenging_19_manifest.jsonl`

Run order:

1. 5-sample smoke from the compact 50 manifest.
2. 19 challenging samples.
3. Compact 50 samples.

Metrics:

- EX
- VSR
- Finish rate
- Average tool calls
- Average execute calls
- Suspicious trigger count
- Exploration count
- Repair count
- Results by difficulty

Trace JSONL fields:

- `sample_id`
- `question`
- `step`
- `action`
- `tool_input`
- `observation`
- `suspicion_reason`
- `scratchpad_summary`
- `current_sql`
- `final_sql`
- `is_correct`

## Commands

Create default manifests:

```bash
PYTHONPATH=src python3 scripts/run_iterative_agent.py --write-default-manifests
```

Run 5-sample smoke:

```bash
PYTHONPATH=src python3 scripts/run_iterative_agent.py \
  --manifest results/iterative_agent/california_schools_50_manifest.jsonl \
  --limit 5 \
  --predictions-output results/iterative_agent/smoke_5_predictions.jsonl \
  --metrics-output results/iterative_agent/smoke_5_metrics.json \
  --traces-output results/iterative_agent/smoke_5_traces.jsonl
```

Run challenging-heavy subset:

```bash
PYTHONPATH=src python3 scripts/run_iterative_agent.py \
  --manifest results/iterative_agent/challenging_19_manifest.jsonl \
  --predictions-output results/iterative_agent/challenging_19_predictions.jsonl \
  --metrics-output results/iterative_agent/challenging_19_metrics.json \
  --traces-output results/iterative_agent/challenging_19_traces.jsonl
```

Run compact 50:

```bash
PYTHONPATH=src python3 scripts/run_iterative_agent.py \
  --manifest results/iterative_agent/california_schools_50_manifest.jsonl \
  --predictions-output results/iterative_agent/california_schools_50_predictions.jsonl \
  --metrics-output results/iterative_agent/california_schools_50_metrics.json \
  --traces-output results/iterative_agent/california_schools_50_traces.jsonl
```
