# Text2SQL Working and Episodic Memory Design

## Why Memory

The iterative Text2SQL agent already benefits from within-question observations: failed SQL, SQLite errors, inspected schemas, sampled rows, and searched values. Without an explicit memory interface, those observations are hard to inspect, hard to ablate, and hard to reuse safely.

Day 3 introduces two memory layers:

- `WorkingMemory`: per-question scratchpad used during one agent attempt.
- `EpisodicMemory`: per-database session memory reused across questions from the same `db_id`.

The goal is not to give the agent oracle feedback. The goal is to reuse legitimate tool observations such as table metadata, sampled rows, and observed column values.

## Working vs Episodic Memory

`WorkingMemory` is scoped to one question. It stores:

- `intent_plan`
- `failed_sql`
- `execution_errors`
- `inspected_schema`
- `sampled_rows`
- `observed_values`
- `revised_hypotheses`
- `avoid_rules`
- `tool_observation_history`

It is useful for repair and re-planning inside a single question.

`EpisodicMemory` is scoped to one `db_id` session and is never shared across databases. It stores:

- `db_id`
- `schema_cache`
- `value_cache`
- `join_hints`
- `sampled_rows`
- `tool_observation_history`
- `access_count`
- `hit_count`
- `write_count`

It is useful when many questions hit the same database. For example, an inspected table schema or searched categorical value can help later questions in the same `db_id`.

## Read/Write Lifecycle

For each sample:

1. The runner creates a fresh `WorkingMemory`.
2. If `memory_mode=episodic`, the runner retrieves the `EpisodicMemory` object for the sample's `db_id`.
3. After schema retrieval and intent planning, episodic memory is read with the current question and intent plan.
4. The relevant memory summary is appended to prompt context as runtime observations.
5. Tool observations are written after safe tools run:
   - `retrieve_schema`
   - `inspect_table`
   - `sample_rows`
   - `search_column_values`
   - `execute_sql`
6. The trace records both `working_memory_summary` and `episodic_memory_summary`.
7. Metrics aggregate `memory_hit_count`, `memory_write_count`, and `memory_hit_rate`.

Supported modes:

- `memory_mode=off`: disables prompt memory and episodic reuse for ablation.
- `memory_mode=working`: keeps the existing per-question scratchpad behavior.
- `memory_mode=episodic`: keeps working memory and also reuses same-db episodic observations.

The default remains `working`, which preserves the previous rule-based iterative behavior.

## Leakage Prevention

Memory must not store evaluator or oracle information. The implementation filters forbidden keys before writing or exporting memory:

- `gold_sql`
- `is_correct`
- `ex`
- `exact_match`
- `exec_match`
- `evaluator_feedback`
- `oracle_correct`
- `standard_answer`

`EpisodicMemory` also enforces its database scope. If an event from another `db_id` is written into a memory object, it raises an error. This prevents cross-database memory sharing.

The memory can store successful or failed SQL execution observations, but it does not store which SQL is the standard answer and does not store evaluator correctness.

## Ablation Setup

The minimal ablation script is:

```bash
PYTHONPATH=src python3 scripts/run_memory_ablation.py
```

It reads:

- `results/iterative_agent/stratified_300_manifest.jsonl`

It selects three databases with at least 10 questions, preferring:

- `financial`
- `formula_1`
- `california_schools`

For each selected database:

- fixed seed: `42`
- random order within the database
- max 20 questions per database
- run `memory_mode=off`
- run `memory_mode=episodic`

Outputs:

- `results/iterative_agent/memory_ablation_metrics.json`
- `results/iterative_agent/memory_ablation_cases.csv`
- `results/iterative_agent/memory_ablation_traces.jsonl`

The metrics include:

- overall memory-off EX / VSR / finish rate
- overall memory-on EX / VSR / finish rate
- per-db memory-off vs memory-on
- first-question EX vs second-and-later EX
- memory hit rate
- memory write count
- average tool-call reduction
- average execute-call reduction
- `inspect_table` call reduction
- `search_column_values` call reduction

## Interpreting Results

If episodic memory improves EX, the likely explanation should be checked in traces: fewer repeated inspections, better value grounding, or better repair context.

If episodic memory does not improve EX, that is still useful. It may mean that same-db observations reduce tool calls but do not address the dominant SQL-generation errors, or that the retrieved memory is too noisy for the prompt. The ablation should be reported as-is; the evaluation should not change its correctness criterion to make memory look better.
