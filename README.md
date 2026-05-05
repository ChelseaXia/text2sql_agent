# text2sql-agent

Lightweight Text2SQL research repo for local BIRD dev experiments.

## Structure

Core code now lives under `src/text2sql/`:

- `text2sql/config.py`: paths and constants
- `text2sql/data.py`: BIRD loading
- `text2sql/db.py`: SQLite execution helpers
- `text2sql/eval.py`: metric helpers
- `text2sql/llm.py`: DeepSeek client
- `text2sql/schema/`: schema parsing, items, linking, formatting
- `text2sql/prompts/`: generation, repair, and agent prompts
- `text2sql/pipelines/`: experiment pipelines
- `text2sql/agents/`: tool-calling agent, tools, trace export

CLI entrypoints now live under `scripts/`.

## New Commands

```bash
PYTHONPATH=src python3 scripts/run_naive_baseline.py --db-id california_schools --limit 5
PYTHONPATH=src python3 scripts/run_schema_linked.py --db-id california_schools --limit 5 --promptfix
PYTHONPATH=src python3 scripts/run_fewshot.py --db-id california_schools --limit 5
PYTHONPATH=src python3 scripts/run_strict_repair.py
PYTHONPATH=src python3 scripts/run_schema_format_ablation.py --db-id california_schools --limit 5
PYTHONPATH=src python3 scripts/run_agent.py --db-id california_schools --limit 5
PYTHONPATH=src python3 scripts/analyze_results.py
PYTHONPATH=src python3 scripts/analyze_traces.py
```

## Compatibility

Old `src/*.py` experiment files are kept as thin wrappers, so existing commands still resolve:

- `src/naive_baseline.py` -> `scripts/run_naive_baseline.py`
- `src/schema_linked_baseline.py` -> `scripts/run_schema_linked.py`
- `src/fewshot_baseline.py` -> `scripts/run_fewshot.py`
- `src/execution_repair_from_day3.py` -> `scripts/run_strict_repair.py`
- `src/schema_ddl_baseline.py` -> `scripts/run_schema_format_ablation.py`
- `src/text2sql_agent.py` -> `scripts/run_agent.py`

Existing `results/day*.jsonl`, `results/day*.json`, and pairwise comparison files are intentionally untouched.

The current default agent outputs are:

- `results/react_agent_traces.jsonl`
- `results/react_agent_metrics.json`

Older `day7_*` names should be treated as historical experiment naming.

The agent controller uses DeepSeek non-thinking mode for stable function calling.

Thinking mode requires preserving `reasoning_content` across tool-call turns and is not used in the default evaluation.

- `execute_sql` is used both for exploration and validation.
- Final answer is only accepted through `finish(sql)`.
- Strict metrics require explicit `finish(sql)`.
- Relaxed metrics use the last successful executed SQL only as a diagnostic fallback.
- The main Agent result uses strict metrics to avoid treating exploratory SQL as the final answer.
