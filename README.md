# text2sql-agent

## Project Overview

This repository is a lightweight Text2SQL research project built around local experiments on the BIRD dev `california_schools` database. It studies how far a practical enterprise-style Text2SQL stack can go with schema retrieval, prompt fixing, and execution-guided repair, without claiming a full BIRD benchmark result or any leaderboard-valid number.

The main result of the project is the Day5.5 strict execution-repair pipeline:

- EX = `36%`
- VSR = `100%`

This is the primary reported result in the repository. The controlled agent is a trace-based implementation close to that pipeline, while the autonomous ReAct agent is kept as a negative ablation.

## Motivation: Text2SQL for Enterprise BI

Enterprise BI questions are rarely answered by a single prompt over a full raw schema. Real settings usually require:

- selecting relevant tables and columns from large schemas,
- reducing column hallucination,
- handling execution failures robustly,
- and distinguishing executable SQL from semantically correct SQL.

This project treats Text2SQL as a practical systems problem rather than a pure prompt-design problem. The focus is on reliable local experimentation, interpretable pipelines, and ablations that help explain where performance comes from.

## Dataset and Evaluation

The experiments in this repository use:

- BIRD dev `california_schools`
- a fixed `50`-sample subset

Important scope note:

- this is not a full BIRD benchmark evaluation,
- this is not a leaderboard submission,
- all reported numbers are for the saved local subset experiments only.

Sample alignment:

- `sample_id` is aligned across Day5.5, the autonomous ReAct agent, and the controlled agent.
- This allows apples-to-apples comparison across those saved 50-sample runs.

Difficulty distribution of the current 50-sample subset:

- `simple = 15`
- `moderate = 16`
- `challenging = 19`

Metrics:

- `EX`: execution-match accuracy. A prediction is counted correct only when its execution result matches the gold SQL result.
- `VSR`: valid SQL rate. A prediction is counted valid when the produced SQL executes successfully.

Model setting:

- The saved results in this repository are reported under a single DeepSeek API setting: `deepseek-v4-flash`.

## Method

### Naive full schema baseline

The starting point is a direct Text2SQL baseline that exposes the full schema to the model and asks for a one-shot SQL generation.

### Schema linking: BM25 + dense retrieval + RRF

To reduce irrelevant schema context, the project uses a hybrid schema linker:

- BM25 lexical retrieval
- dense retrieval
- reciprocal rank fusion (RRF)

The linker selects task-relevant tables and columns before SQL generation.

### Schema-aware promptfix

After schema retrieval, the generation prompt is made more schema-aware and more explicit about SQL formatting and answer constraints. This Day3 setting is the first main quality jump over the naive baseline.

### Execution-guided repair

The Day5.5 pipeline adds strict execution repair:

- generate SQL from the schema-linked prompt,
- execute it,
- and only invoke repair when execution fails.

This improves robustness and raises VSR to `100%` on the saved subset.

### Controlled execution-repair agent

The controlled agent is a trace-oriented implementation of the same basic Day3 plus Day5.5 logic:

1. retrieve schema
2. generate SQL
3. execute SQL
4. repair if execution fails
5. execute repaired SQL
6. finish with the final SQL

It is useful for agent-style tracing and interpretability, but it is not the canonical reported metric. The canonical main result remains the saved Day5.5 strict execution-repair run.

The saved Day5.5 run remains the canonical metric because `controlled_agent` is a rerun-based trace implementation; minor differences can arise from LLM sampling variance even at `temperature=0`.

### Autonomous ReAct agent ablation

The autonomous ReAct agent exposes more tool freedom and longer tool-calling trajectories. In this project it is treated as a negative ablation, not as the best method, because it underperformed the controlled execution-repair pipeline.

## Main Results Table

| Method | EX | VSR | Role |
| --- | ---: | ---: | --- |
| Day2 naive full schema | 26% | 86% | baseline |
| Day3 schema-linked + promptfix | 34% | 88% | improved baseline |
| Day5.5 strict execution repair | 36% | 100% | main result |
| Controlled agent trace implementation | 32% | 98% | near-match implementation |
| Few-shot retrieval | 28% | 100% | negative ablation |
| DDL schema serialization | 28% | 82% | negative ablation |
| Autonomous ReAct agent | 12% | 88% | negative ablation |

Notes:

- The main reported number is Day5.5 strict execution repair: `EX=36%`, `VSR=100%`.
- The controlled agent is close to Day5.5 but is not the canonical metric run.
- The autonomous ReAct agent should not be interpreted as the best method in this repository.

## Agent Architecture Comparison

### Controlled execution-repair agent vs Autonomous ReAct agent

The controlled agent and the autonomous ReAct agent answer a similar high-level question, but they represent two very different control regimes.

Controlled agent:

- stable fixed graph,
- explicit schema retrieval,
- single generation step,
- repair only on execution failure,
- low tool-call overhead.

Autonomous ReAct agent:

- free-form tool choice,
- longer interaction traces,
- heavier exploration,
- unstable finalization,
- much lower EX despite many executable SQLs.

Observed results on the saved runs:

- Controlled agent: `EX=32%`, `VSR=98%`
- Autonomous ReAct agent: `EX=12%`, `VSR=88%`

The controlled agent also matched the Day5.5 EX outcome on `48/50` samples, which further supports the conclusion that the fixed execution-repair graph is the more reliable direction in this repo.

## Error Analysis

Main takeaways from the saved experiments:

- Schema linking helps reduce column hallucination.
  Day3 improved EX from `26%` to `34%` relative to the naive full-schema baseline.

- Execution repair mainly improves VSR.
  Day5.5 raised VSR from `88%` to `100%` and produced the best overall result in the project.

- Few-shot retrieval caused semantic negative transfer.
  It kept VSR high at `100%` but dropped EX to `28%`, suggesting that retrieved examples sometimes biased generation toward the wrong query pattern.

- DDL serialization degraded performance.
  The DDL schema format reached only `EX=28%`, `VSR=82%`, which indicates that this serialization style was less effective than the linked-schema representation used in the main pipeline.

- Challenging-tier questions remain unresolved.
  The strongest gains came from making SQL more executable and less hallucinated, but deeper multi-hop planning and semantic composition remain bottlenecks.

For a longer write-up, see [docs/error_analysis.md](docs/error_analysis.md).

## Limitations

- The evaluation is limited to a `50`-sample `california_schools` subset.
- The project does not claim full BIRD benchmark coverage.
- Challenging-tier query planning remains the main bottleneck.
- The controlled-agent rerun did not exactly reproduce the saved Day5.5 metric:
  `32%` vs saved Day5.5 `36%`, with `48/50` EX outcomes matched.
- No model fine-tuning was used.
- The experiments use a single LLM setting rather than a broad model comparison.

## Future Work

- Plan-then-SQL for harder compositional queries
- SQL skeleton retrieval rather than example-level few-shot transfer
- a semantic verifier for executable but semantically wrong SQL
- multi-database evaluation on `150` or `200` samples
- optional QLoRA SFT if GPU budget allows

## Project Structure

The repository is organized as:

- `src/text2sql/` = active package code
- `scripts/` = runnable CLI entrypoints
- `docs/` = final writeups
- `results/` = selected frozen result summaries
- `archive/` = legacy scripts and full intermediate artifacts

Active package modules under `src/text2sql/`:

- `text2sql/config.py`
- `text2sql/data.py`
- `text2sql/db.py`
- `text2sql/eval.py`
- `text2sql/llm.py`
- `text2sql/schema/`
- `text2sql/prompts/`
- `text2sql/pipelines/`
- `text2sql/agents/`

CLI entrypoints live under `scripts/`.

Useful scripts:

```bash
PYTHONPATH=src python3 scripts/run_naive_baseline.py --db-id california_schools --limit 5
PYTHONPATH=src python3 scripts/run_schema_linked.py --db-id california_schools --limit 5 --promptfix
PYTHONPATH=src python3 scripts/run_fewshot.py --db-id california_schools --limit 5
PYTHONPATH=src python3 scripts/run_strict_repair.py
PYTHONPATH=src python3 scripts/run_controlled_agent.py --db-id california_schools --limit 50
PYTHONPATH=src python3 scripts/run_agent.py --db-id california_schools --limit 50
PYTHONPATH=src python3 scripts/compare_controlled_with_day5.py
```

Legacy day-based `src/*.py` entrypoints are preserved under `archive/legacy_src/`.
