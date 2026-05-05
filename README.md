# text2sql-agent

## TL;DR

This repository is a compact Text2SQL systems project on a fixed 50-sample subset of BIRD dev `california_schools`. The core result is not just `EX=36%` in isolation, but a staged pipeline improvement from `26%` to `34%` to `36%`, a `38.5%` relative EX gain over the naive baseline, and a `VSR` lift from `88%` to `100%`.

The repo also includes an agent-control ablation: a controlled execution-repair graph reaches `EX=32%`, while an autonomous ReAct agent drops to `EX=12%`. The main takeaway is that when task structure is known, explicit control beats free-form tool use in this setup.

```mermaid
flowchart LR
    A[User question]
    B[Schema linker]
    C[SQL generation]
    D[SQLite execution]
    E{Execution fails?}
    F[Conditional repair]
    G[Final SQL]

    A --> B --> C --> D --> E
    E -- No --> G
    E -- Yes --> F --> G
```

```mermaid
flowchart TB
    subgraph Controlled Graph
        C1[Retrieve schema] --> C2[Generate SQL]
        C2 --> C3[Execute]
        C3 --> C4[Repair only if needed]
        C4 --> C5[Finalize]
    end

    subgraph Autonomous ReAct
        R1[Free-form tool choice] --> R2[Longer exploration]
        R2 --> R3[Repeated tool calls]
        R3 --> R4[Unstable finalization]
    end
```

## Why This Repo Is Worth Reading

- It shows a practical Text2SQL stack rather than a single-prompt demo.
- It isolates where gains come from: schema linking, prompt tightening, and execution-guided repair.
- It includes a useful negative ablation: more agent freedom performed worse.
- It keeps scope honest: this is not a leaderboard submission and does not claim full BIRD coverage.

## Main Results

### Pipeline progression

| Method | EX | VSR | Takeaway |
| --- | ---: | ---: | --- |
| Day2 naive full schema | 26% | 86% | weak baseline |
| Day3 schema-linked + promptfix | 34% | 88% | schema retrieval helps semantics |
| Day5.5 strict execution repair | 36% | 100% | best overall pipeline |

- EX improves from `26%` to `34%` to `36%`.
- Going from `26%` to `36%` is a `38.5%` relative EX improvement over the naive baseline.
- VSR improves from `88%` to `100%` once strict repair is added.

### Agent ablation

| Agent setting | EX | VSR | Interpretation |
| --- | ---: | ---: | --- |
| Controlled execution-repair agent | 32% | 98% | stable graph, near the main pipeline |
| Autonomous ReAct agent | 12% | 88% | negative ablation |

- The controlled graph outperforms autonomous ReAct by `20` EX points: `32%` vs `12%`.
- The result suggests that in this repo, free-form tool calling added overhead and instability instead of better reasoning.

### Full snapshot

| Method | EX | VSR | Role |
| --- | ---: | ---: | --- |
| Day2 naive full schema | 26% | 86% | baseline |
| Day3 schema-linked + promptfix | 34% | 88% | improved baseline |
| Day5.5 strict execution repair | 36% | 100% | main result |
| Controlled agent trace implementation | 32% | 98% | controlled graph |
| Few-shot retrieval | 28% | 100% | negative ablation |
| DDL schema serialization | 28% | 82% | negative ablation |
| Autonomous ReAct agent | 12% | 88% | negative ablation |

## Method

The repository focuses on a simple, explicit pipeline:

1. retrieve task-relevant schema items with hybrid linking,
2. generate SQL from the linked schema context,
3. execute the SQL against SQLite,
4. repair only when execution fails,
5. return the final SQL candidate.

This is intentionally closer to an enterprise BI workflow than to a pure prompt-only benchmark setup.

## When Does Agency Hurt?

- Controlled execution-repair agent: `EX=32%`, `VSR=98%`
- Autonomous ReAct agent: `EX=12%`, `VSR=88%`
- Finding: free-form tool calling caused over-exploration and finalization instability.
- This supports using a controlled graph when task structure is known.

## Scope And Evaluation

All reported numbers in the repo are based on:

- BIRD dev `california_schools`
- a fixed `50`-sample subset
- saved result artifacts already committed under `results/`

Difficulty distribution of the subset:

- `simple = 15`
- `moderate = 16`
- `challenging = 19`

Metrics:

- `EX`: execution-match accuracy
- `VSR`: valid SQL rate

This project does not claim full BIRD benchmark coverage or leaderboard-valid results.

## Model Settings

- Core generation and repair pipelines use the repository DeepSeek client setting.
- In code, the core pipeline default is `deepseek-chat`; see [src/text2sql/llm.py](/Users/bytedance/text2sql_agent/src/text2sql/llm.py:10).
- Later agent experiments use DeepSeek function-calling settings; see [src/text2sql/agents/react_agent.py](/Users/bytedance/text2sql_agent/src/text2sql/agents/react_agent.py:21).
- This project does not claim cross-model robustness.

## Installation

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...
```

## Quick Start

Run the schema-linked pipeline:

```bash
PYTHONPATH=src python3 scripts/run_schema_linked.py --db-id california_schools --limit 50 --promptfix
```

Run strict execution repair on top of the saved schema-linked predictions:

```bash
PYTHONPATH=src python3 scripts/run_strict_repair.py
```

Useful additional entrypoints:

```bash
PYTHONPATH=src python3 scripts/run_controlled_agent.py --db-id california_schools --limit 50
PYTHONPATH=src python3 scripts/run_agent.py --db-id california_schools --limit 50
PYTHONPATH=src python3 scripts/compare_controlled_with_day5.py
```

Or use the included `Makefile`:

```bash
make compile
make eval-schema-linked
make eval-repair
make eval-controlled-agent
```

## Project Structure

- `src/text2sql/` contains the active package code.
- `scripts/` contains runnable CLI entrypoints.
- `results/` contains the frozen summaries used by the README.
- `docs/` contains writeups and analysis notes.
- `archive/` has been removed from `main` to keep the repository focused on the active package and headline results.

For the cleanup note, see [docs/archive_plan.md](/Users/bytedance/text2sql_agent/docs/archive_plan.md:1). The pre-cleanup history is preserved on the local branch `full-experiment-history`.

## Notes On Interpretation

- Day5.5 strict execution repair is the canonical main result.
- The controlled agent is a trace-oriented near-match implementation, not the canonical metric run.
- The autonomous ReAct agent is intentionally kept as a negative ablation.

For the longer write-up, see [docs/error_analysis.md](/Users/bytedance/text2sql_agent/docs/error_analysis.md:1).
