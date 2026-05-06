# Expanded Evaluation

## Overview

This document summarizes the expanded local evaluation on the full BIRD dev frozen manifest: `1534` samples evaluated with the same fixed sample set across all compared methods.

The expanded comparison is a local full-dev workflow for controlled method analysis. It is not a leaderboard submission workflow, and the results here should not be presented as leaderboard numbers.

## External Method Names

For public-facing documentation, the compared methods are:

- Full-schema direct prompting baseline
- Schema-linked prompting + prompt constraints
- Schema-linked execution-repair pipeline
- DDL-style schema serialization ablation

## Internal Experiment Aliases

Historical file names, script names, and result artifacts still use internal experiment aliases:

- `Day2` = Full-schema direct prompting baseline
- `Day3` = Schema-linked prompting + prompt constraints
- `Day5.5` = Schema-linked execution-repair pipeline
- `Day6` = DDL-style schema serialization ablation

These aliases are retained for experiment continuity, but they are not the primary outward-facing method names.

## Full Frozen Manifest Results

All methods below were evaluated on the same full BIRD dev frozen manifest with `1534` samples.

| Method | Samples | EX | VSR | Main Role |
| --- | ---: | ---: | ---: | --- |
| Full-schema direct prompting | 1534 | 51.11% | 95.63% | baseline |
| Schema-linked prompting + constraints | 1534 | 55.41% | 97.39% | schema grounding |
| Schema-linked execution-repair pipeline | 1534 | 56.45% | 99.80% | best overall |
| DDL-style schema serialization | 1534 | 55.93% | 97.78% | schema format ablation |

## Key Takeaways

- Schema linking improves EX by `+4.30` percentage points over full-schema prompting, from `51.11%` to `55.41%`, which is a `+8.42%` relative EX gain.
- Execution-guided repair further improves EX by `+1.04` percentage points over schema-linked prompting, from `55.41%` to `56.45%`, which is a `+1.88%` relative EX gain.
- The execution-repair stage also lifts VSR by `+2.41` percentage points, from `97.39%` to `99.80%`.
- DDL-style schema serialization reaches `55.93%` EX on the full manifest, slightly above schema-linked prompting alone (`55.41%`), so it is not negative on the full evaluation.
- DDL-style serialization still does not beat the schema-linked execution-repair pipeline (`56.45%` EX, `99.80%` VSR).
- On the compact `50`-sample subset, this ablation was negative; on the full frozen manifest, it is marginally positive.

## Difficulty Analysis

For the schema-linked execution-repair pipeline, the full-manifest EX breakdown is:

- `simple`: `66.05%`
- `moderate`: `54.18%`
- `challenging`: `25.11%`

The pattern is clear: simple and moderate questions are much stronger than challenging ones, and challenging-tier semantic planning remains the main bottleneck after execution repair.

## Database Heterogeneity

Performance is not uniform across databases.

- Strongest databases: `superhero`, `student_club`, `codebase_community`
- Weakest databases: `financial`, `california_schools`

This spread suggests that better execution robustness alone is not enough. Cross-database variation is still strongly shaped by schema semantics and planning difficulty.

## Why The Frozen Manifest Matters

All compared methods read the same frozen manifest so that:

- every method sees the same `sample_id` set,
- database mix stays fixed,
- difficulty mix stays fixed,
- pairwise gains reflect method changes rather than sample drift.

## Reproduction

Build a manifest:

```bash
PYTHONPATH=src python3 scripts/build_eval_manifest.py \
  --output results/expanded/full_manifest.jsonl
```

Build a deterministic smaller manifest if needed:

```bash
PYTHONPATH=src python3 scripts/build_eval_manifest.py \
  --limit 100 \
  --seed 42 \
  --output results/expanded/manifest_100.jsonl
```

Run the expanded core evaluation on one frozen manifest:

```bash
bash scripts/run_expanded_core_eval.sh results/expanded/manifest_100.jsonl
```

## Schema Linker Setup

The default schema-linked evaluation path keeps the existing `hybrid` schema-linking setup, which requires the dense SentenceTransformer model.

If the dense encoder is not already available locally, configure one of these before running expanded evaluation:

- pre-download `sentence-transformers/all-MiniLM-L6-v2` into the local Hugging Face cache, or
- pass a local model directory with `--embedding-model-path /path/to/all-MiniLM-L6-v2`

For smoke-only debugging, you may explicitly use lexical retrieval instead:

```bash
PYTHONPATH=src python3 scripts/run_schema_linked.py \
  --manifest results/expanded/manifest_100.jsonl \
  --promptfix \
  --schema-linker-mode bm25
```

`bm25` is only an explicit smoke/debug option. The default remains `hybrid` so the main evaluation method definition does not change.

Summarize the outputs:

```bash
PYTHONPATH=src python3 scripts/summarize_expanded_eval.py
```

The summary is written to:

```text
results/expanded/expanded_core_eval_summary.json
```
