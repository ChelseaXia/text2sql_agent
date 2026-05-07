# Execution-Result Self-Consistency Voting

This document tracks the standalone self-consistency voting pipeline for internal experiment work. Its public-facing method name is `execution-result self-consistency voting`.

## Scope

- It is a separate pipeline layered on top of schema-linked prompting and optional execution repair.
- It does not replace the existing baseline, schema-linked, strict-repair, or schema-format ablation pipelines.
- Internal artifact names may still use `Day7`, but `Day7` is only an internal alias and should not be used as the public method name in the main project narrative.

## Method

For each sample:

1. Retrieve linked schema context with the existing schema-linker.
2. Sample `K` SQL candidates with schema-linked prompting and prompt constraints.
3. Execute each candidate in SQLite.
4. If repair is enabled and a candidate fails execution, run one repair pass with the existing repair prompt and execute the repaired SQL.
5. Keep the final executable result for each candidate.
6. Hash executable result sets after canonical row normalization.
7. Cluster candidates by execution-result hash.
8. Select the largest execution-result cluster.

Tie-break rules between equal-size clusters:

1. prefer a cluster whose best representative is unrepaired,
2. then prefer shorter SQL,
3. then prefer smaller `candidate_id`.

## Outputs

Per-sample outputs store:

- full candidate list,
- execution / repair status,
- result-hash cluster assignment,
- selected-vote result,
- oracle correctness across `K` candidates.

Metrics report:

- `selected_EX`
- `selected_VSR`
- `oracle_EX_at_K`
- average valid-candidate count
- average cluster count
- average cluster confidence
- generation bottlenecks
- selection bottlenecks
- difficulty and database breakdowns

## Status

This pipeline is intended for internal evaluation only until its stability and benefit are established on larger manifests.

## Compact 50-sample Result

Execution-result self-consistency voting with K=10 achieves 38% selected EX and 100% VSR on the compact california_schools subset, compared with 36% EX and 100% VSR from the schema-linked execution-repair pipeline.

However, Oracle EX@10 is only 40%, just 2 percentage points above selected EX. Pairwise comparison shows that self-consistency fixes only one additional sample over the repair pipeline, while only one sample has Oracle=True but selected=False.

This indicates that the dominant bottleneck is not candidate selection. Instead, most failures are candidate generation failures: 30 out of 50 samples have Oracle=False, meaning no correct SQL appears among the 10 sampled candidates.

The challenging tier is the clearest failure mode. All 19 challenging samples have Oracle=False, so execution-result voting cannot recover them. This motivates the next stage: a suspicion-triggered iterative tool-using agent that can inspect tables, sample rows, search column values, and revise SQL based on observations.