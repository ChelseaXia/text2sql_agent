# ReAct Agent Negative Ablation

The current autonomous ReAct-style agent is kept as a negative ablation rather than a production direction.

## Current Result

- EX = `12%`
- VSR = `88%`
- finish_rate = `24%`
- fallback_used_count = `32`
- execute_sql_attempt_count = `241`
- avg_execute_call_count = `4.82`

## Conclusion

Fully autonomous ReAct-style tool calling did not outperform the controlled execution-repair pipeline.

## Failure Attribution

- Over-exploration: the agent spends too many steps inspecting tables, sampling rows, and executing intermediate SQL before converging.
- Unstable finalization: even when the agent finds executable SQL, it does not reliably convert that into a stable final answer.
- High fallback dependency: many final outputs depend on post-hoc fallback behavior rather than explicit finalization.
- Executable but semantically weak SQL: the agent often reaches runnable SQL that is still far from the gold semantics.

## Typical Failure Cases

### Over-Exploration Case: `sample_id=20`

Question: high-school statistics in Amador County.

Key trace steps:

- Step 1 `retrieve_schema`: selected multiple relevant tables before any SQL generation.
- Step 2 `inspect_table` x3: inspected several tables up front instead of quickly testing one end-to-end query.
- Step 3 `sample_rows` x3: spent additional steps sampling rows from multiple tables.
- Steps 4-6 `execute_sql` x5: executed many intermediate SQL candidates, but still ended with `final_sql_source=last_successful_execute`.
- Outcome: executable SQL was found, but the path was long and still semantically incorrect.

### Finalization Instability Case: `sample_id=34`

Question: free rate for students aged 5-17 at the school run by Kacey Gibson.

Key trace steps:

- Step 2 `search_column_values` x2: searched for the administrator and school identifier.
- Steps 3-5 `execute_sql`: executed lookup SQL and then a rate-computation SQL successfully.
- Step 6 `finish`: rejected because the submitted final SQL had not been executed in exactly that form.
- Step 7 `execute_sql`: executed another SQL variant.
- Step 8 `finish`: eventually succeeded, showing that finalization was fragile even on a solvable example.

### Probe / Lookup SQL Risk Case: `sample_id=0`

Question: highest free-meal-rate school in Alameda County with characteristics and deviation from county average.

Key trace steps:

- Step 3 `sample_rows` x3: sampled multiple tables before composing a complete answer.
- Steps 4-8 `execute_sql`: executed ranking and lookup style queries incrementally.
- The last successful SQL was a school-level lookup query rather than a full answer query.
- Outcome: `final_sql_source=rejected_probe_sql`, which correctly prevented a probe query from being treated as the final answer.
