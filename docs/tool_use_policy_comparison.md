# Autonomous vs Rule-Based Tool-Use Policy Comparison

## Scope

This Day 2.5 note analyzes existing stratified-300 outputs only. It does not rerun experiments, call the LLM API, change agent code, or change the README.

Inputs:

- `results/iterative_agent/autonomous_stratified300_predictions.jsonl`
- `results/iterative_agent/autonomous_stratified300_traces.jsonl`
- `results/iterative_agent/v1_1_stratified300_predictions.jsonl`
- `results/iterative_agent/v1_1_stratified300_traces.jsonl`
- `results/iterative_agent/stratified_300_manifest.jsonl`

Generated analysis outputs:

- `results/iterative_agent/autonomous_failure_taxonomy.json`
- `results/iterative_agent/autonomous_failure_cases.csv`

## Overall Metrics

| Policy | EX | VSR | Finish Rate | Avg Tool Calls | Avg Execute Calls |
|---|---:|---:|---:|---:|---:|
| autonomous `llm_decided` | 50.00% | 94.33% | 94.33% | 3.79 | 1.26 |
| rule_based v1.1 | 51.33% | 100.00% | 100.00% | 4.72 | 1.81 |

Outcome overlap on 300 samples:

- autonomous only correct: 19
- rule_based only correct: 23
- both correct: 131
- both wrong: 127

Autonomous tool selection is feasible and reaches 50.00% EX on stratified 300. It is close to the bounded rule-based policy on EX, but it is less stable on execution validity and finish behavior.

## Autonomous Failure Counters

| Counter | Count |
|---|---:|
| `over_exploration_count` | 24 |
| `finish_without_successful_execute_count` | 20 |
| `probe_as_final_count` | 24 |
| `budget_exceeded_count` | 24 |
| `premature_finish_count` | 23 |
| `validation_error_count` | 4 |
| `json_parse_error_count` | 1 |

The important pattern is that JSON and tool-schema validity were not the main problem. There was only 1 JSON parse error and 4 validation errors across 300 samples. The larger instability came from finish control and budget control: premature finish, no successful final execution, probe/final confusion, and budget exhaustion.

## Failure Taxonomy

`autonomous_failure_cases.csv` contains 25 selected cases. At least 10 autonomous failure cases were manually/semiautomatically classified using trace evidence. Selected failure labels cover:

| Failure Type | Selected Cases |
|---|---:|
| `tool_argument_error` | 1 |
| `missing_repair` | 1 |
| `premature_finish` | 3 |
| `budget_exceeded` | 2 |
| `probe_as_final` | 2 |
| `wrong_tool_sequence` | 3 |
| `semantic_generation_failure` | 2 |
| `over_exploration` | 1 |

Primary classification across all autonomous-wrong samples:

| Primary Failure Type | Count |
|---|---:|
| `semantic_generation_failure` | 103 |
| `premature_finish` | 17 |
| `wrong_tool_sequence` | 10 |
| `probe_as_final` | 10 |
| `budget_exceeded` | 7 |
| `tool_argument_error` | 2 |
| `missing_repair` | 1 |

Some counters co-occur. For example, budget-exceeded rows often also have over-exploration evidence, so the primary taxonomy undercounts over-exploration as a root behavior. The raw co-occurring row counts are:

- over-exploration rows: 24
- probe-as-final rows: 23
- budget-exceeded rows: 24

## Representative Cases

| Sample | Group | Failure Type | Trace Evidence |
|---:|---|---|---|
| 406 | autonomous wrong / rule_based wrong | `tool_argument_error` | Tool sequence reached `retrieve_schema -> execute_sql -> execute_sql -> finish`; trace records a validation/tool-selection error from an oversized model request. |
| 205 | finish failed | `missing_repair` | Multiple `execute_sql` attempts failed; finish guard blocked because no SQL executed successfully. |
| 71 | finish failed | `premature_finish` | The model explored with `sample_rows`, `inspect_table`, and value search, then attempted finish without any successful `execute_sql`. |
| 48 | autonomous wrong / rule_based correct | `over_exploration` | Value-search budget was exhausted before a successful final execution; finish guard blocked with no successful execute. |
| 18 | budget exceeded | `budget_exceeded` | The run hit `max_execute_calls_exceeded` and fell back to last successful SQL. |
| 50 | probe-as-final | `probe_as_final` | Budget fallback used the last successful SQL even though it was not promoted as a successful final candidate. |
| 411 | autonomous wrong / rule_based wrong | `wrong_tool_sequence` | The sequence mixed retrieve, execute, value search, repeated table inspection, and finish without converging cleanly. |
| 46 | autonomous wrong / rule_based wrong | `semantic_generation_failure` | Tool use was valid and finish succeeded, but the final SQL was semantically wrong. |
| 451 | autonomous wrong / rule_based correct | `wrong_tool_sequence` | Rule-based solved the sample; autonomous trace shows a less disciplined tool sequence before finalization. |
| 598 | autonomous wrong / rule_based correct | `budget_exceeded` | Autonomous exhausted budget where rule-based completed successfully. |
| 30 | autonomous correct / rule_based wrong | `autonomous_success_reference` | Positive contrast: autonomous produced a correct final SQL where rule-based did not. |
| 513 | challenging autonomous correct | `challenging_success_reference` | Positive challenging case showing autonomous selection can work on harder examples. |

Full per-case fields are in `autonomous_failure_cases.csv`:

- `sample_id`
- `db_id`
- `difficulty`
- `question`
- `autonomous_correct`
- `rule_based_correct`
- `failure_type`
- `evidence_from_trace`
- `suggested_fix`

## Interpretation

Autonomous tool selection is viable: it reaches 50.00% EX on stratified 300 and has 19 cases where it is correct while rule_based v1.1 is wrong.

The bounded rule-based workflow is still more stable: it reaches 51.33% EX, 100.00% VSR, and 100.00% finish rate. Autonomous loses most visibly on finish stability and budget control, not on JSON formatting or schema validation.

The practical gap is policy discipline. The autonomous policy needs stronger constraints around:

- reserving budget for final candidate execution and finish;
- preventing finish before successful final-candidate execution;
- separating probe SQL from final SQL;
- forcing repair after failed execution;
- limiting repeated exploration once enough schema/value evidence has been collected.

This is an acceptable result for the experiment: autonomous mode exposes a real tradeoff between flexible LLM-selected tool use and the reliability of a bounded rule-based controller.
