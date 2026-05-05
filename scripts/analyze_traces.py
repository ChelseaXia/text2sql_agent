"""Summarize saved react agent traces without re-running the agent."""

import json
from pathlib import Path

from text2sql.config import PROJECT_ROOT, RESULTS_DIR

DEFAULT_TRACES_PATH = RESULTS_DIR / "react_agent_traces.jsonl"
DEFAULT_STATS_PATH = RESULTS_DIR / "react_agent_trace_stats.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "docs" / "agent_trace_summary.md"
TOOL_NAMES = [
    "retrieve_schema",
    "inspect_table",
    "sample_rows",
    "search_column_values",
    "execute_sql",
    "finish",
]


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_trace_stats(records):
    total_samples = len(records)
    total_steps = 0
    tool_call_count_by_tool = {tool: 0 for tool in TOOL_NAMES}
    execute_sql_attempt_count = 0
    first_execute_success_count = 0
    repaired_success_count = 0
    never_executable_count = 0
    strict_finish_count = 0
    strict_pred_success_count = 0
    strict_ex_count = 0
    relaxed_pred_success_count = 0
    relaxed_ex_count = 0
    safety_blocked_count = 0
    total_repair_rounds = 0
    sample_rows_usage_count = 0
    search_column_values_usage_count = 0
    no_finish_count = 0
    finish_rejected_count = 0
    fallback_finish_count = 0
    executed_but_not_finished_count = 0

    for record in records:
        trace = record.get("trace", [])
        total_steps += len(trace)
        strict_pred_success_count += int(bool(record.get("strict_pred_success", record.get("pred_success"))))
        strict_ex_count += int(bool(record.get("strict_ex", record.get("ex"))))
        relaxed_pred_success_count += int(bool(record.get("relaxed_pred_success", record.get("pred_success"))))
        relaxed_ex_count += int(bool(record.get("relaxed_ex", record.get("ex"))))

        if record.get("strict_final_sql_source", record.get("final_sql_source")) == "finish_tool":
            strict_finish_count += 1
        elif record.get("strict_final_sql_source", record.get("final_sql_source")) == "no_finish":
            no_finish_count += 1
            if record.get("has_successful_execute"):
                executed_but_not_finished_count += 1
        if record.get("relaxed_final_sql_source") == "last_successful_execute":
            fallback_finish_count += 1

        execute_observations = []
        for step in trace:
            action = step.get("action")
            if action in tool_call_count_by_tool:
                tool_call_count_by_tool[action] += 1
            if action == "sample_rows":
                sample_rows_usage_count += 1
            if action == "search_column_values":
                search_column_values_usage_count += 1
            if action == "execute_sql":
                execute_sql_attempt_count += 1
                observation = step.get("observation") or {}
                execute_observations.append(observation)
                if observation.get("safety_blocked"):
                    safety_blocked_count += 1
            if step.get("finish_rejected"):
                finish_rejected_count += 1

        if execute_observations:
            if execute_observations[0].get("success"):
                first_execute_success_count += 1
            if any(ob.get("success") for ob in execute_observations):
                repair_rounds = max(len(execute_observations) - 1, 0)
                total_repair_rounds += repair_rounds
                if not execute_observations[0].get("success") and record.get("pred_success"):
                    repaired_success_count += 1
            else:
                never_executable_count += 1
                total_repair_rounds += max(len(execute_observations) - 1, 0)
        else:
            never_executable_count += 1

    return {
        "total_samples": total_samples,
        "avg_steps": total_steps / total_samples if total_samples else 0.0,
        "tool_call_count_by_tool": tool_call_count_by_tool,
        "execute_sql_attempt_count": execute_sql_attempt_count,
        "first_execute_success_count": first_execute_success_count,
        "repaired_success_count": repaired_success_count,
        "never_executable_count": never_executable_count,
        "finish_rate": strict_finish_count / total_samples if total_samples else 0.0,
        "strict_finish_count": strict_finish_count,
        "finish_count": strict_finish_count,
        "no_finish_count": no_finish_count,
        "fallback_finish_count": fallback_finish_count,
        "finish_rejected_count": finish_rejected_count,
        "executed_but_not_finished_count": executed_but_not_finished_count,
        "strict_VSR": strict_pred_success_count / total_samples if total_samples else 0.0,
        "strict_EX": strict_ex_count / total_samples if total_samples else 0.0,
        "relaxed_VSR": relaxed_pred_success_count / total_samples if total_samples else 0.0,
        "relaxed_EX": relaxed_ex_count / total_samples if total_samples else 0.0,
        "VSR": strict_pred_success_count / total_samples if total_samples else 0.0,
        "EX": strict_ex_count / total_samples if total_samples else 0.0,
        "search_column_values_usage_count": search_column_values_usage_count,
        "sample_rows_usage_count": sample_rows_usage_count,
        "safety_blocked_count": safety_blocked_count,
        "average_repair_rounds": total_repair_rounds / total_samples if total_samples else 0.0,
    }


def write_summary(summary_path, stats):
    lines = [
        "# Agent Trace Summary",
        "",
        "Offline summary of `results/react_agent_traces.jsonl`.",
        "",
        f"- Total samples: `{stats['total_samples']}`",
        f"- Avg steps: `{stats['avg_steps']:.2f}`",
        f"- Execute SQL attempt count: `{stats['execute_sql_attempt_count']}`",
        f"- First execute success count: `{stats['first_execute_success_count']}`",
        f"- Repaired success count: `{stats['repaired_success_count']}`",
        f"- Never executable count: `{stats['never_executable_count']}`",
        f"- Finish rate: `{stats['finish_rate']:.2f}`",
        f"- strict VSR: `{stats['strict_VSR']:.2f}`",
        f"- strict EX: `{stats['strict_EX']:.2f}`",
        f"- relaxed VSR: `{stats['relaxed_VSR']:.2f}`",
        f"- relaxed EX: `{stats['relaxed_EX']:.2f}`",
        f"- search_column_values usage count: `{stats['search_column_values_usage_count']}`",
        f"- sample_rows usage count: `{stats['sample_rows_usage_count']}`",
        f"- safety_blocked count: `{stats['safety_blocked_count']}`",
        f"- Average repair rounds: `{stats['average_repair_rounds']:.2f}`",
        f"- Strict finish count: `{stats['strict_finish_count']}`",
        f"- No finish count: `{stats['no_finish_count']}`",
        f"- Fallback finish count: `{stats['fallback_finish_count']}`",
        f"- Finish rejected count: `{stats['finish_rejected_count']}`",
        f"- Executed but not finished count: `{stats['executed_but_not_finished_count']}`",
        "",
        "## Tool Counts",
        "",
    ]

    for tool_name, count in stats["tool_call_count_by_tool"].items():
        lines.append(f"- {tool_name}: `{count}`")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    records = load_jsonl(DEFAULT_TRACES_PATH)
    stats = compute_trace_stats(records)
    DEFAULT_STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(DEFAULT_SUMMARY_PATH, stats)
    print(json.dumps({"stats_path": str(DEFAULT_STATS_PATH), "summary_path": str(DEFAULT_SUMMARY_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
