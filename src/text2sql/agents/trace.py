"""Trace formatting and export helpers for the agent pipeline."""

import json


def write_trace_examples(traces_path, output_path):
    if not traces_path.exists():
        return

    records = []
    with traces_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent Trace Examples",
        "",
        "Tool-calling traces on the local BIRD dev subset.",
        "",
    ]

    if not records:
        lines.append("No traces available yet.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    picked = []
    for record in records:
        if record.get("ex"):
            picked.append(record)
            break
    for record in records:
        if not record.get("pred_success"):
            picked.append(record)
            break
    if not picked:
        picked = records[:1]

    seen = set()
    for record in picked:
        if record["sample_id"] in seen:
            continue
        seen.add(record["sample_id"])
        lines.extend(
            [
                f"## Sample {record['sample_id']}",
                "",
                f"- Difficulty: `{record['difficulty']}`",
                f"- EX: `{record['ex']}`",
                f"- Final execution success: `{record['pred_success']}`",
                f"- Final SQL source: `{record.get('final_sql_source', '')}`",
                f"- Question: {record['question']}",
                "",
                "Final SQL:",
                "```sql",
                record.get("final_sql") or "-- no final sql --",
                "```",
                "",
                "Trace summary:",
            ]
        )
        for step in record.get("trace", [])[:8]:
            lines.append(f"- Step {step.get('step')}: `{step.get('action')}`")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
