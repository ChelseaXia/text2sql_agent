"""Working and episodic memory containers for Text2SQL agents.

The memory objects intentionally store only runtime observations. They do not
store gold SQL, correctness labels, evaluator feedback, or oracle answers.
"""

from __future__ import annotations

import json
import re
from typing import Any


FORBIDDEN_KEYS = {
    "gold_sql",
    "is_correct",
    "ex",
    "exact_match",
    "exec_match",
    "evaluator_feedback",
    "oracle_correct",
    "standard_answer",
}


def _safe_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _safe_copy(item)
            for key, item in value.items()
            if key not in FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_safe_copy(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_copy(item) for item in value]
    return value


def _compact_text(value: Any, limit: int = 600) -> str:
    text = json.dumps(_safe_copy(value), ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text or "") if len(token) > 2}


class WorkingMemory:
    """Per-question memory used inside one Text2SQL attempt."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.intent_plan = {}
        self.failed_sql = []
        self.execution_errors = []
        self.inspected_schema = {}
        self.sampled_rows = {}
        self.observed_values = []
        self.revised_hypotheses = []
        self.avoid_rules = []
        self.tool_observation_history = []
        self.update_count = 0
        self.hit_count = 0

    def set_intent_plan(self, intent_plan):
        self.intent_plan = _safe_copy(intent_plan or {})
        self.update_count += 1

    def add_failed_sql(self, sql, error):
        self.failed_sql.append(sql or "")
        self.execution_errors.append(error or "Unknown error")
        self.update_count += 1

    def add_inspected_table(self, table_name, observation):
        if table_name:
            self.inspected_schema[table_name] = _safe_copy(observation)
            self.update_count += 1

    def add_sampled_rows(self, table_name, observation):
        if table_name:
            self.sampled_rows[table_name] = _safe_copy(observation)
            self.update_count += 1

    def add_observed_values(self, table_name, column_name, query, observation):
        self.observed_values.append(
            {
                "table_name": table_name,
                "column_name": column_name,
                "query": query,
                "observation": _safe_copy(observation),
            }
        )
        self.update_count += 1

    def add_hypothesis(self, hypothesis):
        if hypothesis:
            self.revised_hypotheses.append(hypothesis)
            self.update_count += 1

    def add_avoid_rule(self, rule):
        if rule:
            self.avoid_rules.append(rule)
            self.update_count += 1

    def write_observation(self, event):
        event = _safe_copy(event or {})
        action = event.get("action") or event.get("llm_selected_tool")
        tool_input = event.get("tool_input") or event.get("tool_args") or {}
        observation = event.get("observation") or {}
        self.tool_observation_history.append(
            {
                "action": action,
                "tool_input": tool_input,
                "observation": observation,
            }
        )
        if action == "inspect_table":
            self.add_inspected_table(tool_input.get("table_name") or observation.get("table_name"), observation)
        elif action == "sample_rows":
            self.add_sampled_rows(tool_input.get("table_name") or observation.get("table_name"), observation)
        elif action == "search_column_values":
            self.add_observed_values(
                tool_input.get("table_name") or observation.get("table_name"),
                tool_input.get("column_name") or observation.get("column_name"),
                tool_input.get("query") or observation.get("query"),
                observation,
            )
        elif action == "execute_sql" and not observation.get("success", True):
            self.add_failed_sql(tool_input.get("sql"), observation.get("error"))
        else:
            self.update_count += 1

    def read_relevant(self, query=None):
        if query is None:
            self.hit_count += 1 if self.tool_observation_history else 0
            return self.export_json()
        query_tokens = _tokens(query)
        matches = []
        for event in self.tool_observation_history:
            event_text = _compact_text(event, 1000)
            if query_tokens & _tokens(event_text):
                matches.append(event)
        if matches:
            self.hit_count += 1
        return matches

    def summarize_for_prompt(self, max_tokens=None, max_items=5):
        sections = []
        if self.intent_plan:
            sections.append("Intent plan:\n" + _compact_text(self.intent_plan, 500))
        if self.failed_sql:
            sections.append("Previous failed SQL:\n" + "\n".join(self.failed_sql[-2:]))
        if self.execution_errors:
            sections.append("Execution errors:\n" + "\n".join(self.execution_errors[-3:]))
        if self.inspected_schema:
            sections.append("Inspected table schema:\n" + _compact_text(self.inspected_schema, 600))
        if self.sampled_rows:
            sections.append("Sampled rows:\n" + _compact_text(self.sampled_rows, 600))
        if self.observed_values:
            sections.append("Observed database values:\n" + _compact_text(self.observed_values[-max_items:], 600))
        if self.revised_hypotheses:
            sections.append("Revised hypothesis:\n" + "\n".join(self.revised_hypotheses[-max_items:]))
        if self.avoid_rules:
            sections.append("Avoid rules:\n" + "\n".join(self.avoid_rules[-max_items:]))
        text = "\n\n".join(sections) or "No working memory observations yet."
        limit = max_tokens or 1400
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    def summary(self, limit=1400):
        return self.summarize_for_prompt(max_tokens=limit)

    def compact(self):
        return {
            "intent_plan": self.intent_plan,
            "failed_sql_count": len(self.failed_sql),
            "latest_execution_error": self.execution_errors[-1] if self.execution_errors else None,
            "inspected_schema_tables": list(self.inspected_schema.keys()),
            "sampled_row_tables": list(self.sampled_rows.keys()),
            "observed_value_count": len(self.observed_values),
            "latest_observed_values": self.observed_values[-3:],
            "revised_hypotheses": self.revised_hypotheses[-3:],
            "avoid_rules": self.avoid_rules[-5:],
            "tool_observation_count": len(self.tool_observation_history),
            "update_count": self.update_count,
        }

    def export_json(self):
        return _safe_copy(
            {
                "intent_plan": self.intent_plan,
                "failed_sql": self.failed_sql,
                "execution_errors": self.execution_errors,
                "inspected_schema": self.inspected_schema,
                "sampled_rows": self.sampled_rows,
                "observed_values": self.observed_values,
                "revised_hypotheses": self.revised_hypotheses,
                "avoid_rules": self.avoid_rules,
                "tool_observation_history": self.tool_observation_history,
                "update_count": self.update_count,
                "hit_count": self.hit_count,
            }
        )


class NullWorkingMemory(WorkingMemory):
    """No-op working memory for ablations."""

    def set_intent_plan(self, intent_plan):
        self.intent_plan = {}

    def add_failed_sql(self, sql, error):
        return None

    def add_inspected_table(self, table_name, observation):
        return None

    def add_sampled_rows(self, table_name, observation):
        return None

    def add_observed_values(self, table_name, column_name, query, observation):
        return None

    def add_hypothesis(self, hypothesis):
        return None

    def add_avoid_rule(self, rule):
        return None

    def write_observation(self, event):
        return None

    def read_relevant(self, query=None):
        return {}

    def summarize_for_prompt(self, max_tokens=None, max_items=5):
        return "Memory disabled."

    def compact(self):
        return {
            "enabled": False,
            "update_count": 0,
            "tool_observation_count": 0,
        }


class EpisodicMemory:
    """Per-database memory shared only within one db_id session."""

    def __init__(self, db_id):
        self.db_id = db_id
        self.reset()

    def reset(self):
        self.schema_cache = {}
        self.value_cache = {}
        self.join_hints = []
        self.sampled_rows = {}
        self.tool_observation_history = []
        self.access_count = 0
        self.hit_count = 0
        self.write_count = 0

    def _check_db(self, event):
        event_db_id = (event or {}).get("db_id") or (event or {}).get("sample", {}).get("db_id")
        if event_db_id and event_db_id != self.db_id:
            raise ValueError(f"EpisodicMemory for {self.db_id} cannot store event from {event_db_id}.")

    def write_observation(self, event):
        event = _safe_copy(event or {})
        self._check_db(event)
        action = event.get("action") or event.get("llm_selected_tool")
        tool_input = event.get("tool_input") or event.get("tool_args") or {}
        observation = event.get("observation") or {}
        record = {
            "sample_id": event.get("sample_id"),
            "db_id": self.db_id,
            "action": action,
            "tool_input": tool_input,
            "observation": observation,
        }
        self.tool_observation_history.append(record)
        if action == "inspect_table":
            table_name = tool_input.get("table_name") or observation.get("table_name")
            if table_name:
                self.schema_cache[table_name] = observation
        elif action == "sample_rows":
            table_name = tool_input.get("table_name") or observation.get("table_name")
            if table_name:
                self.sampled_rows[table_name] = observation
        elif action == "search_column_values":
            table_name = tool_input.get("table_name") or observation.get("table_name")
            column_name = tool_input.get("column_name") or observation.get("column_name")
            query = tool_input.get("query") or observation.get("query") or ""
            if table_name and column_name:
                key = f"{table_name}.{column_name}:{query}".lower()
                self.value_cache[key] = observation
        elif action == "retrieve_schema":
            for column in observation.get("retrieved_columns", [])[:20]:
                if "." in column:
                    table_name = column.split(".", 1)[0]
                    self.schema_cache.setdefault(table_name, {"table_name": table_name, "source": "retrieve_schema"})
        self.write_count += 1

    def read_relevant(self, question, intent_plan=None):
        self.access_count += 1
        query_text = f"{question or ''} {_compact_text(intent_plan or {}, 1000)}"
        query_tokens = _tokens(query_text)
        hits = {
            "schema_cache": {},
            "value_cache": {},
            "sampled_rows": {},
            "join_hints": self.join_hints[-5:],
        }
        for table_name, observation in self.schema_cache.items():
            if table_name.lower() in query_text.lower() or query_tokens & _tokens(_compact_text(observation, 1000)):
                hits["schema_cache"][table_name] = observation
        for key, observation in self.value_cache.items():
            if query_tokens & _tokens(f"{key} {_compact_text(observation, 1000)}"):
                hits["value_cache"][key] = observation
        for table_name, observation in self.sampled_rows.items():
            if table_name.lower() in query_text.lower() or query_tokens & _tokens(_compact_text(observation, 1000)):
                hits["sampled_rows"][table_name] = observation
        has_hit = bool(hits["schema_cache"] or hits["value_cache"] or hits["sampled_rows"] or hits["join_hints"])
        if has_hit:
            self.hit_count += 1
        return hits

    def summarize_for_prompt(self, question, intent_plan=None, max_items=5):
        relevant = self.read_relevant(question, intent_plan=intent_plan)
        sections = []
        if relevant["schema_cache"]:
            sections.append("Prior inspected schema:\n" + _compact_text(dict(list(relevant["schema_cache"].items())[:max_items]), 700))
        if relevant["sampled_rows"]:
            sections.append("Prior sampled rows:\n" + _compact_text(dict(list(relevant["sampled_rows"].items())[:max_items]), 700))
        if relevant["value_cache"]:
            sections.append("Prior observed values:\n" + _compact_text(dict(list(relevant["value_cache"].items())[:max_items]), 700))
        if relevant["join_hints"]:
            sections.append("Prior join hints:\n" + _compact_text(relevant["join_hints"][-max_items:], 400))
        return "\n\n".join(sections) or "No relevant episodic memory for this database yet."

    def export_json(self):
        return _safe_copy(
            {
                "db_id": self.db_id,
                "schema_cache": self.schema_cache,
                "value_cache": self.value_cache,
                "join_hints": self.join_hints,
                "sampled_rows": self.sampled_rows,
                "tool_observation_history": self.tool_observation_history,
                "access_count": self.access_count,
                "hit_count": self.hit_count,
                "write_count": self.write_count,
            }
        )

    def get_stats(self):
        return {
            "db_id": self.db_id,
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "write_count": self.write_count,
            "hit_rate": self.hit_count / self.access_count if self.access_count else 0.0,
            "schema_cache_size": len(self.schema_cache),
            "value_cache_size": len(self.value_cache),
            "sampled_rows_cache_size": len(self.sampled_rows),
            "tool_observation_count": len(self.tool_observation_history),
        }
