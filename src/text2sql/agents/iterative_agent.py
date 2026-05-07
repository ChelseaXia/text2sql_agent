"""Suspicion-triggered iterative tool-using Text-to-SQL agent.

This module intentionally implements a controlled loop rather than free-form
ReAct: the program owns the state machine, suspicion checks, and tool limits.
"""

import argparse
import json
import re
from pathlib import Path

from text2sql.config import RESULTS_DIR
from text2sql.data import load_eval_manifest, resolve_eval_samples
from text2sql.db import run_sql, same_result
from text2sql.llm import call_llm
from text2sql.pipelines.naive import extract_sql
from text2sql.pipelines.schema_linked import build_schema_linker, retrieved_column_names
from text2sql.prompts.generation import build_linked_prompt
from text2sql.prompts.repair import build_repair_prompt
from text2sql.schema.items import quote_identifier
from text2sql.schema.linker import DEFAULT_LINKER_MODE, DEFAULT_TOP_K

METHOD_NAME = "suspicion_triggered_iterative_agent"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "iterative_agent"
DEFAULT_PREDICTIONS_PATH = DEFAULT_OUTPUT_DIR / "predictions.jsonl"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_TRACES_PATH = DEFAULT_OUTPUT_DIR / "traces.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "california_schools_50_manifest.jsonl"
DEFAULT_CHALLENGING_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "challenging_19_manifest.jsonl"
DEFAULT_MAX_STEPS = 5
DEFAULT_SAMPLE_ROWS = 5

LIST_INTENT_TERMS = {
    "list",
    "top",
    "lowest",
    "highest",
    "rank",
    "ranking",
    "group",
    "each",
    "all",
    "schools",
    "counties",
    "districts",
}
MULTI_TABLE_TERMS = {
    "sat",
    "frpm",
    "meal",
    "enrollment",
    "location",
    "charter",
    "opened",
    "closed",
    "county",
    "district",
    "school",
}
SQL_LITERAL_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
TEXT_TYPE_RE = re.compile(r"CHAR|CLOB|TEXT|VARCHAR|NCHAR|NVARCHAR", re.IGNORECASE)
ENTITY_HINT_RE = re.compile(
    r"\b(city|county|school|district|type|category|charter|continuation|office|education)\b",
    re.IGNORECASE,
)


def compact_rows(rows, limit=5):
    return [list(row) for row in rows[:limit]]


def observation_summary(observation, limit=600):
    text = json.dumps(observation, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def parse_json_object(raw_text):
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def table_names_from_sql(sql):
    names = []
    for pattern in (r"\bFROM\s+[`\"\[]?([A-Za-z_][\w]*)", r"\bJOIN\s+[`\"\[]?([A-Za-z_][\w]*)"):
        for name in re.findall(pattern, sql or "", flags=re.IGNORECASE):
            clean = name.strip().strip("`\"[]")
            if clean and clean not in names:
                names.append(clean)
    return names


def sql_literals(sql):
    literals = []
    for left, right in SQL_LITERAL_RE.findall(sql or ""):
        literal = left or right
        if literal and not literal.startswith("%") and not literal.endswith("%"):
            literals.append(literal)
    return literals


def has_list_intent(question):
    text = question.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in LIST_INTENT_TERMS)


def question_has_entity_filter(question):
    text = question or ""
    lowered = text.lower()
    filter_phrases = (" in ", " for ", " named ", " called ", " where ", " county", " district")
    has_capitalized_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
    return has_capitalized_entity or any(phrase in lowered for phrase in filter_phrases)


def likely_requires_multiple_tables(question, evidence, selected_tables):
    if len(selected_tables) < 2:
        return False
    combined = f"{question} {evidence}".lower()
    matched_terms = {term for term in MULTI_TABLE_TERMS if term in combined}
    return len(matched_terms) >= 3


def detect_suspicion(sample, sql, execution_result, selected_tables):
    if not execution_result["success"]:
        return "execution_error"

    rows = execution_result["rows"]
    question = sample["question"]
    evidence = sample.get("evidence", "")
    literals = sql_literals(sql)
    if (literals or WHERE_RE.search(sql or "")) and ENTITY_HINT_RE.search(f"{question} {evidence}"):
        return "literal_or_entity_may_not_match_database"

    if len(rows) == 0:
        return "empty_result"

    is_scalar = len(rows) == 1 and (not rows[0] or len(rows[0]) == 1)
    if is_scalar and has_list_intent(question):
        return "scalar_result_for_list_intent"

    if question_has_entity_filter(question) and not WHERE_RE.search(sql or ""):
        return "question_mentions_entity_or_value_but_sql_has_no_where"

    if likely_requires_multiple_tables(question, evidence, selected_tables) and not JOIN_RE.search(sql or ""):
        return "likely_requires_multiple_tables_but_sql_has_no_join"

    if literals and any(len(literal) >= 3 for literal in literals):
        return "sql_uses_literal_values_that_may_not_exist"

    return None


class WorkingMemory:
    def __init__(self):
        self.intent_plan = {}
        self.failed_sql = []
        self.execution_errors = []
        self.inspected_schema = {}
        self.sampled_rows = {}
        self.observed_values = []
        self.revised_hypotheses = []
        self.avoid_rules = []
        self.update_count = 0

    def set_intent_plan(self, intent_plan):
        self.intent_plan = intent_plan or {}
        self.update_count += 1

    def add_failed_sql(self, sql, error):
        self.failed_sql.append(sql or "")
        self.execution_errors.append(error or "Unknown error")
        self.update_count += 1

    def add_inspected_table(self, table_name, observation):
        self.inspected_schema[table_name] = observation
        self.update_count += 1

    def add_sampled_rows(self, table_name, observation):
        self.sampled_rows[table_name] = observation
        self.update_count += 1

    def add_observed_values(self, table_name, column_name, query, observation):
        self.observed_values.append(
            {
                "table_name": table_name,
                "column_name": column_name,
                "query": query,
                "observation": observation,
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

    def summary(self, limit=1400):
        sections = []
        if self.intent_plan:
            sections.append("Intent plan:\n" + observation_summary(self.intent_plan, 500))
        if self.failed_sql:
            sections.append("Previous failed SQL:\n" + "\n".join(self.failed_sql[-2:]))
        if self.execution_errors:
            sections.append("Execution errors:\n" + "\n".join(self.execution_errors[-3:]))
        if self.inspected_schema:
            sections.append("Inspected table schema:\n" + observation_summary(self.inspected_schema, 500))
        if self.sampled_rows:
            sections.append("Sampled rows:\n" + observation_summary(self.sampled_rows, 500))
        if self.observed_values:
            sections.append("Observed database values:\n" + observation_summary(self.observed_values[-5:], 500))
        if self.revised_hypotheses:
            sections.append("Revised hypothesis:\n" + "\n".join(self.revised_hypotheses[-3:]))
        if self.avoid_rules:
            sections.append("Avoid rules:\n" + "\n".join(self.avoid_rules[-5:]))
        text = "\n\n".join(sections) or "No working memory observations yet."
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

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
            "update_count": self.update_count,
        }


class IterativeAgentTools:
    def __init__(self, sample, schema_linker, top_k_schema=DEFAULT_TOP_K):
        self.sample = sample
        self.schema_linker = schema_linker
        self.top_k_schema = top_k_schema
        self._schema_items = None
        self._items_by_table = None

    @property
    def schema_items(self):
        if self._schema_items is None:
            from text2sql.schema.items import build_schema_items

            self._schema_items = build_schema_items(self.sample["db_path"])
        return self._schema_items

    @property
    def items_by_table(self):
        if self._items_by_table is None:
            grouped = {}
            for item in self.schema_items:
                grouped.setdefault(item["table"], []).append(item)
            self._items_by_table = grouped
        return self._items_by_table

    def retrieve_schema(self):
        linked_items, linked_schema_text = self.schema_linker.retrieve(
            self.sample["question"],
            self.sample.get("evidence", ""),
            top_k=self.top_k_schema,
        )
        selected_tables = []
        for item in linked_items:
            if item["table"] not in selected_tables:
                selected_tables.append(item["table"])
        return {
            "selected_tables": selected_tables,
            "retrieved_columns": retrieved_column_names(linked_items),
            "linked_schema_text": linked_schema_text,
        }

    def inspect_table(self, table_name):
        table_items = self._validate_table(table_name)
        return {
            "table_name": table_name,
            "columns": [
                {
                    "name": item["column"],
                    "type": item["type"],
                    "description": item.get("description") or "",
                    "sample_values": item.get("sample_values") or [],
                    "is_pk": bool(item.get("is_pk")),
                    "is_fk": bool(item.get("is_fk")),
                    "fk_ref": item.get("fk_ref"),
                }
                for item in table_items
            ],
        }

    def sample_rows(self, table_name, n=DEFAULT_SAMPLE_ROWS):
        self._validate_table(table_name)
        row_limit = max(1, min(int(n), 10))
        sql = f"SELECT * FROM {quote_identifier(table_name)} LIMIT {row_limit}"
        result = run_sql(sql, self.sample["db_path"])
        return {
            "table_name": table_name,
            "success": result["success"],
            "rows": compact_rows(result["rows"], row_limit),
            "row_count": len(result["rows"]),
            "error": result["error"],
        }

    def search_column_values(self, table_name, column_name, query):
        self._validate_column(table_name, column_name)
        sql = (
            f"SELECT DISTINCT {quote_identifier(column_name)} "
            f"FROM {quote_identifier(table_name)} "
            f"WHERE {quote_identifier(column_name)} IS NOT NULL "
            f"AND CAST({quote_identifier(column_name)} AS TEXT) LIKE ? "
            "LIMIT 20"
        )
        import sqlite3

        connection = sqlite3.connect(self.sample["db_path"])
        try:
            cursor = connection.execute(sql, (f"%{query}%",))
            values = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return {
                "table_name": table_name,
                "column_name": column_name,
                "query": query,
                "success": True,
                "values": values,
                "error": None,
            }
        except sqlite3.Error as exc:
            return {
                "table_name": table_name,
                "column_name": column_name,
                "query": query,
                "success": False,
                "values": [],
                "error": str(exc),
            }
        finally:
            connection.close()

    def execute_sql(self, sql):
        return run_sql(sql, self.sample["db_path"]) if sql else {
            "success": False,
            "rows": [],
            "error": "Empty SQL prediction",
        }

    def finish(self, final_sql):
        return {"final_sql": final_sql}

    def _validate_table(self, table_name):
        normalized = self.normalize_table_name(table_name)
        if normalized not in self.items_by_table:
            raise ValueError(f"Unknown table: {table_name}")
        return self.items_by_table[normalized]

    def _validate_column(self, table_name, column_name):
        normalized_column = self.normalize_column_name(table_name, column_name)
        for item in self._validate_table(table_name):
            if item["column"] == normalized_column:
                return item
        raise ValueError(f"Unknown column: {table_name}.{column_name}")

    def normalize_table_name(self, table_name):
        text = (table_name or "").strip().strip("`\"[]")
        if text in self.items_by_table:
            return text
        first_token = text.split()[0] if text else ""
        if first_token in self.items_by_table:
            return first_token
        lowered = text.lower()
        for known_table in self.items_by_table:
            if known_table.lower() == lowered:
                return known_table
        return text

    def normalize_column_name(self, table_name, column_name):
        text = (column_name or "").strip().strip("`\"[]")
        if "." in text:
            text = text.split(".")[-1].strip().strip("`\"[]")
        table = self.normalize_table_name(table_name)
        for item in self.items_by_table.get(table, []):
            if item["column"] == text or item["column"].lower() == text.lower():
                return item["column"]
        return text


def build_intent_plan_prompt(sample, linked_schema_text):
    evidence = sample.get("evidence") or "None"
    return f"""Create a lightweight intent plan for a Text-to-SQL task.

Use only the question, evidence, and provided schema. Do not assume gold SQL or evaluator feedback.

Return one JSON object with these keys:
{{
  "target_tables": ["..."],
  "required_joins": ["..."],
  "filters_to_verify": ["..."],
  "aggregation_or_ranking": "...",
  "expected_result_shape": "...",
  "likely_literal_values": ["..."]
}}

Schema:
{linked_schema_text}

Evidence:
{evidence}

Question:
{sample["question"]}
"""


def build_fallback_intent_plan(sample, selected_tables):
    literals = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b", sample["question"])
    question = sample["question"].lower()
    aggregation_or_ranking = "ranking_or_ordering" if has_list_intent(sample["question"]) else "unknown"
    if any(term in question for term in ("average", "avg", "count", "sum", "rate", "percent")):
        aggregation_or_ranking = "aggregation_or_computed_metric"
    return {
        "target_tables": selected_tables,
        "required_joins": ["verify join keys if more than one target table is used"],
        "filters_to_verify": literals,
        "aggregation_or_ranking": aggregation_or_ranking,
        "expected_result_shape": "list_or_table" if has_list_intent(sample["question"]) else "scalar_or_single_record",
        "likely_literal_values": literals,
    }


def generate_intent_plan(sample, linked_schema_text, selected_tables):
    raw_response = ""
    try:
        raw_response = call_llm(build_intent_plan_prompt(sample, linked_schema_text))
        plan = parse_json_object(raw_response)
        return plan, raw_response, None
    except Exception as exc:
        return build_fallback_intent_plan(sample, selected_tables), raw_response, str(exc)


def build_revision_prompt(sample, linked_schema_text, working_memory_summary, current_sql, suspicion_reason):
    evidence = sample.get("evidence") or "None"
    return f"""Given the relevant SQLite schema, evidence, question, current SQL, and within-turn observations, write a revised correct SQLite SQL query.

Rules:
- Return only one SQLite SQL query.
- Use exact table and column names from the schema.
- Wrap column names containing spaces, parentheses, hyphens, percent signs, or slashes with backticks.
- Do not use observations as ground truth answers; use them only to ground schema, joins, and literal values.
- Address the suspicion reason directly.
- If observed database values are present, prefer those exact values over guessed string literals.
- Follow avoid rules from working memory.

{linked_schema_text}

Evidence:
{evidence}

Question:
{sample["question"]}

Suspicion reason:
{suspicion_reason}

Current SQL:
{current_sql or "-- none --"}

Within-turn working memory:
{working_memory_summary}

Return only the revised SQL query."""


def build_tool_choice_prompt(sample, linked_schema_text, working_memory_summary, current_sql, suspicion_reason, selected_tables):
    tables = ", ".join(selected_tables)
    return f"""Choose exactly one exploration tool call for a controlled Text-to-SQL agent.

Allowed tools:
- inspect_table(table_name)
- sample_rows(table_name, n=5)
- search_column_values(table_name, column_name, query)

Do not choose execute_sql or finish. The controller will execute SQL separately.
Prefer:
- search_column_values when a literal value may not exist or may need exact spelling.
- inspect_table when joins, column ownership, or table semantics are uncertain.
- sample_rows when empty/scalar results suggest filters or data shape may be wrong.

Return a single JSON object with keys:
{{"action": "...", "table_name": "...", "column_name": "...", "query": "...", "hypothesis": "..."}}
Use null for irrelevant keys.

Relevant tables: {tables}

Schema:
{linked_schema_text}

Question:
{sample["question"]}

Current SQL:
{current_sql or "-- none --"}

Suspicion reason:
{suspicion_reason}

Working memory:
{working_memory_summary}
"""


def choose_fallback_tool(current_sql, suspicion_reason, selected_tables, retrieved_columns):
    tables_in_sql = table_names_from_sql(current_sql)
    table_name = (tables_in_sql or selected_tables or [""])[0]
    action = "inspect_table"
    column_name = None
    query = None

    literals = sql_literals(current_sql)
    if suspicion_reason == "sql_uses_literal_values_that_may_not_exist" and literals:
        for column_ref in retrieved_columns:
            if "." in column_ref:
                maybe_table, maybe_column = column_ref.split(".", 1)
                if maybe_table == table_name:
                    action = "search_column_values"
                    column_name = maybe_column
                    query = literals[0]
                    break
    elif suspicion_reason in {"empty_result", "scalar_result_for_list_intent"} and table_name:
        action = "sample_rows"
    return {
        "action": action,
        "table_name": table_name,
        "column_name": column_name,
        "query": query,
        "hypothesis": f"Fallback exploration for {suspicion_reason}.",
    }


def choose_exploration_tool(sample, linked_schema_text, working_memory, current_sql, suspicion_reason, selected_tables, retrieved_columns):
    prompt = build_tool_choice_prompt(
        sample=sample,
        linked_schema_text=linked_schema_text,
        working_memory_summary=working_memory.summary(),
        current_sql=current_sql,
        suspicion_reason=suspicion_reason,
        selected_tables=selected_tables,
    )
    raw_response = ""
    try:
        raw_response = call_llm(prompt)
        choice = parse_json_object(raw_response)
    except Exception:
        choice = choose_fallback_tool(current_sql, suspicion_reason, selected_tables, retrieved_columns)
        choice["raw_response"] = raw_response
        return choice

    allowed_actions = {"inspect_table", "sample_rows", "search_column_values"}
    if choice.get("action") not in allowed_actions:
        choice = choose_fallback_tool(current_sql, suspicion_reason, selected_tables, retrieved_columns)
    choice["raw_response"] = raw_response
    return choice


def execute_exploration_tool(tools, choice):
    action = choice.get("action")
    table_name = tools.normalize_table_name(choice.get("table_name"))
    choice["table_name"] = table_name
    try:
        if action == "inspect_table":
            return tools.inspect_table(table_name)
        if action == "sample_rows":
            return tools.sample_rows(table_name, n=DEFAULT_SAMPLE_ROWS)
        if action == "search_column_values":
            column_name = tools.normalize_column_name(table_name, choice.get("column_name"))
            choice["column_name"] = column_name
            return tools.search_column_values(table_name, column_name, choice.get("query") or "")
        raise ValueError(f"Unsupported exploration action: {action}")
    except Exception as exc:
        fallback_table = next(iter(tools.items_by_table))
        choice["action"] = "inspect_table"
        choice["table_name"] = fallback_table
        choice["column_name"] = None
        choice["query"] = None
        return {
            "success": False,
            "error": str(exc),
            "fallback_action": "inspect_table",
            "fallback_observation": tools.inspect_table(fallback_table),
        }


def is_text_column(item):
    column = item["column"].lower()
    description = (item.get("description") or "").lower()
    return (
        bool(TEXT_TYPE_RE.search(item.get("type") or ""))
        or any(term in column for term in ("name", "type", "city", "county", "district", "school", "option", "status"))
        or any(term in description for term in ("name", "type", "city", "county", "district", "school", "category"))
    )


def extract_question_entity_values(sample, intent_plan):
    values = []
    for value in sql_literals(" ".join(f"'{item}'" for item in intent_plan.get("likely_literal_values", []))):
        if value not in values:
            values.append(value)
    for match in re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b", sample["question"]):
        if match not in {"For", "Please", "Include"} and match not in values:
            values.append(match)
    for phrase in ("Continuation School", "Charter School", "Fresno County Office of Education"):
        if phrase.lower() in sample["question"].lower() and phrase not in values:
            values.append(phrase)
    return [value for value in values if len(value) >= 3][:6]


def literal_grounding_values(sample, sql, intent_plan):
    values = []
    for value in sql_literals(sql):
        if len(value) >= 3 and value not in values:
            values.append(value)
    if not values and WHERE_RE.search(sql or ""):
        for value in extract_question_entity_values(sample, intent_plan):
            if value not in values:
                values.append(value)
    return values[:6]


def should_ground_literals(sample, sql):
    return bool((sql_literals(sql) or WHERE_RE.search(sql or "")) and ENTITY_HINT_RE.search(sample["question"]))


def search_relevant_column_values(tools, selected_tables, searched_values):
    observations = []
    affected_columns = []
    candidate_items = [
        item
        for item in tools.schema_items
        if item["table"] in selected_tables and is_text_column(item)
    ]
    for value in searched_values:
        per_value_matches = []
        for item in candidate_items[:40]:
            observation = tools.search_column_values(item["table"], item["column"], value)
            affected_columns.append(f"{item['table']}.{item['column']}")
            if observation["values"]:
                per_value_matches.append(observation)
        observations.append(
            {
                "searched_value": value,
                "matched_values": per_value_matches[:8],
                "affected_columns": affected_columns[-40:],
            }
        )
    return observations


def make_trace_event(sample, step, action, tool_input, observation, suspicion_reason, working_memory, current_sql, final_sql=None, is_correct=None):
    return {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "step": step,
        "action": action,
        "tool_input": tool_input,
        "observation": observation,
        "suspicion_reason": suspicion_reason,
        "scratchpad_summary": working_memory.summary(),
        "working_memory_summary": working_memory.compact(),
        "current_sql": current_sql,
        "final_sql": final_sql,
        "is_correct": is_correct,
    }


def generate_initial_sql(sample, linked_schema_text):
    raw_response = call_llm(build_linked_prompt(sample, linked_schema_text))
    return extract_sql(raw_response), raw_response


def generate_targeted_repair_sql(sample, linked_schema_text, previous_sql, execution_error):
    raw_response = call_llm(
        build_repair_prompt(
            question=sample["question"],
            evidence=sample.get("evidence", ""),
            linked_schema_text=linked_schema_text,
            previous_sql=previous_sql,
            sqlite_error=execution_error,
        )
    )
    return extract_sql(raw_response), raw_response


def generate_revised_sql(sample, linked_schema_text, working_memory, current_sql, suspicion_reason):
    prompt = build_revision_prompt(
        sample=sample,
        linked_schema_text=linked_schema_text,
        working_memory_summary=working_memory.summary(),
        current_sql=current_sql,
        suspicion_reason=suspicion_reason,
    )
    raw_response = call_llm(prompt)
    return extract_sql(raw_response), raw_response


def run_one_sample(sample, schema_linker, max_steps=DEFAULT_MAX_STEPS, top_k_schema=DEFAULT_TOP_K):
    tools = IterativeAgentTools(sample, schema_linker, top_k_schema=top_k_schema)
    working_memory = WorkingMemory()
    trace = []
    stats = {
        "tool_call_count": 0,
        "execute_call_count": 0,
        "suspicious_trigger_count": 0,
        "exploration_count": 0,
        "repair_count": 0,
        "repair_attempt_count": 0,
        "repair_success_count": 0,
        "search_column_values_count": 0,
        "intent_plan_success_count": 0,
        "working_memory_update_count": 0,
        "finish_count": 0,
    }
    step_id = 1

    schema_observation = tools.retrieve_schema()
    stats["tool_call_count"] += 1
    selected_tables = schema_observation["selected_tables"]
    retrieved_columns = schema_observation["retrieved_columns"]
    linked_schema_text = schema_observation["linked_schema_text"]
    trace.append(
        make_trace_event(
            sample,
            step_id,
            "retrieve_schema",
            {"question": sample["question"], "db_id": sample["db_id"], "top_k_schema": top_k_schema},
            schema_observation,
            None,
            working_memory,
            "",
        )
    )
    step_id += 1

    intent_plan, intent_raw_response, intent_error = generate_intent_plan(sample, linked_schema_text, selected_tables)
    working_memory.set_intent_plan(intent_plan)
    stats["intent_plan_success_count"] += 1 if intent_plan else 0
    trace.append(
        make_trace_event(
            sample,
            step_id,
            "intent_plan",
            {"question": sample["question"], "db_id": sample["db_id"]},
            {"intent_plan": intent_plan, "raw_response": intent_raw_response, "error": intent_error},
            None,
            working_memory,
            "",
        )
    )
    step_id += 1

    current_sql = ""
    final_result = {"success": False, "rows": [], "error": "No SQL generated"}
    llm_error = None
    explored_suspicion_reasons = set()
    try:
        current_sql, raw_response = generate_initial_sql(sample, linked_schema_text)
    except Exception as exc:
        llm_error = str(exc)
        working_memory.add_failed_sql("", llm_error)
        raw_response = ""
    trace.append(
        make_trace_event(
            sample,
            step_id,
            "generate_sql",
            {"question": sample["question"], "evidence": sample.get("evidence", "")},
            {"sql": current_sql, "raw_response": raw_response, "llm_error": llm_error},
            None,
            working_memory,
            current_sql,
        )
    )
    step_id += 1

    for attempt_index in range(1, max_steps + 1):
        final_result = tools.execute_sql(current_sql)
        stats["tool_call_count"] += 1
        stats["execute_call_count"] += 1
        trace.append(
            make_trace_event(
                sample,
                step_id,
                "execute_sql",
                {"db_id": sample["db_id"], "sql": current_sql},
                {
                    "success": final_result["success"],
                    "row_count": len(final_result["rows"]),
                    "rows_preview": compact_rows(final_result["rows"]),
                    "error": final_result["error"],
                },
                None,
                working_memory,
                current_sql,
            )
        )
        step_id += 1

        suspicion_reason = detect_suspicion(sample, current_sql, final_result, selected_tables)

        if suspicion_reason == "execution_error":
            stats["suspicious_trigger_count"] += 1
            stats["repair_count"] += 1
            stats["repair_attempt_count"] += 1
            execution_error = final_result["error"] or "Unknown SQLite error"
            original_sql = current_sql
            working_memory.add_failed_sql(original_sql, execution_error)

            repaired_sql = ""
            repair_raw_response = ""
            repair_error = None
            repair_result = {"success": False, "rows": [], "error": "Repair was not generated"}
            try:
                repaired_sql, repair_raw_response = generate_targeted_repair_sql(
                    sample,
                    linked_schema_text,
                    original_sql,
                    execution_error,
                )
                repair_result = tools.execute_sql(repaired_sql)
                stats["tool_call_count"] += 1
                stats["execute_call_count"] += 1
            except Exception as exc:
                repair_error = str(exc)

            repair_success = bool(repair_result["success"])
            stats["repair_success_count"] += 1 if repair_success else 0
            if repair_success:
                current_sql = repaired_sql
                final_result = repair_result
                working_memory.add_hypothesis("Targeted execution repair produced an executable SQL candidate.")
            else:
                working_memory.add_failed_sql(repaired_sql or original_sql, repair_result.get("error") or repair_error)
                working_memory.add_avoid_rule("Do not finish after a failed execution repair; re-plan before the next candidate.")
            trace.append(
                make_trace_event(
                    sample,
                    step_id,
                    "repair_sql",
                    {
                        "original_sql": original_sql,
                        "execution_error": execution_error,
                    },
                    {
                        "original_sql": original_sql,
                        "repaired_sql": repaired_sql,
                        "execution_error": execution_error,
                        "repair_success": repair_success,
                        "repair_error": repair_error,
                        "repair_raw_response": repair_raw_response,
                        "repaired_execute": {
                            "success": repair_result["success"],
                            "row_count": len(repair_result["rows"]),
                            "rows_preview": compact_rows(repair_result["rows"]),
                            "error": repair_result["error"],
                        },
                    },
                    "execution_error",
                    working_memory,
                    current_sql,
                )
            )
            step_id += 1

            if not repair_success:
                if attempt_index >= max_steps:
                    break
                try:
                    revised_sql, raw_response = generate_revised_sql(
                        sample,
                        linked_schema_text,
                        working_memory,
                        original_sql,
                        "execution_error_after_failed_repair",
                    )
                    current_sql = revised_sql
                    llm_error = None
                except Exception as exc:
                    llm_error = str(exc)
                    working_memory.add_failed_sql(original_sql, llm_error)
                    raw_response = ""
                trace.append(
                    make_trace_event(
                        sample,
                        step_id,
                        "regenerate_sql",
                        {"suspicion_reason": "execution_error_after_failed_repair"},
                        {"sql": current_sql, "raw_response": raw_response, "llm_error": llm_error},
                        "execution_error_after_failed_repair",
                        working_memory,
                        current_sql,
                    )
                )
                step_id += 1
                continue

            suspicion_reason = detect_suspicion(sample, current_sql, final_result, selected_tables)

        if suspicion_reason is None:
            if not current_sql or not final_result["success"]:
                break
            finish_observation = tools.finish(current_sql)
            stats["tool_call_count"] += 1
            stats["finish_count"] += 1
            trace.append(
                make_trace_event(
                    sample,
                    step_id,
                    "finish",
                    {"final_sql": current_sql},
                    finish_observation,
                    None,
                    working_memory,
                    current_sql,
                    final_sql=current_sql,
                )
            )
            break

        if final_result["success"] and suspicion_reason in explored_suspicion_reasons:
            if not current_sql or not final_result["success"]:
                break
            finish_observation = tools.finish(current_sql)
            stats["tool_call_count"] += 1
            stats["finish_count"] += 1
            trace.append(
                make_trace_event(
                    sample,
                    step_id,
                    "finish",
                    {"final_sql": current_sql, "repeated_suspicion_reason": suspicion_reason},
                    {
                        **finish_observation,
                        "reason": "same_suspicion_already_explored_for_this_sample",
                    },
                    suspicion_reason,
                    working_memory,
                    current_sql,
                    final_sql=current_sql,
                )
            )
            break

        stats["suspicious_trigger_count"] += 1

        if attempt_index >= max_steps:
            break

        if final_result["success"]:
            explored_suspicion_reasons.add(suspicion_reason)
            if should_ground_literals(sample, current_sql):
                searched_values = literal_grounding_values(sample, current_sql, working_memory.intent_plan)
                grounding_observations = search_relevant_column_values(tools, selected_tables, searched_values)
                stats["tool_call_count"] += len(grounding_observations)
                stats["exploration_count"] += 1 if grounding_observations else 0
                stats["search_column_values_count"] += len(grounding_observations)
                matched_values = []
                affected_columns = []
                for observation in grounding_observations:
                    matched_values.extend(observation["matched_values"])
                    affected_columns.extend(observation["affected_columns"])
                    working_memory.add_observed_values(
                        "multiple",
                        "multiple",
                        observation["searched_value"],
                        observation,
                    )
                if matched_values:
                    working_memory.add_hypothesis("Use exact observed database values for entity and literal filters.")
                trace.append(
                    make_trace_event(
                        sample,
                        step_id,
                        "search_column_values",
                        {
                            "searched_value": searched_values,
                            "selected_tables": selected_tables,
                        },
                        {
                            "searched_value": searched_values,
                            "matched_values": matched_values,
                            "affected_columns": sorted(set(affected_columns))[:80],
                            "observations": grounding_observations,
                        },
                        "literal_or_entity_may_not_match_database",
                        working_memory,
                        current_sql,
                    )
                )
                step_id += 1
            else:
                choice = choose_exploration_tool(
                    sample=sample,
                    linked_schema_text=linked_schema_text,
                    working_memory=working_memory,
                    current_sql=current_sql,
                    suspicion_reason=suspicion_reason,
                    selected_tables=selected_tables,
                    retrieved_columns=retrieved_columns,
                )
                tool_observation = execute_exploration_tool(tools, choice)
                stats["tool_call_count"] += 1
                stats["exploration_count"] += 1
                if choice.get("action") == "inspect_table":
                    working_memory.add_inspected_table(choice.get("table_name"), tool_observation)
                elif choice.get("action") == "sample_rows":
                    working_memory.add_sampled_rows(choice.get("table_name"), tool_observation)
                elif choice.get("action") == "search_column_values":
                    stats["search_column_values_count"] += 1
                    working_memory.add_observed_values(
                        choice.get("table_name"),
                        choice.get("column_name"),
                        choice.get("query"),
                        tool_observation,
                    )
                working_memory.add_hypothesis(choice.get("hypothesis"))
                trace.append(
                    make_trace_event(
                        sample,
                        step_id,
                        choice.get("action"),
                        choice,
                        tool_observation,
                        suspicion_reason,
                        working_memory,
                        current_sql,
                    )
                )
                step_id += 1

        try:
            revised_sql, raw_response = generate_revised_sql(
                sample,
                linked_schema_text,
                working_memory,
                current_sql,
                suspicion_reason,
            )
            current_sql = revised_sql
            llm_error = None
        except Exception as exc:
            llm_error = str(exc)
            working_memory.add_failed_sql(current_sql, llm_error)
            raw_response = ""
        trace.append(
            make_trace_event(
                sample,
                step_id,
                "regenerate_sql",
                {"suspicion_reason": suspicion_reason},
                {"sql": current_sql, "raw_response": raw_response, "llm_error": llm_error},
                suspicion_reason,
                working_memory,
                current_sql,
            )
        )
        step_id += 1

    final_sql = current_sql
    gold_result = run_sql(sample["gold_sql"], sample["db_path"])
    ex = bool(final_result["success"] and gold_result["success"] and same_result(final_result["rows"], gold_result["rows"]))
    pred_success = bool(final_result["success"])
    if not stats["finish_count"]:
        trace.append(
            make_trace_event(
                sample,
                step_id,
                "no_finish",
                {"final_sql": final_sql},
                {
                    "final_sql": final_sql,
                    "reason": "finish_guard_blocked_empty_or_unexecuted_or_failed_sql",
                    "last_execute_success": final_result["success"],
                },
                None,
                working_memory,
                final_sql,
                final_sql=final_sql,
                is_correct=ex,
            )
        )

    for event in trace:
        if event["action"] == "finish":
            event["final_sql"] = final_sql
            event["is_correct"] = ex

    stats["working_memory_update_count"] = working_memory.update_count

    record = {
        "sample_id": sample["sample_id"],
        "db_id": sample["db_id"],
        "db_path": sample["db_path"],
        "difficulty": sample["difficulty"],
        "question": sample["question"],
        "evidence": sample.get("evidence", ""),
        "gold_sql": sample["gold_sql"],
        "method": METHOD_NAME,
        "schema_linker_mode": DEFAULT_LINKER_MODE,
        "retrieved_columns": retrieved_columns,
        "selected_tables": selected_tables,
        "pred_sql": final_sql,
        "predicted_sql": final_sql,
        "final_sql": final_sql,
        "pred_success": pred_success,
        "is_executable": pred_success,
        "gold_success": gold_result["success"],
        "error": None if pred_success else final_result["error"],
        "failure_reason": None if pred_success else final_result["error"],
        "pred_row_count": len(final_result["rows"]),
        "gold_row_count": len(gold_result["rows"]),
        "pred_rows_preview": compact_rows(final_result["rows"]),
        "gold_rows_preview": compact_rows(gold_result["rows"]),
        "ex": ex,
        "is_correct": ex,
        "final_sql_source": "finish_tool" if stats["finish_count"] else "no_finish",
        "max_steps": max_steps,
        "trace": trace,
        **stats,
    }
    return record, trace


def empty_bucket():
    return {
        "sample_count": 0,
        "ex_count": 0,
        "vsr_count": 0,
        "finish_count": 0,
        "tool_call_count": 0,
        "execute_call_count": 0,
        "suspicious_trigger_count": 0,
        "exploration_count": 0,
        "repair_count": 0,
        "repair_attempt_count": 0,
        "repair_success_count": 0,
        "search_column_values_count": 0,
        "intent_plan_success_count": 0,
        "working_memory_update_count": 0,
    }


def finalize_bucket(bucket):
    n = bucket["sample_count"]
    return {
        "sample_count": n,
        "EX": bucket["ex_count"] / n if n else 0.0,
        "VSR": bucket["vsr_count"] / n if n else 0.0,
        "finish_rate": bucket["finish_count"] / n if n else 0.0,
        "avg_tool_calls": bucket["tool_call_count"] / n if n else 0.0,
        "avg_execute_calls": bucket["execute_call_count"] / n if n else 0.0,
        "suspicious_trigger_count": bucket["suspicious_trigger_count"],
        "exploration_count": bucket["exploration_count"],
        "repair_count": bucket["repair_count"],
        "repair_attempt_count": bucket["repair_attempt_count"],
        "repair_success_count": bucket["repair_success_count"],
        "search_column_values_count": bucket["search_column_values_count"],
        "intent_plan_success_count": bucket["intent_plan_success_count"],
        "working_memory_update_count": bucket["working_memory_update_count"],
    }


def compute_iterative_metrics(records):
    overall = empty_bucket()
    by_difficulty = {}
    for record in records:
        for bucket in (overall, by_difficulty.setdefault(record["difficulty"], empty_bucket())):
            bucket["sample_count"] += 1
            bucket["ex_count"] += 1 if record["ex"] else 0
            bucket["vsr_count"] += 1 if record["pred_success"] else 0
            bucket["finish_count"] += 1 if record["final_sql_source"] == "finish_tool" else 0
            bucket["tool_call_count"] += record["tool_call_count"]
            bucket["execute_call_count"] += record["execute_call_count"]
            bucket["suspicious_trigger_count"] += record["suspicious_trigger_count"]
            bucket["exploration_count"] += record["exploration_count"]
            bucket["repair_count"] += record["repair_count"]
            bucket["repair_attempt_count"] += record["repair_attempt_count"]
            bucket["repair_success_count"] += record["repair_success_count"]
            bucket["search_column_values_count"] += record["search_column_values_count"]
            bucket["intent_plan_success_count"] += record["intent_plan_success_count"]
            bucket["working_memory_update_count"] += record["working_memory_update_count"]

    metrics = finalize_bucket(overall)
    metrics["by_difficulty"] = {name: finalize_bucket(bucket) for name, bucket in sorted(by_difficulty.items())}
    return metrics


def write_records_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_trace_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            for event in record["trace"]:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_iterative_agent(
    samples,
    predictions_path=DEFAULT_PREDICTIONS_PATH,
    metrics_path=DEFAULT_METRICS_PATH,
    traces_path=DEFAULT_TRACES_PATH,
    max_steps=DEFAULT_MAX_STEPS,
    top_k_schema=DEFAULT_TOP_K,
    embedding_model_path=None,
):
    records = []
    linker_cache = {}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path.parent.mkdir(parents=True, exist_ok=True)

    with predictions_path.open("w", encoding="utf-8") as pred_file, traces_path.open("w", encoding="utf-8") as trace_file:
        for index, sample in enumerate(samples, start=1):
            db_path = sample["db_path"]
            if db_path not in linker_cache:
                linker_cache[db_path] = build_schema_linker(
                    db_path=db_path,
                    top_k=top_k_schema,
                    schema_linker_mode=DEFAULT_LINKER_MODE,
                    embedding_model_path=embedding_model_path,
                )
            record, trace = run_one_sample(
                sample,
                schema_linker=linker_cache[db_path],
                max_steps=max_steps,
                top_k_schema=top_k_schema,
            )
            records.append(record)
            pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_file.flush()
            for event in trace:
                trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            trace_file.flush()
            print(
                f"{index}: sample_id={record['sample_id']} "
                f"pred_success={record['pred_success']} ex={record['ex']} "
                f"suspicious={record['suspicious_trigger_count']} explorations={record['exploration_count']}"
            )

    metrics = compute_iterative_metrics(records)
    metrics["max_steps"] = max_steps
    metrics["schema_linker_mode"] = DEFAULT_LINKER_MODE
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "traces_path": str(traces_path),
    }


def write_default_manifests(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_manifest = output_dir / "california_schools_50_manifest.jsonl"
    challenging_manifest = output_dir / "challenging_19_manifest.jsonl"

    source_manifest = RESULTS_DIR / "self_consistency" / "california_schools_50_manifest.jsonl"
    source_predictions = RESULTS_DIR / "self_consistency" / "california_schools_50_predictions.jsonl"
    if source_manifest.exists():
        compact_rows = load_eval_manifest(source_manifest)
    else:
        compact_rows = resolve_eval_samples(limit=50, db_id="california_schools")

    write_records_jsonl(compact_rows, compact_manifest)

    challenging_ids = None
    if source_predictions.exists():
        prediction_rows = [json.loads(line) for line in source_predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        challenging_ids = [
            row["sample_id"]
            for row in prediction_rows
            if row.get("difficulty") == "challenging" and not row.get("oracle_correct")
        ]
    if challenging_ids:
        id_set = set(challenging_ids)
        challenging_rows = [row for row in compact_rows if row["sample_id"] in id_set]
    else:
        challenging_rows = [row for row in compact_rows if row.get("difficulty") == "challenging"]

    write_records_jsonl(challenging_rows, challenging_manifest)
    return compact_manifest, challenging_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Run the suspicion-triggered iterative Text-to-SQL agent.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--db-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--top-k-schema", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_PATH)
    parser.add_argument("--write-default-manifests", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.write_default_manifests:
        compact_manifest, challenging_manifest = write_default_manifests()
        print(
            json.dumps(
                {
                    "compact_manifest": str(compact_manifest),
                    "challenging_manifest": str(challenging_manifest),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.manifest is None and args.limit is None:
            return

    if args.manifest is None and args.db_id is None:
        args.db_id = "california_schools"
    if args.manifest is None and args.limit is None:
        args.limit = 50

    samples = resolve_eval_samples(limit=args.limit, db_id=args.db_id, manifest_path=args.manifest)
    summary = run_iterative_agent(
        samples=samples,
        predictions_path=args.predictions_output,
        metrics_path=args.metrics_output,
        traces_path=args.traces_output,
        max_steps=args.max_steps,
        top_k_schema=args.top_k_schema,
        embedding_model_path=args.embedding_model_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
