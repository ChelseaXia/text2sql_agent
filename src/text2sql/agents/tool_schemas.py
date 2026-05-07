"""Standard tool schemas for Text2SQL agent tool calls."""

from __future__ import annotations

import copy
import json
from typing import Any


JSON_TYPE_TO_PYTHON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


TOOL_SCHEMAS = [
    {
        "name": "retrieve_schema",
        "description": "Retrieve question-relevant database schema context for the current Text2SQL task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question to answer with SQL.",
                },
                "db_id": {
                    "type": "string",
                    "description": "Database identifier for the active sample.",
                },
            },
            "required": ["question", "db_id"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "selected_tables": {"type": "array", "items": {"type": "string"}},
                "retrieved_columns": {"type": "array", "items": {"type": "string"}},
                "linked_schema_text": {"type": "string"},
            },
            "required": ["selected_tables", "retrieved_columns", "linked_schema_text"],
        },
        "when_to_use": [
            "At the beginning of a task to identify relevant tables and columns.",
            "After execution failures that suggest the SQL used the wrong schema objects.",
        ],
        "failure_modes": [
            "Unknown or inactive db_id.",
            "Question is too vague to retrieve enough relevant schema context.",
            "Schema linker returns incomplete coverage for implicit joins or values.",
        ],
        "usage_notes": [
            "Use before drafting SQL unless schema context is already available.",
            "This tool exposes schema context only; it does not reveal gold_sql or evaluator correctness.",
        ],
    },
    {
        "name": "inspect_table",
        "description": "Inspect columns and metadata for one table in the active database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "db_id": {
                    "type": "string",
                    "description": "Database identifier for the active sample.",
                },
                "table_name": {
                    "type": "string",
                    "description": "Exact table name to inspect.",
                },
            },
            "required": ["db_id", "table_name"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "sample_values": {"type": "array"},
                            "is_pk": {"type": "boolean"},
                            "is_fk": {"type": "boolean"},
                            "fk_ref": {"type": ["string", "null"]},
                        },
                    },
                },
            },
            "required": ["table_name", "columns"],
        },
        "when_to_use": [
            "When column names, types, primary keys, or foreign keys are ambiguous.",
            "When deciding how a table can join to another table.",
        ],
        "failure_modes": [
            "Unknown table_name.",
            "The table exists but metadata descriptions are sparse.",
        ],
        "usage_notes": [
            "Use exact table names from retrieve_schema when possible.",
            "Prefer this over guessing join keys from column names alone.",
        ],
    },
    {
        "name": "sample_rows",
        "description": "Preview a small number of rows from a table to understand value formats and row shape.",
        "input_schema": {
            "type": "object",
            "properties": {
                "db_id": {
                    "type": "string",
                    "description": "Database identifier for the active sample.",
                },
                "table_name": {
                    "type": "string",
                    "description": "Exact table name to sample.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of rows to return. Defaults to 5.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["db_id", "table_name"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "success": {"type": "boolean"},
                "rows": {"type": "array"},
                "error": {"type": ["string", "null"]},
            },
            "required": ["table_name", "success", "rows", "error"],
        },
        "when_to_use": [
            "When the question depends on value formats, categorical labels, or denormalized columns.",
            "When generated SQL succeeds but the returned rows look suspicious.",
        ],
        "failure_modes": [
            "Unknown table_name.",
            "SQLite execution error while reading the table.",
            "Rows may be unrepresentative because only a small preview is returned.",
        ],
        "usage_notes": [
            "Keep n small; this is an exploration tool, not a full table scan.",
            "Do not use sampled rows as evaluator feedback or proof of correctness.",
        ],
    },
    {
        "name": "search_column_values",
        "description": "Search distinct values in a column using a text query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "db_id": {
                    "type": "string",
                    "description": "Database identifier for the active sample.",
                },
                "table_name": {
                    "type": "string",
                    "description": "Exact table name containing the column.",
                },
                "column_name": {
                    "type": "string",
                    "description": "Exact column name to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Text fragment or entity value to search for.",
                },
            },
            "required": ["db_id", "table_name", "column_name", "query"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "column_name": {"type": "string"},
                "success": {"type": "boolean"},
                "values": {"type": "array"},
                "error": {"type": ["string", "null"]},
            },
            "required": ["table_name", "column_name", "success", "values", "error"],
        },
        "when_to_use": [
            "When a question mentions an entity or literal value and exact database spelling is uncertain.",
            "When SQL returns empty rows and a WHERE literal may not match stored values.",
        ],
        "failure_modes": [
            "Unknown table_name or column_name.",
            "Column values are non-textual or stored in unexpected formats.",
            "The query string is too broad or too narrow to find useful candidates.",
        ],
        "usage_notes": [
            "Search candidate values before hard-coding uncertain string literals.",
            "Returned values are observations, not gold answers.",
        ],
    },
    {
        "name": "execute_sql",
        "description": "Execute a read-only SQL query against the active database and return a row preview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "db_id": {
                    "type": "string",
                    "description": "Database identifier for the active sample.",
                },
                "sql": {
                    "type": "string",
                    "description": "Read-only SQL query. Only SELECT or WITH statements are allowed.",
                },
            },
            "required": ["db_id", "sql"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "rows": {"type": "array"},
                "row_count": {"type": "integer"},
                "error": {"type": ["string", "null"]},
                "safety_blocked": {"type": "boolean"},
                "block_reason": {"type": ["string", "null"]},
            },
            "required": ["success", "rows", "row_count", "error", "safety_blocked", "block_reason"],
        },
        "when_to_use": [
            "After drafting or revising SQL to check whether it executes and what shape it returns.",
            "After a targeted repair to confirm syntax and runtime behavior.",
        ],
        "failure_modes": [
            "SQL safety check rejects non-read-only or multi-statement SQL.",
            "SQLite syntax, table, column, or type errors.",
            "Successful execution can still be semantically wrong.",
        ],
        "usage_notes": [
            "Only use read-only SELECT or WITH SQL.",
            "Execution observations do not include evaluator correctness or gold_sql comparison.",
        ],
    },
    {
        "name": "finish",
        "description": "Return the final SQL answer and stop tool use for the current task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "final_sql": {
                    "type": "string",
                    "description": "Final SQL answer for the natural language question.",
                },
            },
            "required": ["final_sql"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "final_sql": {"type": "string"},
            },
            "required": ["final_sql"],
        },
        "when_to_use": [
            "When the SQL is ready to submit.",
            "When the tool budget is exhausted and the best available SQL must be returned.",
        ],
        "failure_modes": [
            "final_sql is empty or not valid SQL.",
            "The final SQL may execute but still fail semantic correctness.",
        ],
        "usage_notes": [
            "Call finish exactly once at the end of a tool-selection loop.",
            "finish is not an evaluator and does not compare against gold_sql.",
        ],
    },
]


def _schemas_by_name() -> dict[str, dict[str, Any]]:
    return {schema["name"]: schema for schema in TOOL_SCHEMAS}


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """Return a copy of one tool schema by name, or None when missing."""
    schema = _schemas_by_name().get(name)
    if schema is None:
        return None
    return copy.deepcopy(schema)


def list_tool_schemas() -> list[dict[str, Any]]:
    """Return copies of all tool schemas in prompt order."""
    return copy.deepcopy(TOOL_SCHEMAS)


def _type_names(expected_type: str | list[str]) -> str:
    if isinstance(expected_type, list):
        return " or ".join(expected_type)
    return expected_type


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    python_types = JSON_TYPE_TO_PYTHON_TYPES.get(expected_type)
    if python_types is None:
        return True
    return isinstance(value, python_types)


def validate_tool_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate a tool call against its input schema.

    Returns:
        {"valid": True, "error": None} when valid, otherwise
        {"valid": False, "error": "..."} with a direct error message.
    """
    schema = _schemas_by_name().get(tool_name)
    if schema is None:
        valid_names = ", ".join(sorted(_schemas_by_name()))
        return {
            "valid": False,
            "error": f"Unknown tool '{tool_name}'. Available tools: {valid_names}.",
        }

    if not isinstance(args, dict):
        return {
            "valid": False,
            "error": f"Arguments for tool '{tool_name}' must be an object/dict.",
        }

    input_schema = schema["input_schema"]
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    missing_args = [arg_name for arg_name in required if arg_name not in args]
    if missing_args:
        return {
            "valid": False,
            "error": f"Tool '{tool_name}' missing required argument(s): {', '.join(missing_args)}.",
        }

    if input_schema.get("additionalProperties") is False:
        unknown_args = [arg_name for arg_name in args if arg_name not in properties]
        if unknown_args:
            return {
                "valid": False,
                "error": f"Tool '{tool_name}' received unknown argument(s): {', '.join(unknown_args)}.",
            }

    for arg_name, value in args.items():
        expected_type = properties.get(arg_name, {}).get("type")
        if expected_type is None:
            continue
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_matches_json_type(value, type_name) for type_name in expected_types):
            return {
                "valid": False,
                "error": (
                    f"Tool '{tool_name}' argument '{arg_name}' must be "
                    f"{_type_names(expected_type)}, got {type(value).__name__}."
                ),
            }

    return {"valid": True, "error": None}


def format_tools_for_prompt() -> str:
    """Format tool schemas as prompt-ready instructions for JSON tool calls."""
    prompt_payload = {
        "tool_call_format": {
            "thought": "Brief reasoning for why this tool is the next best action.",
            "tool": "One of the tool names listed below.",
            "args": "JSON object matching the selected tool input_schema.",
        },
        "tools": TOOL_SCHEMAS,
    }
    return (
        "You may call one Text2SQL tool at a time. Return only valid JSON with this shape:\n"
        '{\n  "thought": "...",\n  "tool": "...",\n  "args": {...}\n}\n\n'
        "Tool definitions:\n"
        f"{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}"
    )
