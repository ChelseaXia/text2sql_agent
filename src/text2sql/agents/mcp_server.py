"""MCP server exposing safe Text2SQL database tools.

This module is intentionally independent from the agent control loops. It
does not call an LLM and does not use gold SQL or evaluator correctness.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # Keep py_compile/import diagnostics usable without MCP installed.
    FastMCP = None

from text2sql.agents.tool_schemas import get_tool_schema
from text2sql.agents.tools import check_sql_safety
from text2sql.data import get_db_path
from text2sql.schema.items import build_schema_items, quote_identifier
from text2sql.schema.linker import DEFAULT_TOP_K, SchemaLinker


LOGGER = logging.getLogger(__name__)
DEFAULT_ROW_LIMIT = 20
DEFAULT_SCHEMA_LINKER_MODE = os.getenv("TEXT2SQL_MCP_SCHEMA_LINKER_MODE", "bm25")
DEFAULT_TOP_K_SCHEMA = int(os.getenv("TEXT2SQL_MCP_TOP_K_SCHEMA", str(DEFAULT_TOP_K)))


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _error(message: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **payload}


def _jsonable_rows(rows) -> list[list[Any]]:
    return [list(row) for row in rows]


def _db_path_for_id(db_id: str) -> Path:
    db_path = get_db_path(db_id)
    if not db_path.exists():
        raise FileNotFoundError(f"Unknown db_id or missing database file: {db_id} ({db_path})")
    return db_path


@lru_cache(maxsize=16)
def _schema_items_for_db(db_id: str) -> tuple[dict[str, Any], ...]:
    return tuple(build_schema_items(str(_db_path_for_id(db_id))))


@lru_cache(maxsize=8)
def _schema_linker_for_db(db_id: str, mode: str = DEFAULT_SCHEMA_LINKER_MODE) -> SchemaLinker:
    return SchemaLinker(str(_db_path_for_id(db_id)), mode=mode)


def _items_by_table(db_id: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in _schema_items_for_db(db_id):
        grouped.setdefault(item["table"], []).append(dict(item))
    return grouped


def _normalize_table_name(db_id: str, table_name: str) -> str:
    text = (table_name or "").strip().strip("`\"[]")
    tables = _items_by_table(db_id)
    if text in tables:
        return text
    first_token = text.split()[0] if text else ""
    if first_token in tables:
        return first_token
    lowered = text.lower()
    for known_table in tables:
        if known_table.lower() == lowered:
            return known_table
    raise ValueError(f"Unknown table for db_id={db_id}: {table_name}")


def _normalize_column_name(db_id: str, table_name: str, column_name: str) -> str:
    text = (column_name or "").strip().strip("`\"[]")
    if "." in text:
        text = text.split(".")[-1].strip().strip("`\"[]")
    table = _normalize_table_name(db_id, table_name)
    for item in _items_by_table(db_id).get(table, []):
        if item["column"] == text or item["column"].lower() == text.lower():
            return item["column"]
    raise ValueError(f"Unknown column for db_id={db_id}: {table_name}.{column_name}")


def _selected_tables_from_items(items: list[dict[str, Any]]) -> list[str]:
    tables = []
    for item in items:
        if item["table"] not in tables:
            tables.append(item["table"])
    return tables


def _schema_metadata(tool_name: str) -> dict[str, Any]:
    schema = get_tool_schema(tool_name) or {}
    return {
        "tool_schema_name": schema.get("name", tool_name),
        "tool_schema_description": schema.get("description", ""),
    }


def retrieve_schema_impl(question: str, db_id: str) -> dict[str, Any]:
    try:
        linker = _schema_linker_for_db(db_id)
        linked_items, linked_schema_text = linker.retrieve(question, "", top_k=DEFAULT_TOP_K_SCHEMA)
        linked_items = [dict(item) for item in linked_items]
        return _ok(
            **_schema_metadata("retrieve_schema"),
            db_id=db_id,
            selected_tables=_selected_tables_from_items(linked_items),
            retrieved_columns=[f"{item['table']}.{item['column']}" for item in linked_items],
            linked_schema_text=linked_schema_text,
            schema_linker_mode=DEFAULT_SCHEMA_LINKER_MODE,
        )
    except Exception as exc:
        LOGGER.exception("retrieve_schema failed")
        return _error(str(exc), db_id=db_id)


def inspect_table_impl(db_id: str, table_name: str) -> dict[str, Any]:
    try:
        normalized_table = _normalize_table_name(db_id, table_name)
        columns = [
            {
                "name": item["column"],
                "type": item["type"],
                "description": item.get("description") or "",
                "sample_values": item.get("sample_values") or [],
                "is_pk": bool(item.get("is_pk")),
                "is_fk": bool(item.get("is_fk")),
                "fk_ref": item.get("fk_ref"),
            }
            for item in _items_by_table(db_id)[normalized_table]
        ]
        return _ok(
            **_schema_metadata("inspect_table"),
            db_id=db_id,
            table_name=normalized_table,
            columns=columns,
        )
    except Exception as exc:
        LOGGER.exception("inspect_table failed")
        return _error(str(exc), db_id=db_id, table_name=table_name)


def sample_rows_impl(db_id: str, table_name: str, n: int = 5) -> dict[str, Any]:
    try:
        normalized_table = _normalize_table_name(db_id, table_name)
        row_limit = max(1, min(int(n), DEFAULT_ROW_LIMIT))
        db_path = _db_path_for_id(db_id)
        sql = f"SELECT * FROM {quote_identifier(normalized_table)} LIMIT {row_limit}"
        with sqlite3.connect(str(db_path.resolve())) as connection:
            cursor = connection.execute(sql)
            columns = [description[0] for description in cursor.description or []]
            rows = cursor.fetchall()
        return _ok(
            **_schema_metadata("sample_rows"),
            db_id=db_id,
            table_name=normalized_table,
            n=row_limit,
            columns=columns,
            rows=_jsonable_rows(rows),
            row_count=len(rows),
        )
    except Exception as exc:
        LOGGER.exception("sample_rows failed")
        return _error(str(exc), db_id=db_id, table_name=table_name)


def search_column_values_impl(db_id: str, table_name: str, column_name: str, query: str) -> dict[str, Any]:
    try:
        normalized_table = _normalize_table_name(db_id, table_name)
        normalized_column = _normalize_column_name(db_id, normalized_table, column_name)
        db_path = _db_path_for_id(db_id)
        sql = (
            f"SELECT DISTINCT {quote_identifier(normalized_column)} "
            f"FROM {quote_identifier(normalized_table)} "
            f"WHERE {quote_identifier(normalized_column)} IS NOT NULL "
            f"AND CAST({quote_identifier(normalized_column)} AS TEXT) LIKE ? "
            f"LIMIT {DEFAULT_ROW_LIMIT}"
        )
        with sqlite3.connect(str(db_path.resolve())) as connection:
            cursor = connection.execute(sql, (f"%{query}%",))
            values = [row[0] for row in cursor.fetchall()]
        return _ok(
            **_schema_metadata("search_column_values"),
            db_id=db_id,
            table_name=normalized_table,
            column_name=normalized_column,
            query=query,
            values=values,
            value_count=len(values),
        )
    except Exception as exc:
        LOGGER.exception("search_column_values failed")
        return _error(str(exc), db_id=db_id, table_name=table_name, column_name=column_name)


def execute_sql_impl(db_id: str, sql: str) -> dict[str, Any]:
    try:
        is_safe, block_reason = check_sql_safety(sql)
        if not is_safe:
            return _ok(
                **_schema_metadata("execute_sql"),
                db_id=db_id,
                execution_success=False,
                columns=[],
                rows=[],
                row_count=0,
                error=f"SQL safety check failed: {block_reason}",
                safety_blocked=True,
                block_reason=block_reason,
            )

        db_path = _db_path_for_id(db_id)
        limited_sql = sql.strip().rstrip(";")
        with sqlite3.connect(str(db_path.resolve())) as connection:
            cursor = connection.execute(limited_sql)
            columns = [description[0] for description in cursor.description or []]
            rows = cursor.fetchmany(DEFAULT_ROW_LIMIT)
        return _ok(
            **_schema_metadata("execute_sql"),
            db_id=db_id,
            execution_success=True,
            columns=columns,
            rows=_jsonable_rows(rows),
            row_count=len(rows),
            error=None,
            safety_blocked=False,
            max_rows=DEFAULT_ROW_LIMIT,
        )
    except Exception as exc:
        LOGGER.exception("execute_sql failed")
        return _ok(
            **_schema_metadata("execute_sql"),
            db_id=db_id,
            execution_success=False,
            columns=[],
            rows=[],
            row_count=0,
            error=str(exc),
            safety_blocked=False,
        )


def create_server():
    if FastMCP is None:
        raise RuntimeError(
            "MCP Python SDK is not installed. Install with `pip install 'mcp[cli]'` "
            "in a Python 3.10+ environment."
        )

    mcp = FastMCP("text2sql-tools")

    @mcp.tool()
    def retrieve_schema(question: str, db_id: str) -> dict[str, Any]:
        """Retrieve question-relevant schema context for a local BIRD database."""
        return retrieve_schema_impl(question=question, db_id=db_id)

    @mcp.tool()
    def inspect_table(db_id: str, table_name: str) -> dict[str, Any]:
        """Inspect column metadata for one table in a local BIRD database."""
        return inspect_table_impl(db_id=db_id, table_name=table_name)

    @mcp.tool()
    def sample_rows(db_id: str, table_name: str, n: int = 5) -> dict[str, Any]:
        """Return a small row preview from a table, capped at 20 rows."""
        return sample_rows_impl(db_id=db_id, table_name=table_name, n=n)

    @mcp.tool()
    def search_column_values(db_id: str, table_name: str, column_name: str, query: str) -> dict[str, Any]:
        """Search distinct text-matched values in one column."""
        return search_column_values_impl(
            db_id=db_id,
            table_name=table_name,
            column_name=column_name,
            query=query,
        )

    @mcp.tool()
    def execute_sql(db_id: str, sql: str) -> dict[str, Any]:
        """Execute one safe read-only SELECT/WITH SQL query and return at most 20 rows."""
        return execute_sql_impl(db_id=db_id, sql=sql)

    return mcp


# Expose a global FastMCP object for `mcp dev`.
# When the MCP SDK is not installed, keep module importable and fail clearly in main().
mcp = create_server() if FastMCP is not None else None
server = mcp
app = mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Text2SQL MCP tool server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.getenv("TEXT2SQL_MCP_TRANSPORT", "stdio"),
        help="MCP transport. Defaults to stdio for MCP Inspector and desktop clients.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=os.getenv("TEXT2SQL_MCP_LOG_LEVEL", "WARNING"), stream=sys.stderr)
    args = parse_args()
    if mcp is None:
        message = (
            "MCP Python SDK is not installed. Install with `pip install 'mcp[cli]'` "
            "in a Python 3.10+ environment."
        )
        LOGGER.error("%s", message)
        print(message, file=sys.stderr)
        return 1

    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
