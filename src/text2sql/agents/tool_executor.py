"""ToolExecutor backends for the iterative Text2SQL agent."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from text2sql.agents.tools import check_sql_safety
from text2sql.db import run_sql
from text2sql.pipelines.schema_linked import retrieved_column_names
from text2sql.schema.items import build_schema_items, quote_identifier
from text2sql.schema.linker import DEFAULT_TOP_K


DEFAULT_SAMPLE_ROWS = 5
DEFAULT_VALUE_SEARCH_LIMIT = 20
MCP_METADATA_KEYS = {
    "ok",
    "db_id",
    "tool_schema_name",
    "tool_schema_description",
    "schema_linker_mode",
    "max_rows",
    "n",
    "value_count",
}


def compact_rows(rows, limit=5):
    return [list(row) for row in rows[:limit]]


class ToolExecutor(ABC):
    """Backend-neutral database tool interface used by the agent loop."""

    backend_name = "abstract"

    @abstractmethod
    def retrieve_schema(self, question: str | None = None, db_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def inspect_table(self, table_name: str, db_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sample_rows(self, table_name: str, n: int = DEFAULT_SAMPLE_ROWS, db_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def search_column_values(
        self,
        table_name: str,
        column_name: str,
        query: str,
        db_id: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def execute_sql(self, sql: str, db_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def finish(self, final_sql: str) -> dict[str, Any]:
        return {"final_sql": final_sql}

    def close(self) -> None:
        return None


class LocalToolExecutor(ToolExecutor):
    """Local Python tool backend preserving the iterative agent's behavior."""

    backend_name = "local"

    def __init__(self, sample: dict[str, Any], schema_linker: Any, top_k_schema: int = DEFAULT_TOP_K):
        self.sample = sample
        self.schema_linker = schema_linker
        self.top_k_schema = top_k_schema
        self._schema_items = None
        self._items_by_table = None

    @property
    def schema_items(self):
        if self._schema_items is None:
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

    def retrieve_schema(
        self,
        question: str | None = None,
        db_id: str | None = None,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        self._validate_db_id(db_id)
        effective_question = question or self.sample["question"]
        effective_evidence = self.sample.get("evidence", "") if question is None and evidence is None else (evidence or "")
        linked_items, linked_schema_text = self.schema_linker.retrieve(
            effective_question,
            effective_evidence,
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

    def inspect_table(self, table_name: str, db_id: str | None = None) -> dict[str, Any]:
        self._validate_db_id(db_id)
        table_name = self.normalize_table_name(table_name)
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

    def sample_rows(self, table_name: str, n: int = DEFAULT_SAMPLE_ROWS, db_id: str | None = None) -> dict[str, Any]:
        self._validate_db_id(db_id)
        table_name = self.normalize_table_name(table_name)
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

    def search_column_values(
        self,
        table_name: str,
        column_name: str,
        query: str,
        db_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_db_id(db_id)
        table_name = self.normalize_table_name(table_name)
        column_name = self.normalize_column_name(table_name, column_name)
        self._validate_column(table_name, column_name)
        sql = (
            f"SELECT DISTINCT {quote_identifier(column_name)} "
            f"FROM {quote_identifier(table_name)} "
            f"WHERE {quote_identifier(column_name)} IS NOT NULL "
            f"AND CAST({quote_identifier(column_name)} AS TEXT) LIKE ? "
            f"LIMIT {DEFAULT_VALUE_SEARCH_LIMIT}"
        )
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

    def execute_sql(self, sql: str, db_id: str | None = None) -> dict[str, Any]:
        self._validate_db_id(db_id)
        if not sql:
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "error": "Empty SQL prediction",
                "safety_blocked": False,
                "block_reason": None,
            }
        is_safe, block_reason = check_sql_safety(sql)
        if not is_safe:
            return {
                "success": False,
                "rows": [],
                "row_count": 0,
                "error": f"SQL safety check failed: {block_reason}",
                "safety_blocked": True,
                "block_reason": block_reason,
            }
        result = run_sql(sql, self.sample["db_path"])
        return {
            "success": result["success"],
            "rows": compact_rows(result["rows"], len(result["rows"])),
            "row_count": len(result["rows"]),
            "error": result["error"],
            "safety_blocked": False,
            "block_reason": None,
        }

    def _validate_db_id(self, db_id: str | None) -> None:
        if db_id is not None and db_id != self.sample["db_id"]:
            raise ValueError(f"ToolExecutor for {self.sample['db_id']} cannot query db_id={db_id}.")

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


class MCPToolExecutor(ToolExecutor):
    """MCP client backend that calls the Text2SQL MCP server over stdio."""

    backend_name = "mcp"

    def __init__(
        self,
        sample: dict[str, Any],
        schema_linker: Any | None = None,
        top_k_schema: int = DEFAULT_TOP_K,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
    ):
        self.sample = sample
        self.top_k_schema = top_k_schema
        self._local_names = LocalToolExecutor(sample, schema_linker, top_k_schema) if schema_linker is not None else None
        self.server_command = server_command or sys.executable
        self.server_args = server_args or ["-m", "text2sql.agents.mcp_server", "--transport", "stdio"]
        env = dict(os.environ)
        src_dir = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else src_dir
        if server_env:
            env.update(server_env)
        self.server_env = env

    @property
    def schema_items(self):
        return self._require_local_names().schema_items

    @property
    def items_by_table(self):
        return self._require_local_names().items_by_table

    def retrieve_schema(self, question: str | None = None, db_id: str | None = None) -> dict[str, Any]:
        payload = self._call_tool(
            "retrieve_schema",
            {"question": question or self.sample["question"], "db_id": db_id or self.sample["db_id"]},
        )
        return self._normalize_mcp_observation("retrieve_schema", payload)

    def inspect_table(self, table_name: str, db_id: str | None = None) -> dict[str, Any]:
        table_name = self.normalize_table_name(table_name)
        payload = self._call_tool("inspect_table", {"db_id": db_id or self.sample["db_id"], "table_name": table_name})
        return self._normalize_mcp_observation("inspect_table", payload)

    def sample_rows(self, table_name: str, n: int = DEFAULT_SAMPLE_ROWS, db_id: str | None = None) -> dict[str, Any]:
        table_name = self.normalize_table_name(table_name)
        payload = self._call_tool(
            "sample_rows",
            {"db_id": db_id or self.sample["db_id"], "table_name": table_name, "n": n},
        )
        return self._normalize_mcp_observation("sample_rows", payload)

    def search_column_values(
        self,
        table_name: str,
        column_name: str,
        query: str,
        db_id: str | None = None,
    ) -> dict[str, Any]:
        table_name = self.normalize_table_name(table_name)
        column_name = self.normalize_column_name(table_name, column_name)
        payload = self._call_tool(
            "search_column_values",
            {
                "db_id": db_id or self.sample["db_id"],
                "table_name": table_name,
                "column_name": column_name,
                "query": query,
            },
        )
        return self._normalize_mcp_observation("search_column_values", payload)

    def execute_sql(self, sql: str, db_id: str | None = None) -> dict[str, Any]:
        payload = self._call_tool("execute_sql", {"db_id": db_id or self.sample["db_id"], "sql": sql})
        return self._normalize_mcp_observation("execute_sql", payload)

    def normalize_table_name(self, table_name):
        return self._require_local_names().normalize_table_name(table_name)

    def normalize_column_name(self, table_name, column_name):
        return self._require_local_names().normalize_column_name(table_name, column_name)

    def _require_local_names(self) -> LocalToolExecutor:
        if self._local_names is None:
            raise ValueError("MCPToolExecutor requires schema_linker for local table/column normalization.")
        return self._local_names

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._call_tool_async(tool_name, arguments))

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP backend requires the MCP Python SDK. Install `mcp[cli]` in the Python "
                "environment used to run the agent."
            ) from exc

        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=self.server_env,
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return self._decode_call_result(result)

    def _decode_call_result(self, result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        content = getattr(result, "content", None) or []
        if content:
            first = content[0]
            text = getattr(first, "text", None)
            if text:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
        if isinstance(result, dict):
            return result
        raise ValueError(f"Unable to decode MCP tool result: {result!r}")

    def _normalize_mcp_observation(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("ok", False):
            return {
                "success": False,
                "error": payload.get("error", "MCP tool call failed"),
                "mcp_payload": payload,
            }
        normalized = {key: value for key, value in payload.items() if key not in MCP_METADATA_KEYS}
        if tool_name == "sample_rows":
            normalized.pop("columns", None)
            normalized["success"] = True
            normalized.setdefault("error", None)
        elif tool_name == "search_column_values":
            normalized["success"] = True
            normalized.setdefault("error", None)
        elif tool_name == "execute_sql":
            normalized.pop("columns", None)
            normalized["success"] = bool(payload.get("execution_success"))
            normalized["rows"] = compact_rows(payload.get("rows", []), payload.get("row_count", 0))
            normalized["row_count"] = payload.get("row_count", len(normalized["rows"]))
            normalized["error"] = payload.get("error")
            normalized["safety_blocked"] = bool(payload.get("safety_blocked"))
            normalized["block_reason"] = payload.get("block_reason")
            normalized.pop("execution_success", None)
        return normalized


def build_tool_executor(
    backend: str,
    sample: dict[str, Any],
    schema_linker: Any,
    top_k_schema: int = DEFAULT_TOP_K,
) -> ToolExecutor:
    if backend == "local":
        return LocalToolExecutor(sample, schema_linker, top_k_schema=top_k_schema)
    if backend == "mcp":
        return MCPToolExecutor(sample, schema_linker=schema_linker, top_k_schema=top_k_schema)
    raise ValueError("tool_backend must be 'local' or 'mcp'.")
