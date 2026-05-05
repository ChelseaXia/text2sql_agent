"""Tool implementations for the ReAct-style Text2SQL agent."""

import sqlite3
from dataclasses import dataclass

from text2sql.db import run_sql
from text2sql.schema.items import build_schema_items, quote_identifier

DEFAULT_ROWS_PREVIEW_LIMIT = 5
DEFAULT_VALUE_SEARCH_LIMIT = 20


def rows_preview(rows, limit=DEFAULT_ROWS_PREVIEW_LIMIT):
    return rows[:limit]


def selected_tables_from_items(items):
    tables = []
    for item in items:
        if item["table"] not in tables:
            tables.append(item["table"])
    return tables


@dataclass
class SampleContext:
    sample: dict
    schema_linker: object
    schema_items: list
    items_by_table: dict


def build_sample_context(sample, schema_linker):
    items = build_schema_items(sample["db_path"])
    items_by_table = {}
    for item in items:
        items_by_table.setdefault(item["table"], []).append(item)
    return SampleContext(
        sample=sample,
        schema_linker=schema_linker,
        schema_items=items,
        items_by_table=items_by_table,
    )


class AgentTools:
    def __init__(self, context, top_k_schema):
        self.context = context
        self.top_k_schema = top_k_schema

    def _validate_table(self, table_name):
        if table_name not in self.context.items_by_table:
            raise ValueError(f"Unknown table: {table_name}")
        return self.context.items_by_table[table_name]

    def _validate_column(self, table_name, column_name):
        for item in self._validate_table(table_name):
            if item["column"] == column_name:
                return item
        raise ValueError(f"Unknown column: {table_name}.{column_name}")

    def retrieve_schema(self, question, evidence, db_id, db_path):
        sample = self.context.sample
        if db_id != sample["db_id"] or db_path != sample["db_path"]:
            raise ValueError("retrieve_schema must use the current sample db_id and db_path.")
        linked_items, linked_schema_text = self.context.schema_linker.retrieve(
            question,
            evidence,
            top_k=self.top_k_schema,
        )
        return {
            "selected_tables": selected_tables_from_items(linked_items),
            "retrieved_columns": [f"{item['table']}.{item['column']}" for item in linked_items],
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

    def sample_rows(self, table_name, n=3):
        self._validate_table(table_name)
        row_limit = max(1, min(int(n), 10))
        sql = f'SELECT * FROM {quote_identifier(table_name)} LIMIT {row_limit}'
        result = run_sql(sql, self.context.sample["db_path"])
        return {
            "table_name": table_name,
            "success": result["success"],
            "rows": rows_preview(result["rows"], limit=row_limit),
            "error": result["error"],
        }

    def search_column_values(self, table_name, column_name, keyword, limit=DEFAULT_VALUE_SEARCH_LIMIT):
        self._validate_column(table_name, column_name)
        search_limit = max(1, min(int(limit), DEFAULT_VALUE_SEARCH_LIMIT))
        connection = sqlite3.connect(self.context.sample["db_path"])
        try:
            sql = (
                f"SELECT DISTINCT {quote_identifier(column_name)} "
                f"FROM {quote_identifier(table_name)} "
                f"WHERE {quote_identifier(column_name)} IS NOT NULL "
                f"AND CAST({quote_identifier(column_name)} AS TEXT) LIKE ? "
                f"LIMIT {search_limit}"
            )
            cursor = connection.execute(sql, (f"%{keyword}%",))
            values = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return {
                "table_name": table_name,
                "column_name": column_name,
                "success": True,
                "values": values,
                "error": None,
            }
        except sqlite3.Error as exc:
            return {
                "table_name": table_name,
                "column_name": column_name,
                "success": False,
                "values": [],
                "error": str(exc),
            }
        finally:
            connection.close()

    def execute_sql(self, sql):
        result = run_sql(sql, self.context.sample["db_path"])
        return {
            "success": result["success"],
            "rows": rows_preview(result["rows"]),
            "row_count": len(result["rows"]),
            "error": result["error"],
        }

    def finish(self, sql):
        return {"final_sql": sql}

