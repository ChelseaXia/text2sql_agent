"""Parse SQLite schema into plain text for prompting."""

import sqlite3
from pathlib import Path
from typing import Dict, List


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _get_user_tables(connection: sqlite3.Connection) -> List[str]:
    cursor = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [row[0] for row in cursor.fetchall()]


def _get_columns(connection: sqlite3.Connection, table_name: str) -> List[Dict]:
    cursor = connection.execute(f"PRAGMA table_info({_quote_identifier(table_name)})")
    return [
        {
            "name": row[1],
            "type": row[2] or "UNKNOWN",
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": int(row[5]),
        }
        for row in cursor.fetchall()
    ]


def _get_foreign_keys(connection: sqlite3.Connection, table_name: str) -> List[Dict]:
    cursor = connection.execute(f"PRAGMA foreign_key_list({_quote_identifier(table_name)})")
    return [
        {
            "from": row[3],
            "to_table": row[2],
            "to": row[4],
        }
        for row in cursor.fetchall()
    ]


def get_schema_info(db_path: str) -> Dict:
    """Return structured table, column, primary-key, and foreign-key metadata."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    connection = sqlite3.connect(str(db_file.resolve()))
    try:
        tables = []
        for table_name in _get_user_tables(connection):
            columns = _get_columns(connection, table_name)
            foreign_keys = _get_foreign_keys(connection, table_name)
            primary_keys = [column["name"] for column in columns if column["pk"] > 0]
            tables.append(
                {
                    "name": table_name,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                }
            )
        return {"db_path": str(db_file.resolve()), "tables": tables}
    finally:
        connection.close()


def format_schema_text(schema_info: Dict) -> str:
    """Format structured SQLite metadata as concise full-schema prompt text."""
    lines = ["SQLite schema:"]

    for table in schema_info["tables"]:
        lines.append(f"\nTable: {table['name']}")
        lines.append("Columns:")
        for column in table["columns"]:
            attrs = [column["type"]]
            if column["pk"] > 0:
                attrs.append("PRIMARY KEY")
            if column["notnull"]:
                attrs.append("NOT NULL")
            lines.append(f"- {column['name']} ({', '.join(attrs)})")

        if table["foreign_keys"]:
            lines.append("Foreign keys:")
            for fk in table["foreign_keys"]:
                lines.append(f"- {fk['from']} -> {fk['to_table']}.{fk['to']}")

    return "\n".join(lines)


def get_full_schema_text(db_path: str) -> str:
    """Return the full SQLite schema text used by the naive baseline."""
    return format_schema_text(get_schema_info(db_path))


if __name__ == "__main__":
    try:
        from load_bird import load_bird_dev
    except ModuleNotFoundError:
        from src.load_bird import load_bird_dev

    sample = load_bird_dev(limit=1, db_id="california_schools")[0]
    print(get_full_schema_text(sample["db_path"]))
