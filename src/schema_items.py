"""Build column-level schema items from a SQLite database."""

import csv
import sqlite3
from pathlib import Path
from typing import Dict, List


def quote_identifier(name: str) -> str:
    """Quote a SQLite identifier, including names with spaces or punctuation."""
    return '"' + name.replace('"', '""') + '"'


def _read_descriptions(db_path: str) -> Dict[str, Dict[str, str]]:
    db_file = Path(db_path)
    description_dir = db_file.parent / "database_description"
    descriptions: Dict[str, Dict[str, str]] = {}

    if not description_dir.exists():
        return descriptions

    for csv_path in sorted(description_dir.glob("*.csv")):
        table = csv_path.stem
        table_descriptions: Dict[str, str] = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                column = (row.get("original_column_name") or "").strip()
                if not column:
                    continue
                parts = [
                    (row.get("column_name") or "").strip(),
                    (row.get("column_description") or "").strip(),
                    (row.get("value_description") or "").strip(),
                ]
                table_descriptions[column] = " ".join(part for part in parts if part)
        descriptions[table] = table_descriptions

    return descriptions


def _get_tables(connection: sqlite3.Connection) -> List[str]:
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


def _get_fk_columns(connection: sqlite3.Connection, table: str) -> Dict[str, Dict[str, str]]:
    cursor = connection.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})")
    return {
        row[3]: {
            "to_table": row[2],
            "to_column": row[4],
        }
        for row in cursor.fetchall()
    }


def _get_sample_values(connection: sqlite3.Connection, table: str, column: str, limit: int = 3) -> List:
    sql = (
        f"SELECT DISTINCT {quote_identifier(column)} "
        f"FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(column)} IS NOT NULL "
        f"LIMIT {limit}"
    )
    try:
        cursor = connection.execute(sql)
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error:
        return []


def build_schema_items(db_path: str, sample_value_limit: int = 3) -> List[Dict]:
    """Create one schema item per table.column."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    descriptions = _read_descriptions(db_path)
    connection = sqlite3.connect(str(db_file.resolve()))
    items: List[Dict] = []

    try:
        for table in _get_tables(connection):
            fk_columns = _get_fk_columns(connection, table)
            cursor = connection.execute(f"PRAGMA table_info({quote_identifier(table)})")
            for row in cursor.fetchall():
                column = row[1]
                column_type = row[2] or "UNKNOWN"
                is_pk = int(row[5]) > 0
                fk_info = fk_columns.get(column)
                sample_values = _get_sample_values(connection, table, column, sample_value_limit)
                items.append(
                    {
                        "table": table,
                        "column": column,
                        "type": column_type,
                        "is_pk": is_pk,
                        "is_fk": fk_info is not None,
                        "fk_ref": fk_info,
                        "description": descriptions.get(table, {}).get(column, ""),
                        "sample_values": sample_values,
                    }
                )
    finally:
        connection.close()

    return items


def format_schema_item(item: Dict) -> str:
    """Format one schema item for linked-schema prompts."""
    flags = []
    if item["is_pk"]:
        flags.append("PK")
    if item["is_fk"] and item.get("fk_ref"):
        flags.append(f"FK->{item['fk_ref']['to_table']}.{item['fk_ref']['to_column']}")

    suffix = f" [{' ; '.join(flags)}]" if flags else ""
    description = f" | description: {item['description']}" if item["description"] else ""
    sample_values = item.get("sample_values") or []
    samples = f" | sample_values: {sample_values}" if sample_values else ""
    return f"- {item['table']}.{item['column']} ({item['type']}){suffix}{description}{samples}"


def format_linked_schema_text(items: List[Dict]) -> str:
    """Format selected schema items as prompt text."""
    lines = ["Linked SQLite schema columns:"]
    lines.extend(format_schema_item(item) for item in items)
    return "\n".join(lines)


def is_join_key(item: Dict) -> bool:
    """Return whether a schema item should be emphasized as a join key."""
    return (
        item["column"] in {"CDSCode", "cds", "NCESDist", "NCESSchool"}
        or item.get("is_pk", False)
        or item.get("is_fk", False)
    )


def format_table_linked_schema_text(selected_items: List[Dict], all_items: List[Dict]) -> str:
    """Format selected tables as full schemas and mark retrieved columns."""
    selected_tables = []
    selected_columns = set()

    for item in selected_items:
        if item["table"] not in selected_tables:
            selected_tables.append(item["table"])
        selected_columns.add((item["table"], item["column"]))

    lines = ["Relevant SQLite schema:"]
    for table in selected_tables:
        lines.append(f"\nTable: {table}")
        lines.append("Columns:")
        for item in all_items:
            if item["table"] != table:
                continue

            tags = []
            if (item["table"], item["column"]) in selected_columns:
                tags.append("RETRIEVED")
            if is_join_key(item):
                tags.append("JOIN_KEY")
            if item.get("is_pk"):
                tags.append("PK")
            if item.get("is_fk") and item.get("fk_ref"):
                tags.append(f"FK->{item['fk_ref']['to_table']}.{item['fk_ref']['to_column']}")

            tag_text = f" [{' ; '.join(tags)}]" if tags else ""
            description = f" | description: {item['description']}" if item["description"] else ""
            sample_values = item.get("sample_values") or []
            samples = f" | sample_values: {sample_values}" if sample_values else ""
            lines.append(f"- {item['column']} ({item['type']}){tag_text}{description}{samples}")

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        from load_bird import load_bird_dev
    except ModuleNotFoundError:
        from src.load_bird import load_bird_dev

    sample = load_bird_dev(limit=1, db_id="california_schools")[0]
    for schema_item in build_schema_items(sample["db_path"])[:10]:
        print(format_schema_item(schema_item))
