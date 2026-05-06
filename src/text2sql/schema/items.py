"""Build column-level schema items from a SQLite database."""

import csv
import sqlite3
from pathlib import Path


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def _read_descriptions(db_path):
    db_file = Path(db_path)
    description_dir = db_file.parent / "database_description"
    descriptions = {}
    if not description_dir.exists():
        return descriptions

    for csv_path in sorted(description_dir.glob("*.csv")):
        table_descriptions = {}
        reader = None
        last_error = None
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                with csv_path.open("r", encoding=encoding, newline="") as file:
                    reader = list(csv.DictReader(file))
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if reader is None:
            raise last_error

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
        descriptions[csv_path.stem] = table_descriptions
    return descriptions


def _get_tables(connection):
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


def _get_fk_columns(connection, table):
    cursor = connection.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})")
    return {row[3]: {"to_table": row[2], "to_column": row[4]} for row in cursor.fetchall()}


def _get_sample_values(connection, table, column, limit=3):
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


def build_schema_items(db_path, sample_value_limit=3):
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    descriptions = _read_descriptions(db_path)
    connection = sqlite3.connect(str(db_file.resolve()))
    items = []
    try:
        for table in _get_tables(connection):
            fk_columns = _get_fk_columns(connection, table)
            cursor = connection.execute(f"PRAGMA table_info({quote_identifier(table)})")
            for row in cursor.fetchall():
                column = row[1]
                fk_info = fk_columns.get(column)
                items.append(
                    {
                        "table": table,
                        "column": column,
                        "type": row[2] or "UNKNOWN",
                        "is_pk": int(row[5]) > 0,
                        "is_fk": fk_info is not None,
                        "fk_ref": fk_info,
                        "description": descriptions.get(table, {}).get(column, ""),
                        "sample_values": _get_sample_values(connection, table, column, sample_value_limit),
                    }
                )
    finally:
        connection.close()
    return items


def format_schema_item(item):
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


def format_linked_schema_text(items):
    lines = ["Linked SQLite schema columns:"]
    lines.extend(format_schema_item(item) for item in items)
    return "\n".join(lines)


def is_join_key(item):
    return (
        item["column"] in {"CDSCode", "cds", "NCESDist", "NCESSchool"}
        or item.get("is_pk", False)
        or item.get("is_fk", False)
    )


def format_table_linked_schema_text(selected_items, all_items):
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
