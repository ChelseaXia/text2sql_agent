"""Schema serialization helpers for linked-schema prompting."""

from itertools import combinations

from text2sql.schema.items import is_join_key


def quote_identifier(name):
    return "`" + name.replace("`", "``") + "`"


def _selected_tables(selected_items):
    tables = []
    for item in selected_items:
        if item["table"] not in tables:
            tables.append(item["table"])
    return tables


def _sample_values_text(sample_values):
    values = []
    for value in list(sample_values)[:3]:
        values.append(repr(value) if isinstance(value, str) else str(value))
    return ", ".join(values)


def _column_comment(item, retrieved_columns):
    tags = []
    if (item["table"], item["column"]) in retrieved_columns:
        tags.append("RETRIEVED")
    if is_join_key(item):
        tags.append("JOIN_KEY")

    parts = []
    if tags:
        parts.append("[" + " ; ".join(tags) + "]")
    if item.get("description"):
        parts.append(item["description"])
    samples = _sample_values_text(item.get("sample_values") or [])
    if samples:
        parts.append(f"samples: {samples}")
    return " -- " + " | ".join(parts) if parts else ""


def _normalized_join_group(column_name):
    normalized = "".join(ch.lower() for ch in column_name if ch.isalnum())
    groups = {
        "cdscode": "cdscode",
        "cds": "cdscode",
        "ncesdist": "ncesdist",
        "ncesschool": "ncesschool",
    }
    return groups.get(normalized)


def _join_key_edges(selected_tables, all_items):
    candidates = {}
    explicit_edges = set()
    for item in all_items:
        if item["table"] not in selected_tables:
            continue

        fk_ref = item.get("fk_ref")
        if fk_ref and fk_ref["to_table"] in selected_tables:
            explicit_edges.add((item["table"], item["column"], fk_ref["to_table"], fk_ref["to_column"]))

        group = _normalized_join_group(item["column"])
        if group and is_join_key(item):
            candidates.setdefault(group, []).append((item["table"], item["column"]))

    inferred_edges = set()
    for group_items in candidates.values():
        for left, right in combinations(sorted(set(group_items)), 2):
            if left[0] != right[0]:
                inferred_edges.add((left[0], left[1], right[0], right[1]))
    return sorted(explicit_edges | inferred_edges)


def format_table_linked_schema_ddl(selected_items, all_items):
    selected_tables = _selected_tables(selected_items)
    retrieved_columns = {(item["table"], item["column"]) for item in selected_items}
    lines = ["Relevant SQLite schema:", ""]

    for table_index, table in enumerate(selected_tables):
        table_items = [item for item in all_items if item["table"] == table]
        lines.append(f"CREATE TABLE {quote_identifier(table)} (")
        for index, item in enumerate(table_items):
            suffix = "," if index < len(table_items) - 1 else ""
            comment = _column_comment(item, retrieved_columns)
            lines.append(f"  {quote_identifier(item['column'])} {item['type']}{suffix}{comment}")
        lines.append(");")
        if table_index < len(selected_tables) - 1:
            lines.append("")

    edges = _join_key_edges(selected_tables, all_items)
    lines.append("")
    lines.append("Foreign keys / join keys:")
    if edges:
        for left_table, left_column, right_table, right_column in edges:
            lines.append(
                f"- {quote_identifier(left_table)}.{quote_identifier(left_column)} = "
                f"{quote_identifier(right_table)}.{quote_identifier(right_column)}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines), selected_tables
