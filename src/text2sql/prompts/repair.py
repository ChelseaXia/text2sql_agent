"""Prompt builders for SQL repair experiments."""


def build_repair_prompt(question, evidence, linked_schema_text, previous_sql, sqlite_error):
    safe_evidence = evidence or "None"
    return f"""You are repairing a SQLite query for a Text-to-SQL task.

The previous SQL failed during execution.

Question:
{question}

Evidence:
{safe_evidence}

Relevant SQLite schema:
{linked_schema_text}

Previous SQL:
{previous_sql}

SQLite Error:
{sqlite_error}

Repair rules:
1. Use only tables and columns from the provided schema.
2. Use exact table and column names.
3. Do not invent normalized column names.
4. If a column name contains spaces, parentheses, hyphens, slashes, or percent signs, wrap it with backticks.
5. Be careful about table aliases.
6. Preserve the intended meaning of the original question.
7. Output only the corrected SQLite SQL. Do not include explanation.

Corrected SQLite SQL:"""

