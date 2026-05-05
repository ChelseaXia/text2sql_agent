"""Prompt builders for SQL generation experiments."""


def build_naive_prompt(sample, schema_text):
    evidence = sample.get("evidence") or "None"
    return f"""Given the SQLite database schema, evidence, and question, write the correct SQLite SQL query.

{schema_text}

Evidence:
{evidence}

Question:
{sample["question"]}

Return only the SQL query."""


def build_linked_prompt(sample, linked_schema_text):
    evidence = sample.get("evidence") or "None"
    return f"""Given the relevant SQLite schema tables, evidence, and question, write the correct SQLite SQL query.

SQL generation instruction:
- Use exact table and column names from the provided schema.
- Never invent normalized column names such as Enrollment, FRPM_Count, or Percent_Eligible_FRPM if the schema provides columns like `Enrollment (K-12)` or `Percent (%) Eligible FRPM (K-12)`.
- Wrap column names containing spaces, parentheses, hyphens, percent signs, or slashes with backticks.
- Be careful about table-column association: only use a column under the table where it appears in the schema.

{linked_schema_text}

Evidence:
{evidence}

Question:
{sample["question"]}

Return only the SQL query."""


def build_fewshot_prompt(sample, linked_schema_text, examples):
    evidence = sample.get("evidence") or "None"
    sections = [
        "You are an expert SQLite generator for Text-to-SQL tasks.",
        "",
        "Use the provided schema and examples to write a correct SQLite query.",
        "",
        "SQL generation rules:",
        "1. Use exact table and column names from the provided schema.",
        "2. Never invent normalized column names.",
        "3. Wrap column names containing spaces, parentheses, hyphens, percent signs, or slashes with backticks.",
        "4. Only use a column under the table where it appears in the schema.",
        "5. Output only the final SQLite SQL. Do not include explanation.",
        "",
        "Relevant SQLite schema:",
        linked_schema_text,
        "",
        "Evidence:",
        evidence,
    ]

    if examples:
        sections.extend(["", "Examples:"])
        for index, example in enumerate(examples, start=1):
            sections.extend(
                [
                    f"[Example {index}]",
                    f"Question: {example['question']}",
                    "SQL:",
                    example["gold_sql"],
                    "",
                ]
            )

    sections.extend(
        [
            "Current Question:",
            sample["question"],
            "",
            "SQLite SQL:",
        ]
    )
    return "\n".join(sections)

