# Text2SQL Tool Interface

## Why standardize tool definitions

The current Text2SQL tools are implemented as internal Python methods. That is enough for a controlled loop, but it is not enough for prompt-based function calling, autonomous tool selection, or an MCP server boundary. A standardized schema gives every tool a stable contract:

- `name`: the tool identifier the model or orchestrator can call.
- `description`: the short semantic purpose of the tool.
- `input_schema`: the JSON object expected from a tool call.
- `output_schema`: the JSON object returned as an observation.
- `when_to_use`: guidance for tool-selection prompts and future policies.
- `failure_modes`: expected error or uncertainty cases.
- `usage_notes`: operational constraints and safety notes.

This separates tool definition from tool implementation. The same schema can be used in prompts today and reused later by an autonomous controller or MCP adapter without changing the tool vocabulary.

## Tool inputs and outputs

### `retrieve_schema(question, db_id)`

Retrieves schema context relevant to the natural language question.

Input:

```json
{
  "question": "string",
  "db_id": "string"
}
```

Output:

```json
{
  "selected_tables": ["string"],
  "retrieved_columns": ["table.column"],
  "linked_schema_text": "string"
}
```

### `inspect_table(db_id, table_name)`

Returns column metadata for one table.

Input:

```json
{
  "db_id": "string",
  "table_name": "string"
}
```

Output:

```json
{
  "table_name": "string",
  "columns": [
    {
      "name": "string",
      "type": "string",
      "description": "string",
      "sample_values": [],
      "is_pk": true,
      "is_fk": false,
      "fk_ref": "string or null"
    }
  ]
}
```

### `sample_rows(db_id, table_name, n=5)`

Returns a small row preview from a table.

Input:

```json
{
  "db_id": "string",
  "table_name": "string",
  "n": 5
}
```

Output:

```json
{
  "table_name": "string",
  "success": true,
  "rows": [],
  "error": "string or null"
}
```

### `search_column_values(db_id, table_name, column_name, query)`

Searches distinct values in a column using a text query.

Input:

```json
{
  "db_id": "string",
  "table_name": "string",
  "column_name": "string",
  "query": "string"
}
```

Output:

```json
{
  "table_name": "string",
  "column_name": "string",
  "success": true,
  "values": [],
  "error": "string or null"
}
```

### `execute_sql(db_id, sql)`

Executes a read-only SQL query and returns execution status plus a row preview.

Input:

```json
{
  "db_id": "string",
  "sql": "string",
  "final_candidate": false
}
```

`final_candidate` is optional and defaults to `false`. Autonomous tool-selection mode uses it to distinguish exploratory probe SQL from SQL that may later be submitted with `finish`.

Output:

```json
{
  "success": true,
  "rows": [],
  "row_count": 0,
  "error": "string or null",
  "safety_blocked": false,
  "block_reason": "string or null"
}
```

### `finish(final_sql)`

Submits the final SQL answer and ends tool use for the task.

Input:

```json
{
  "final_sql": "string"
}
```

Output:

```json
{
  "final_sql": "string"
}
```

## Prompt-based function calling format

`format_tools_for_prompt()` emits tool instructions that can be pasted directly into an LLM prompt. The model must return only valid JSON in this shape:

```json
{
  "thought": "...",
  "tool": "...",
  "args": {}
}
```

`tool` must be one of:

- `retrieve_schema`
- `inspect_table`
- `sample_rows`
- `search_column_values`
- `execute_sql`
- `finish`

`args` must match the selected tool's `input_schema`. The caller can pass the returned JSON through `validate_tool_call(tool_name, args)` before dispatching to the actual implementation.

## Relationship to autonomous mode and MCP

This Day 1 interface is a contract layer only. It does not implement autonomous mode, memory, or an MCP server.

Later autonomous mode can use the same schemas as its action space: the controller can show these definitions in the prompt, validate each proposed call, execute the matching tool, append the observation, and continue until `finish`.

An MCP server can also expose the same six tools by translating each schema into MCP tool metadata and mapping MCP calls to the existing Python implementations. Keeping the names and JSON shapes stable reduces glue code and avoids a second tool vocabulary.

## No gold SQL or evaluator correctness

These tools are runtime aids for schema exploration, value inspection, SQL execution, and final answer submission. They do not directly use `gold_sql`, execution-match labels, evaluator correctness, or any oracle signal.

`execute_sql` can report whether SQL executed successfully and what rows were returned. A successful execution is only an observation; it is not proof that the SQL is semantically correct. `finish` records the final SQL, but it does not compare against the gold answer.
