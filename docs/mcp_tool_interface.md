# Text2SQL MCP Tool Interface

## Purpose

`src/text2sql/agents/mcp_server.py` exposes the Text2SQL database tools through a real MCP server so MCP clients and the MCP Inspector can call schema and SQL-execution tools directly.

The server is a tool boundary only. It does not call an LLM, does not run evaluator correctness, and does not use `gold_sql`. The iterative agent can now use this server as a replaceable runtime tool backend through `MCPToolExecutor`.

## Exposed Tools

The MCP server exposes five tools:

- `retrieve_schema(question: str, db_id: str)`
- `inspect_table(db_id: str, table_name: str)`
- `sample_rows(db_id: str, table_name: str, n: int = 5)`
- `search_column_values(db_id: str, table_name: str, column_name: str, query: str)`
- `execute_sql(db_id: str, sql: str)`

Each tool returns a JSON-serializable dictionary. Successful calls include `{"ok": true, ...}`. Tool-level errors return `{"ok": false, "error": "..."}`. SQL execution errors are reported as `{"ok": true, "execution_success": false, ...}` because the tool call itself succeeded and returned an execution observation.

## Why `finish` Is Not Exposed

`finish(final_sql)` is an agent-internal finalization action. It tells an agent loop to stop and submit a final answer. External MCP clients need database observations, not an internal stop signal, so `finish` is intentionally not exposed as an MCP tool.

## SQL Safety

`execute_sql` is read-only and guarded before execution.

Allowed:

- `SELECT`
- `WITH`

Blocked:

- `DROP`
- `DELETE`
- `UPDATE`
- `INSERT`
- `ALTER`
- `CREATE`
- `REPLACE`
- `ATTACH`
- `DETACH`
- `VACUUM`
- `.load`
- multiple SQL statements

Returned rows are capped at 20. The response includes:

- `execution_success`
- `columns`
- `rows`
- `row_count`
- `error`
- `safety_blocked`

## Startup

The default transport is stdio, which is the normal mode for MCP Inspector and desktop clients:

```bash
PYTHONPATH=src python3 -m text2sql.agents.mcp_server
```

You can also pass the transport explicitly:

```bash
PYTHONPATH=src python3 -m text2sql.agents.mcp_server --transport stdio
```

The implementation accepts `--transport streamable-http` and `TEXT2SQL_MCP_TRANSPORT=streamable-http`, but stdio is the primary supported local path.

The current local Python in this workspace is 3.9.6. Recent MCP Python SDK releases generally require Python 3.10+. If `mcp` is not installed, the server module still compiles, but startup reports that `mcp[cli]` must be installed in a Python 3.10+ environment.

## Inspector / Client Test

In a Python 3.10+ environment with the MCP SDK installed:

```bash
pip install "mcp[cli]"
PYTHONPATH=src mcp dev src/text2sql/agents/mcp_server.py
```

Or configure an MCP client to run:

```bash
PYTHONPATH=src python3 -m text2sql.agents.mcp_server
```

Then list tools and call a small smoke tool, for example:

```json
{
  "db_id": "california_schools",
  "table_name": "schools",
  "n": 2
}
```

for `sample_rows`.

## Relationship To `tool_schemas.py`

Day 1 standardized the Text2SQL tool definitions in `tool_schemas.py`. The MCP server reuses that vocabulary and includes schema metadata in tool responses where useful. The MCP server intentionally exposes only externally useful tools; it omits `finish`.

The MCP signatures are aligned with the standardized tools:

- `retrieve_schema`
- `inspect_table`
- `sample_rows`
- `search_column_values`
- `execute_sql`

## Relationship To Autonomous Tool Selection

Autonomous tool selection can use the same conceptual tool surface, but it remains separate from this MCP server. The autonomous policy decides which tool to call inside an agent loop. The MCP server exposes callable tools to external MCP clients and to the iterative agent's MCP backend adapter.

This runtime adapter does not change autonomous policy, self-consistency, or stable pipelines. Memory remains backend-agnostic and only observes normalized tool input/output after the agent loop calls a `ToolExecutor`.

## Validation Scope

Do not run long experiments for MCP validation. The intended validation is tool connectivity and fixed tool-call parity:

- module compiles;
- server imports in an environment with MCP installed;
- server starts over stdio;
- MCP Inspector can list the five exposed tools;
- fixed local-vs-MCP tool parity passes or records clear mismatch reasons;
- small agent-level smoke runs show MCP backend tool calls and memory observations.

Commands:

```bash
PYTHONPATH=src python3 scripts/compare_tool_backends.py
PYTHONPATH=src python3 scripts/run_mcp_backend_smoke.py --limit 3 --max-steps 2
```

These checks are not full benchmarks and should not be reported as large-scale MCP backend EX results.
