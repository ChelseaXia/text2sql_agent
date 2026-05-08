# Agent System Runtime Design

## Runtime Boundary

The iterative agent now calls database tools through a `ToolExecutor` abstraction instead of binding the loop directly to one implementation.

Supported backends:

- `local`: calls the existing Python tool implementation in-process.
- `mcp`: calls the same tool vocabulary through the MCP server over stdio with an MCP client.

The default backend is still `local`, so existing iterative-agent commands remain compatible unless `--tool-backend mcp` is explicitly passed.

## Tool Surface

`ToolExecutor` exposes the five runtime database tools:

- `retrieve_schema`
- `inspect_table`
- `sample_rows`
- `search_column_values`
- `execute_sql`

`finish` remains an agent-internal finalization action. It is not served by MCP because external MCP clients need database observations, not agent stop control.

## Backend Behavior

`LocalToolExecutor` wraps the existing local Python behavior used by the iterative agent. It keeps local execution as the default and preserves the rule-based and LLM-decided policy logic.

`MCPToolExecutor` starts the MCP tool server with stdio, using:

```bash
python -m text2sql.agents.mcp_server --transport stdio
```

The MCP server exposes the same five tool names. The executor normalizes MCP responses into the same semantic shape that local memory and traces expect.

## Memory Contract

`WorkingMemory` and `EpisodicMemory` do not call tools directly. They observe normalized tool events after the agent loop calls `tool_executor`.

Each memory event records:

- `sample_id`
- `db_id`
- `tool_backend`
- `action`
- `tool_input`
- `observation`

This keeps memory backend-agnostic: with `memory_mode=working` or `memory_mode=episodic`, local and MCP tool observations are written through the same path.

## Validation Scope

Backend parity is checked at the tool-call level with a fixed diagnostic suite:

```bash
PYTHONPATH=src python3 scripts/compare_tool_backends.py
```

Output:

```text
results/iterative_agent/backend_parity_report.json
```

The MCP backend also has a small runtime smoke script:

```bash
PYTHONPATH=src python3 scripts/run_mcp_backend_smoke.py --limit 3 --max-steps 2
```

This is only smoke validation. It verifies MCP tool calls, memory observation writes, and SQL safety guard behavior. It is not a stratified-300 or full-dev benchmark, and it should not be reported as a performance result.
