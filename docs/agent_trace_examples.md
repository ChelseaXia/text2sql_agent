# Agent Trace Examples

Tool-calling traces on the local BIRD dev subset.

## Sample 0

- Difficulty: `challenging`
- EX: `False`
- Final execution success: `False`
- Final SQL source: `no_finish`
- Question: For the school with the highest free meal rate in Alameda County, what are its characteristics including whether it's a charter school, what grades it serves, its SAT performance level, and how much its free meal rate deviates from the county average?

Final SQL:
```sql
-- no final sql --
```

Trace summary:
- Step 1: `retrieve_schema`
- Step 2: `timeout`
- Step 3: `no_finish`

