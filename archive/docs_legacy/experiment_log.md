## Step 1 Check

- BIRD dev databases loaded successfully.
- 11 SQLite databases detected.
- Standardized BIRD samples with question / gold_sql / db_id / db_path / evidence / difficulty.
- Randomly sampled 20 gold SQL queries with seed=42.
- Execution success: 20/20.
- SQL execution environment verified.


## Step 2: Naive Full-Schema Baseline

Dataset: BIRD dev, california_schools subset  
Sample size: 50  
Model: DeepSeek Chat  
Prompt: full schema + evidence + question  

### Metrics

| Metric | Value |
|---|---:|
| Gold Success | 50/50 |
| Pred Success | 43/50 |
| VSR | 86.0% |
| EX | 26.0% |

### Difficulty Breakdown

| Difficulty | Samples | VSR | EX |
|---|---:|---:|---:|
| Simple | 15 | 93.3% | 60.0% |
| Moderate | 16 | 100.0% | 25.0% |
| Challenging | 19 | 68.4% | 0.0% |

### Observations

Naive full-schema prompting can solve many simple questions, but fails sharply on moderate and challenging cases. The main gap is not SQL executability, since VSR is 86%, but semantic correctness. Typical failures include wrong column selection, imprecise filtering values, incorrect joins, and failure to reproduce complex business logic such as ranking, CASE expressions, and derived metrics.

## Step 3: Schema Linking + Schema-Aware SQL Prompting

We implemented a schema linking module using BM25 + dense retrieval + RRF to retrieve top-k relevant columns. Retrieved columns are expanded to selected tables, and the full schemas of selected tables are provided to the LLM. Retrieved columns and join keys are explicitly marked.

A schema-aware SQL prompting constraint was added to reduce column hallucination:
- use exact table and column names from the provided schema;
- do not invent normalized column names;
- wrap complex column names with backticks;
- only use a column under the table where it appears.

### Results

| Method | VSR | EX |
|---|---:|---:|
| Full-schema naive | 86.0% | 26.0% |
| Schema-linked + promptfix | 88.0% | 34.0% |

### Difficulty Breakdown

| Difficulty | Full-schema EX | Schema-linked + promptfix EX |
|---|---:|---:|
| Simple | 60.0% | 60.0% |
| Moderate | 25.0% | 50.0% |
| Challenging | 0.0% | 0.0% |

### Observation

Schema linking with schema-aware SQL constraints improves overall EX by 8 percentage points. The gain mainly comes from moderate questions, where EX improves from 25.0% to 50.0%. Simple questions remain stable, while challenging questions are still unsolved, suggesting that schema linking alone is insufficient for complex SQL reasoning involving multi-step logic, ranking, and derived metrics.