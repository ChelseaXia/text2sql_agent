# Error Analysis

This document summarizes the final error-analysis takeaways from the frozen saved runs in this repository. It does not introduce any new experiments, and it does not claim full BIRD benchmark coverage.

## Scope

All conclusions below are based on:

- BIRD dev `california_schools`
- a fixed `50`-sample subset
- saved result files already present in `results/`

Difficulty distribution of the current subset:

- `simple = 15`
- `moderate = 16`
- `challenging = 19`

The main result of the project is still:

- Day5.5 strict execution repair: `EX=36%`, `VSR=100%`

The controlled agent is discussed as a trace-based implementation close to Day5.5, and the autonomous ReAct agent is treated as a negative ablation.

Model-setting note:

- core generation and repair pipelines use the repository DeepSeek client setting
- in code, the core pipeline default is `deepseek-chat`
- later agent experiments use DeepSeek function-calling settings
- this project does not claim cross-model robustness

## Final Quantitative Snapshot

| Method | EX | VSR | Interpretation |
| --- | ---: | ---: | --- |
| Day2 naive full schema | 26% | 86% | weak baseline |
| Day3 schema-linked + promptfix | 34% | 88% | schema retrieval helps |
| Day5.5 strict execution repair | 36% | 100% | main result |
| Controlled agent trace implementation | 32% | 98% | close but not exact reproduction |
| Few-shot retrieval | 28% | 100% | negative ablation |
| DDL schema serialization | 28% | 82% | negative ablation |
| Autonomous ReAct agent | 12% | 88% | negative ablation |

Pipeline headline:

- EX improves from `26%` to `34%` to `36%`
- this is a `38.5%` relative EX improvement from the naive baseline to the final strict-repair pipeline
- VSR improves from `88%` to `100%` after strict repair

## 1. Schema Linking Helps Reduce Column Hallucination

The first major improvement comes from moving away from the naive full-schema baseline.

- Day2 naive full schema: `EX=26%`, `VSR=86%`
- Day3 schema-linked + promptfix: `EX=34%`, `VSR=88%`

This suggests that the hybrid schema linker improves relevance selection enough to reduce common failure modes such as:

- joining unnecessary tables,
- using semantically adjacent but wrong columns,
- and overloading the prompt with irrelevant schema context.

The gain is larger in EX than in VSR, which is consistent with the idea that schema retrieval helps semantic correctness more than mere executability.

## 2. Execution Repair Mainly Improves VSR

The strongest pipeline result is the Day5.5 strict execution-repair run:

- Day3 schema-linked + promptfix: `EX=34%`, `VSR=88%`
- Day5.5 strict execution repair: `EX=36%`, `VSR=100%`

The main effect is a full lift in executability from `88%` to `100%`. EX also improves, but only modestly. This indicates that execution-guided repair is especially effective at fixing:

- syntax failures,
- schema-reference mistakes,
- and other directly executable error patterns.

It is less effective at correcting SQL that is executable but answers the wrong question.

## 3. Few-Shot Retrieval Caused Semantic Negative Transfer

Few-shot retrieval achieved:

- `EX=28%`
- `VSR=100%`

This pattern is important. The model remained highly executable, but semantic accuracy dropped below the schema-linked baseline and below the strict repair pipeline. The likely interpretation is semantic negative transfer:

- retrieved examples can encourage fluent SQL generation,
- but the query pattern transferred from the retrieved example is not always the correct one for the target question.

In other words, example retrieval improved form but often hurt task-specific meaning.

## 4. DDL Schema Serialization Degraded Performance

The DDL serialization ablation reached:

- `EX=28%`
- `VSR=82%`

This underperformed both the schema-linked promptfix setting and the strict repair pipeline. The result suggests that, in this project, raw DDL-style schema presentation was less useful than the linked-schema format that emphasizes:

- relevant tables,
- relevant columns,
- join keys,
- and localized evidence.

The main failure pattern here is not just semantic error but also lower executability.

## 5. When Does Agency Hurt?

The clearest agent comparison in the repo is:

- Controlled execution-repair agent: `EX=32%`, `VSR=98%`
- Autonomous ReAct agent: `EX=12%`, `VSR=88%`

Finding:

- free-form tool calling caused over-exploration and finalization instability
- this supports using a controlled graph when task structure is known

## 6. Autonomous ReAct Tool Calling Was a Negative Ablation

The autonomous ReAct agent produced:

- `EX=12%`
- `VSR=88%`

This is much worse than the main pipeline and much worse than the controlled agent. It is retained as a negative ablation because it highlights an important lesson: more tool freedom did not produce better task performance here.

Observed failure themes from the saved traces:

- over-exploration,
- unstable finalization,
- high dependence on fallback behavior,
- and many executable exploratory SQLs that were still semantically wrong.

The result should not be interpreted as evidence that ReAct-style systems are generally bad. It only shows that this specific autonomous formulation underperformed on this project setup.

## 7. Controlled Agent Is Close to Day5.5, but Not the Canonical Metric

The controlled execution-repair agent produced:

- `EX=32%`
- `VSR=98%`

This is close to Day5.5 but not identical. The saved comparison result shows:

- `48/50` EX outcomes matched with Day5.5

This makes the controlled agent useful as an interpretable trace implementation, but it should not replace the canonical Day5.5 metric run in the final write-up.

The right interpretation is:

- Day5.5 strict execution repair is the main result.
- The controlled agent is a near-match implementation for trace analysis.

The saved Day5.5 run remains the canonical metric because `controlled_agent` is a rerun-based trace implementation; minor differences can arise from LLM sampling variance even at `temperature=0`.

## 8. Why Controlled Agent Is Not the Canonical Metric

The controlled agent is intentionally documented as a trace implementation rather than the main quantitative result.

- `controlled_agent` matches Day5.5 on `48/50` EX outcomes.
- The remaining gap is consistent with rerun variance and initial SQL mismatch rather than a different intended algorithm.
- Therefore the saved Day5.5 run remains the main quantitative result reported by the project.

## 9. Challenging-Tier Queries Remain the Main Bottleneck

Even the strongest settings in the repository do not fully solve the hardest questions. The remaining gap appears to come less from syntax and more from semantic planning problems such as:

- composing multi-stage reasoning in one SQL query,
- choosing the right aggregation or ranking logic,
- preserving all requested output attributes,
- and matching the exact semantics of enterprise-style analytical questions.

This is why VSR can become very high while EX remains substantially lower.

## 10. Overall Takeaway

The project’s final evidence supports a simple conclusion:

- schema retrieval is necessary,
- prompt fixing helps,
- execution-guided repair is the most reliable improvement,
- and unconstrained autonomous tool calling is not automatically beneficial.

For this repository, the best practical recommendation is still the saved Day5.5 strict execution-repair pipeline, with the controlled agent serving as an interpretable implementation and the autonomous ReAct agent serving as a negative ablation.
