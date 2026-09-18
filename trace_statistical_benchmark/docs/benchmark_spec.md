# Trace Statistical Benchmark Specification

## 1. Purpose / 目标

This benchmark evaluates whether a language model can perform reliable statistical analysis over LLM-agent traces. It is not a collection of unrelated questions. The workload follows a systematic design: each query has a defined data scope, relational shape, statistical method, output contract, deterministic Python oracle, and scoring policy.

本 benchmark 用来评估大模型能否可靠地分析 Agent Trace。它不是一组互不相关的问题。每道查询都必须明确数据范围、关系结构、统计方法、输出格式、Python 标准答案和评分规则。

The design follows the main idea of the Wisconsin database benchmark: vary known dimensions in a controlled way so that a result can be attributed to a specific capability rather than to an accidental difference between questions.

设计参考 Wisconsin database benchmark 的核心思路：有控制地改变已知因素，从而判断结果来自哪项能力，而不是来自题目之间无法解释的偶然差异。

## 2. Evaluation unit / 评估单位

A benchmark item contains:

每道 benchmark 题目包含：

1. A natural-language query shown to the model. / 交给模型的自然语言问题。
2. A frozen snapshot of canonical trace data. / 固定版本的标准 Trace 数据。
3. A machine-readable output schema. / 机器可读的输出结构。
4. A deterministic Python reference implementation. / 确定性的 Python 标准实现。
5. An expected answer generated from the frozen snapshot. / 根据固定数据生成的标准答案。
6. A scoring policy for values, sets, order, missing fields, and evidence IDs. / 对数值、集合、顺序、缺失字段和证据 ID 的评分规则。

One model run must answer the query without seeing the reference code or expected answer.

模型运行时不能看到 Python 标准实现或标准答案。

## 3. Canonical relational model / 标准关系模型

The public benchmark uses normalized logical tables even if the storage format is JSONL or a Langfuse export.

即使底层文件使用 JSONL 或 Langfuse 导出格式，benchmark 也按照下面的逻辑表理解数据。

| Table | Primary key | Main relationships and fields |
|---|---|---|
| `models` | `model_id` | provider, model name, version, configuration |
| `tasks` | `task_id` | task family, difficulty, prompt version, reference outcome |
| `sessions` | `session_id` | user or cohort, start/end time, environment |
| `traces` | `trace_id` | links to session, task, and model; attempt number, status, time, usage, cost |
| `observations` | `observation_id` | links to trace and parent observation; type, status, time |
| `generations` | `observation_id` | model input/output, token usage, latency |
| `tool_calls` | `observation_id` | tool name, arguments summary, result summary, error state |
| `model_answers` | `answer_id` | links task and trace; parsed answer, correctness, grading version |
| `annotations` | `annotation_id` | target ID, JSON label, annotator, label time, ingest time, schema version |
| `annotation_history` | `annotation_id` | superseded/retracted relationships and immutable revisions |

Required relationships:

- One session can contain multiple traces.
- One task can be attempted by multiple models and multiple times.
- One trace belongs to exactly one task, model, and session.
- One trace can contain a tree of observations.
- A generation or tool call is also an observation.
- An annotation targets one trace or observation and never silently overwrites history.

必要关系：一个 Session 可以包含多条 Trace；同一个 Task 可以由多个模型多次执行；一条 Trace 必须属于一个 Task、Model 和 Session；Observation 构成树；标注必须保留历史，不能静默覆盖。

## 4. Query families / 查询类别

The complete workload contains 30 queries: six capability families with five queries each.

完整题组包含30道查询，由六类能力组成，每类五题。

| Family | Capability under test |
|---|---|
| `descriptive_distribution` | Counts, distributions, quantiles, and output availability |
| `relational_integrity` | Hierarchical joins, lineage, ordering, and data integrity |
| `workflow_failure` | Tool behavior, failures, retries, and recovery |
| `comparative_performance` | Fair comparison of statuses or models on matched data |
| `inferential_robustness` | Association, uncertainty, trend, concentration, and drift |
| `annotation_provenance` | JSON labels, history, annotator agreement, bias, and timing |

All 30 queries have executable Python reference implementations. Queries 16-30 extend the workload with relational, inferential, and annotation-centered tasks.

30题现在都有可执行的 Python 标准实现。第16至30题补充关系查询、统计推断和标注分析。

## 5. Difficulty model / 难度定义

Difficulty is a vector, not a subjective label. Each query is described on four axes from 1 to 5.

难度不是凭感觉写成“简单或困难”，而是由四个1到5级的维度组成。

| Axis | Level 1 | Level 3 | Level 5 |
|---|---|---|---|
| Relational depth `R` | One table | Two or three linked tables | Hierarchy, history, or four-plus tables |
| Statistical depth `S` | Count or direct lookup | Distribution or paired summary | Inference, resampling, drift, or reliability |
| Data complexity `D` | Clean and balanced | Skew, ties, or moderate missingness | Heavy tails, imbalance, corruption, or delayed records |
| Temporal depth `T` | No time relation | Ordered attempts or windows | Recovery, revisions, drift, or censoring |

The display label is derived from the vector:

- Easy: total score 4-7.
- Medium: total score 8-11.
- Hard: total score 12-15.
- Advanced: total score 16-20.

This rule is applied to generated benchmark variants. The original baseline labels are retained for backward compatibility.

该规则用于后续生成的数据版本。现有15题原来的难度标签暂时保留，以避免破坏已经上传的 Langfuse Dataset。

## 6. Controlled data generation / 可控数据生成

Real Astra traces provide realistic field names, event order, and approximate distributions. They do not provide enough observations or known causal truth for a research benchmark. The benchmark generator therefore creates synthetic records from explicit parameters and records the true parameters in a private manifest.

真实 Astra Trace 用来提供合理的数据结构、顺序和分布范围，但样本量不足，也没有完全已知的真实机制。因此正式 benchmark 使用参数明确的模拟数据，并在独立清单中保存真实参数。

Controlled factors include:

| Factor | Example levels |
|---|---|
| Corpus size | 100, 1,000, 10,000, 100,000 traces |
| Model count | 2, 4, 8 models |
| Task difficulty | balanced, easy-heavy, hard-heavy |
| Attempts per task | one, fixed repeats, uneven repeats |
| Trace depth | shallow, medium, deep observation trees |
| Tool failure rate | 0%, 5%, 20% |
| Retry and recovery | none, moderate, high |
| Latency distribution | light-tailed, log-normal, extreme outliers |
| Missingness | none, MCAR, group-dependent missingness |
| Model effect | equal, small difference, large difference |
| Time behavior | stable, gradual drift, abrupt change point |
| Annotator behavior | high agreement, prevalence imbalance, systematic bias |
| Annotation history | no revisions, delayed ingest, conflicting revisions |

Every generated corpus must store its random seed, generator version, parameter file, schema version, and data checksum. Text can be templated or generated separately, but numeric ground truth must come from deterministic code rather than an LLM.

每份模拟数据必须保存随机种子、生成器版本、参数、Schema 版本和校验值。文本可以使用模板或单独生成，但数值真值不能由大模型决定。

## 7. Reference answers / 标准答案

Each query has one Python oracle. The oracle must:

- Read only the frozen benchmark snapshot.
- Validate required fields before calculation.
- Use a documented statistical definition.
- Resolve ties and missing values deterministically.
- Return JSON that follows the declared output schema.
- Include sample size and evidence IDs when they are part of the question.
- Record warnings when the data do not support a requested conclusion.

每道题只有一个 Python 标准实现。它必须检查字段、固定缺失值和并列值处理方式、返回规定的 JSON，并在样本不足时明确给出警告。

Reference implementations are tested on hand-calculated fixtures, edge cases, and at least one generated corpus with known parameters.

标准实现必须通过人工可计算的小样例、边界情况和至少一份已知参数的模拟数据测试。

## 8. Statistical rules / 统计规则

The benchmark uses the following rules unless a query overrides them:

- Paired model comparisons use only tasks attempted by both models.
- Binary paired outcomes use an exact McNemar test.
- Paired latency comparisons report a median paired difference and a seeded bootstrap confidence interval.
- Correlations use Spearman rank correlation when strong skew or outliers are present.
- Multiple comparisons state and apply their correction method.
- Effect sizes and confidence intervals are reported with p-values.
- Missing data handling is declared; missing values are not silently converted to zero.
- Correlation or temporal order alone must not be described as causation.

默认规则包括：模型比较使用相同题目的配对样本；二分类配对结果使用精确 McNemar 检验；延迟使用配对中位数差和固定随机种子的 bootstrap 区间；偏态数据优先使用 Spearman 相关；显著性结果同时报告效应大小和置信区间；缺失值不能直接当作零；相关性不能解释成因果关系。

## 9. Annotation contract / 标注约定

Annotations are immutable events. A correction creates a new annotation that points to the previous one.

标注是不可变事件。修改标注时新增一条记录，并指向旧记录。

Required fields:

```json
{
  "annotation_id": "ann-...",
  "target_type": "observation",
  "target_id": "obs-...",
  "label_schema_version": "1.0",
  "label_json": {"answer_correct": true, "error_type": null},
  "annotator_id": "reviewer-...",
  "labeled_at": "2026-09-14T12:00:00Z",
  "ingested_at": "2026-09-14T12:00:05Z",
  "source": "human",
  "supersedes_annotation_id": null,
  "retracted": false
}
```

The benchmark keeps `labeled_at` and `ingested_at` separately. `annotator_id` may be pseudonymous, but it must be stable within an evaluation. JSON labels are validated against a versioned schema.

标注发生时间和入库时间必须分开保存。标注者可以匿名化，但同一实验中编号必须稳定。JSON 标签必须按照有版本的 Schema 检查。

## 10. Answer contract and scoring / 答案格式与评分

Scoring is decomposed so that a formatting error is not confused with a statistical error.

评分需要拆开，不能把格式错误和统计错误混为一谈。

| Score | Meaning |
|---|---|
| `valid_json` | Output can be parsed and follows the required top-level type |
| `schema_accuracy` | Required fields and field types are present |
| `factual_field_accuracy` | Counts, IDs, categories, and values are correct |
| `numeric_accuracy` | Numeric values are within declared absolute or relative tolerance |
| `method_accuracy` | Required statistical method and denominator are correct |
| `evidence_accuracy` | Returned trace or observation IDs support the answer |
| `interpretation_safety` | No unsupported significance or causal claim |
| `latency`, `tokens`, `cost` | Efficiency metrics, reported separately from correctness |

Rows that represent sets are order-insensitive. Timelines and ranked results are order-sensitive. Tolerances are declared per field. An overall score is not reported without its component scores.

集合类结果不比较顺序；时间线和排名必须比较顺序。每个数值字段单独规定误差。总分必须与各分项一起报告。

## 11. Langfuse mapping / Langfuse 对应关系

| Benchmark object | Langfuse object |
|---|---|
| One agent attempt | Trace |
| One model call or tool call | Observation |
| One statistical query plus its data scope | Dataset item |
| One model evaluated on a dataset version | Dataset run / experiment |
| One scoring component | Score |
| Human review event | Observation- or trace-linked annotation/score plus canonical annotation record |

Langfuse is the experiment and observability layer. Python remains the source of truth for deterministic answers and statistical calculations.

Langfuse 负责保存实验、Trace 和评分；Python 仍然是标准答案和统计计算的真值来源。

## 12. Fair comparison protocol / 公平比较规则

All compared models receive the same dataset snapshot, question text, evidence serialization, output schema, and retry policy. Model-specific adapters may only address API format differences. They must not add task hints.

不同模型必须使用同一份数据、题目、证据格式、输出Schema和重试规则。模型适配代码只能处理API差异，不能额外提供解题提示。

Each reported result includes model version, provider, temperature, seed when supported, date, prompt version, data checksum, number of repeats, failures, latency, tokens, and cost. Main model comparisons are paired by dataset item and repeat number.

每份结果必须记录模型版本、参数、日期、Prompt版本、数据校验值、重复次数、失败、延迟、Token和成本。主要比较必须按题目和重复轮次配对。

## 13. Acceptance criteria / 完成标准

Version 1 of the full benchmark is complete only when:

- The canonical schema and annotation schema are versioned.
- All 30 queries have executable Python oracles.
- Every query has an output schema and scoring policy.
- The six families contain exactly five queries each.
- Oracle tests cover normal, missing, tied, and corrupted cases where relevant.
- At least three controlled corpus sizes can be generated reproducibly.
- At least two models are evaluated on the same frozen dataset version.
- Langfuse shows the dataset items, model runs, traces, and component scores.
- The release contains no credentials or private raw logs.

只有当30题全部有Python实现、输出Schema、评分规则和测试，并且模拟数据可复现、模型实验公平、Langfuse记录完整且不泄露内部日志时，才能称为完整的v1 benchmark。

## 14. Design references / 设计参考

The specification uses ideas from the following benchmark families while keeping a distinct focus on statistical analysis of agent traces:

- [Wisconsin Benchmark: Benchmarking Database Systems, A Systematic Approach](https://www.vldb.org/conf/1983/P008.PDF): controlled relations, query families, selectivity, and repeatable comparisons.
- [DataBench](https://aclanthology.org/2024.lrec-main.1179/): natural-language questions over real tabular datasets.
- [QRData](https://arxiv.org/abs/2402.17644): statistical and causal reasoning with attached data.
- [StatQA](https://papers.nips.cc/paper_files/paper/2024/file/729786203d330da046dd8091c2d92a66-Paper-Datasets_and_Benchmarks_Track.pdf): statistical method selection and applicability.
- [TRAIL](https://arxiv.org/abs/2505.08638): trace-native error classification and localization.
- [Insights Generator](https://arxiv.org/abs/2605.21347): corpus-level trace diagnosis and cohort comparison.
- [LogNLQ-Bench](https://arxiv.org/abs/2607.03884): execution-verified natural-language queries over log data.
- [UDA-Bench](https://arxiv.org/abs/2510.27119): operator, selectivity, and complexity coverage over a relational ground-truth view.

These references motivate individual design choices; they do not define the benchmark's ground truth or scoring rules.

这些工作分别提供了受控查询、表格问答、统计推理、Trace错误定位、日志查询和复杂度设计方面的参考。本 benchmark 的标准答案和评分规则仍由本项目独立定义。
