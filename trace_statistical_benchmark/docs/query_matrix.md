# Thirty-Query Design Matrix / 30题设计矩阵

`Implemented` means the deterministic Python oracle is available in `reference_queries.py` or `advanced_reference_queries.py`.

`Implemented` 表示已经在 `reference_queries.py` 或 `advanced_reference_queries.py` 中提供可运行的 Python 标准实现。

| ID | Family | Query / 查询 | Main method | Status |
|---|---|---|---|---|
| 01 | Descriptive | Trace status totals / Trace状态总数 | Count by status | Implemented |
| 04 | Descriptive | Longest attempt / 最长任务 | Maximum with deterministic ties | Implemented |
| 07 | Descriptive | Token usage by trace / Trace Token分布 | Sum and maximum | Implemented |
| 08 | Descriptive | Generation count and P95 latency / 模型调用次数和P95延迟 | Nearest-rank quantile | Implemented |
| 13 | Descriptive | Final-output availability / 最终输出完整性 | Missingness and Unicode length | Implemented |
| 03 | Relational | Latest attempt per session / 每个Session最后一次任务 | Grouped temporal selection | Implemented |
| 12 | Relational | Session execution timeline / Session执行时间线 | Ordered one-to-many relation | Implemented |
| 14 | Relational | Notebook version lineage / Notebook版本关系 | Cross-field extraction | Implemented |
| 15 | Relational | Canonical data quality audit / 标准数据质量检查 | Referential and temporal integrity | Implemented |
| 16 | Relational | Task-to-answer lineage reconstruction / Task到回答的完整关系 | Multi-table lineage and orphan audit | Implemented |
| 02 | Workflow | Failed attempt details / 失败任务明细 | Conditional selection | Implemented |
| 05 | Workflow | Tool usage summary / 工具调用汇总 | Grouped status counts | Implemented |
| 06 | Workflow | Failed tool calls / 失败工具调用 | Child-record filtering | Implemented |
| 09 | Workflow | Slowest tool calls / 最慢工具调用 | Top-k with deterministic ties | Implemented |
| 11 | Workflow | Later success after failure / 失败后的后续成功 | Within-session recovery search | Implemented |
| 10 | Comparison | Success-versus-error profiles / 成功与失败任务比较 | Descriptive group comparison | Implemented |
| 17 | Comparison | Paired model accuracy / 配对模型正确率 | Contingency table and exact McNemar test | Implemented |
| 18 | Comparison | Paired model latency / 配对模型延迟 | Median paired difference and bootstrap CI | Implemented |
| 19 | Comparison | Quality-cost Pareto frontier / 质量成本前沿 | Accuracy, cost, and dominance | Implemented |
| 20 | Comparison | Model-by-difficulty interaction / 模型和题目难度交互 | Difference-in-differences with bootstrap CI | Implemented |
| 21 | Inference | Trace complexity and correctness / Trace复杂度与正确率 | Spearman correlation and permutation test | Implemented |
| 22 | Inference | Tool failure and task failure / 工具失败与任务失败 | Risk ratio and Fisher exact test | Implemented |
| 23 | Inference | Retry count and recovery / 重试次数与恢复率 | Ordered trend test with effect sizes | Implemented |
| 24 | Inference | Failure concentration by tool / 工具错误集中程度 | HHI, Gini, and bootstrap CI | Implemented |
| 25 | Inference | Performance drift and change point / 表现漂移和变化点 | Windowed estimates and corrected change-point test | Implemented |
| 26 | Annotation | Annotation validity audit / 标注有效性检查 | JSON schema, target, identity, and timestamp checks | Implemented |
| 27 | Annotation | Effective-label reconstruction / 当前有效标注还原 | Immutable revision-chain resolution | Implemented |
| 28 | Annotation | Inter-annotator agreement / 标注者一致性 | Agreement rate and Cohen/Fleiss kappa | Implemented |
| 29 | Annotation | Annotator bias on shared items / 共同样本上的标注偏差 | Paired positive-rate comparison | Implemented |
| 30 | Annotation | Revision and ingestion delay / 修改和入库延迟 | Transition matrix and latency distribution | Implemented |

## Coverage summary / 覆盖情况

| Family | Implemented | Specified | Total |
|---|---:|---:|---:|
| Descriptive distribution | 5 | 0 | 5 |
| Relational integrity | 5 | 0 | 5 |
| Workflow failure and recovery | 5 | 0 | 5 |
| Comparative performance | 5 | 0 | 5 |
| Inferential robustness | 5 | 0 | 5 |
| Annotation provenance | 5 | 0 | 5 |
| **Total** | **30** | **0** | **30** |

All 30 queries now have executable deterministic reference implementations. Advanced tests return explicit warnings when a dataset does not support the requested inference.

30题现在都有可运行的确定性标准实现。数据量不足以支持统计推断时，高级查询会明确返回警告。
