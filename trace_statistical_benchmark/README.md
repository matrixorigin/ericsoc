# Trace Statistical Benchmark

This folder contains the first 15 deterministic queries for evaluating statistical analysis over LLM-agent traces. The queries were developed from a small set of Astra notebook execution traces and are published here as the baseline workload for a broader trace-analysis benchmark.

本文件夹包含首批15个可确定计算结果的查询，用于评估大模型分析 Agent Trace 的能力。这些查询最初基于一组 Astra Notebook 执行日志开发，是后续 Trace 统计分析 benchmark 的基础题组。

## What is included / 包含内容

| Path / 路径 | Purpose / 用途 |
|---|---|
| `reference_queries.py` | Complete Python reference implementations and task definitions / 完整的 Python 标准实现和题目定义 |
| `queries/query01.py` to `query15.py` | One command-line entry point per query / 每道题一个独立运行入口 |
| `query_catalog.json` | Machine-readable query metadata / 机器可读的查询目录 |
| `sample_data/` | Small synthetic dataset for local checks; no real Astra logs / 用于本地检查的虚构样例，不含真实 Astra 日志 |
| `docs/data_contract.md` | Input schema used by the query runner / 查询程序使用的数据结构 |
| `tests/test_queries.py` | Smoke tests for all 15 queries / 15道查询的基础测试 |

## Query coverage / 查询范围

| ID | Query / 查询内容 | Level / 难度 |
|---|---|---|
| `query01` | Count total, successful, and failed traces / 统计 Trace 总数、成功数和失败数 | Easy |
| `query02` | List failed attempts and their error details / 列出失败任务及错误信息 | Easy |
| `query03` | Find the latest attempt in each session / 找出每个 Session 的最后一次任务 | Medium |
| `query04` | Find the longest task attempt / 找出耗时最长的任务 | Easy |
| `query05` | Summarize successful and failed calls by tool / 按工具统计成功和失败调用 | Medium |
| `query06` | List individual failed tool calls / 列出具体失败的工具调用 | Medium |
| `query07` | Compare token usage across traces / 比较不同 Trace 的 Token 使用量 | Medium |
| `query08` | Calculate generation count and nearest-rank P95 latency / 计算模型调用次数和 P95 耗时 | Hard |
| `query09` | Find the five slowest tool calls / 找出最慢的五次工具调用 | Medium |
| `query10` | Compare successful and failed attempts descriptively / 描述性比较成功与失败任务 | Hard |
| `query11` | Detect a later success after each failure / 检查失败后是否出现后续成功 | Hard |
| `query12` | Rebuild the ordered timeline for every session / 重建每个 Session 的执行时间线 | Medium |
| `query13` | Audit final-output availability and length / 检查最终输出是否存在及其长度 | Easy |
| `query14` | Find notebook versions mentioned across trace content / 查找 Trace 中提到的 Notebook 版本 | Hard |
| `query15` | Audit links, roots, timestamps, and missing fields / 检查关联、根步骤、时间和缺失字段 | Hard |

## Run a query / 运行查询

Python 3.10 or newer is sufficient. The reference implementation uses only the Python standard library.

使用 Python 3.10 或更高版本即可，标准实现只依赖 Python 标准库。

From this folder, run one query against the included synthetic data:

在本文件夹中，对虚构样例运行一道查询：

```bash
python3 -m queries.query01 --data-dir sample_data
```

Run a different query by changing the module name:

修改编号即可运行其他查询：

```bash
python3 -m queries.query08 --data-dir sample_data
```

Run the complete smoke test:

运行全部基础测试：

```bash
python3 -m unittest discover -s tests -v
```

## Use another trace dataset / 使用其他 Trace 数据

Prepare `traces.jsonl` and `observations.jsonl` according to [`docs/data_contract.md`](docs/data_contract.md), then pass their parent folder to `--data-dir`.

按照 [`docs/data_contract.md`](docs/data_contract.md) 准备 `traces.jsonl` 和 `observations.jsonl`，再通过 `--data-dir` 指定其所在文件夹。

```bash
python3 -m queries.query15 --data-dir /path/to/canonical_trace_data
```

Each command prints a deterministic JSON answer. These Python results can be stored as expected answers when the same natural-language questions are evaluated with an LLM in Langfuse.

每条命令都会输出确定性的 JSON 答案。将相同的自然语言问题交给大模型和 Langfuse 运行时，可以把这些 Python 结果作为标准答案。

## Scope / 当前范围

This baseline focuses on trace status, errors, sessions, latency, tokens, tool calls, recovery, output availability, artifact references, and data quality. It is an initial workload, not a claim of full statistical coverage. Real Astra logs and platform credentials are not included.

本基线覆盖 Trace 状态、错误、Session、耗时、Token、工具调用、恢复、最终输出、文件版本和数据质量。它是第一组基础查询，不代表已经覆盖全部统计分析能力。仓库不包含真实 Astra 日志或平台密钥。
