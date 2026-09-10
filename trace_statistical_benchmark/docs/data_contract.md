# Input data contract / 输入数据约定

The query runner reads two UTF-8 JSONL files. Each line must be one JSON object.

查询程序读取两个 UTF-8 JSONL 文件，每一行是一条 JSON 对象。

## `traces.jsonl`

Required fields used by the baseline queries:

基础查询使用以下字段：

| Field | Meaning / 含义 |
|---|---|
| `trace_id` | Stable trace identifier / 稳定的 Trace 编号 |
| `session_id` | Parent session identifier / 所属 Session 编号 |
| `status` | Usually `success` or `error` / 通常为成功或失败 |
| `start_time`, `end_time` | ISO 8601 timestamps / ISO 8601 时间 |
| `duration_ms` | Trace duration in milliseconds / Trace 总耗时 |
| `model` | Model name / 模型名称 |
| `input`, `output` | Redacted task input and final output / 已脱敏的输入和最终输出 |
| `error` | Error object or `null` / 错误对象或空值 |
| `usage` | Input, output, and cache-read tokens / 输入、输出及缓存 Token |
| `metadata` | Turn, tool count, and LLM round count / 轮次、工具次数及模型调用次数 |

## `observations.jsonl`

Required fields used by the baseline queries:

基础查询使用以下字段：

| Field | Meaning / 含义 |
|---|---|
| `observation_id` | Stable observation identifier / 稳定的 Observation 编号 |
| `trace_id` | Parent trace identifier / 所属 Trace 编号 |
| `parent_observation_id` | Parent observation or `null` for the root / 父步骤，根步骤为空 |
| `type` | `span`, `generation`, or `tool` / 步骤、模型调用或工具调用 |
| `name` | Observation or tool name / 步骤或工具名称 |
| `status` | Observation status / 步骤状态 |
| `start_time`, `end_time` | ISO 8601 timestamps / ISO 8601 时间 |
| `input`, `output` | Redacted previews / 已脱敏的内容预览 |
| `metadata` | Tool index, model round, and related fields / 工具顺序、模型轮次等字段 |

Real logs are intentionally excluded from this repository. Before using private data, convert it to this contract and review all text fields for secrets and personal information.

仓库不包含真实日志。使用内部数据前，需要先转换为这个结构，并检查所有文本字段是否含有密钥或个人信息。
