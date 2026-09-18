# Input data contract / 输入数据约定

Schema v2 is the research contract. It separates repeated entities so that the benchmark can compare models on the same task, rebuild a full agent path, and keep every annotation revision.

Schema v2 是正式研究约定。它把重复信息拆成独立表，使 benchmark 可以比较同一任务上的不同模型、重建完整 Agent 路径，并保留每次标注修改。

## Canonical schema v2 / 标准 Schema v2

| Table | Main purpose / 主要用途 |
|---|---|
| `models.jsonl` | One row per model configuration / 每个模型配置一行 |
| `tasks.jsonl` | Stable question or task shared across model attempts / 可供多个模型共同作答的稳定任务 |
| `sessions.jsonl` | User- or workflow-level container / 用户或工作流层级容器 |
| `traces.jsonl` | One end-to-end attempt / 一次完整任务尝试 |
| `observations.jsonl` | Ordered spans, generations, tools, and events / 步骤、模型调用、工具和事件 |
| `generations.jsonl` | Model-call details and token use / 模型调用和 Token 信息 |
| `tool_calls.jsonl` | Tool choice, input, output, status, and latency / 工具选择、输入输出、状态和耗时 |
| `model_answers.jsonl` | Final answers linked to tasks and models / 与任务、模型关联的最终答案 |
| `annotations.jsonl` | Immutable label events and revision links / 不可覆盖的标注事件和修改关系 |

The exact table list and primary keys are in `schemas/schema_manifest.json`. Logical row definitions are in `schemas/canonical_bundle.schema.json`.

准确表清单和主键位于 `schemas/schema_manifest.json`，各类记录的结构位于 `schemas/canonical_bundle.schema.json`。

The stable `task_id` is essential for fair comparisons. Two traces may only be treated as answers to the same task when a source adapter provides or verifies the same task identity. The baseline converter marks its generated task mapping as unresolved instead of guessing.

稳定的 `task_id` 是公平比较的基础。只有来源转换程序明确提供或验证了相同任务编号，两个 Trace 才能被视为同题回答。基线转换程序不会猜测，而是将任务映射标记为未解决。

## Annotation history / 标注历史

Each annotation row is an event. A correction creates a new row and points `supersedes_annotation_id` to the earlier row. Retraction also creates a new event with `retracted=true`; old evidence is never deleted.

每条标注都是一个事件。修改标注时新增一行，并使用 `supersedes_annotation_id` 指向旧版本。撤回也新增一条 `retracted=true` 的记录，不删除历史证据。

`labeled_at` is when the decision was made. `ingested_at` is when the record entered the canonical store. Their difference measures ingestion delay. `label_json` keeps the full custom label, while common fields can also be sent to Langfuse as separate typed scores for filtering.

`labeled_at` 是做出标注的时间，`ingested_at` 是记录进入标准数据层的时间，两者之差就是入库延迟。完整自定义标签存入 `label_json`，常用字段还可以作为独立 Score 上传 Langfuse，方便筛选。

## Compact baseline compatibility / 精简基线兼容

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

Real logs are intentionally excluded from this folder. Before using private data, convert it to schema v2 and review all text fields for secrets and personal information.

仓库不包含真实日志。使用内部数据前，需要先转换为这个结构，并检查所有文本字段是否含有密钥或个人信息。
