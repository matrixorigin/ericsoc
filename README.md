# Chicago TNP Analytics and Event Simulation

# Chicago TNP 分析与赛事模拟

This repository contains an end-to-end analytics workflow for the City of Chicago Transportation Network Provider (TNP) trip data. The workflow starts with full CSV validation and MatrixOne tables, builds hourly H3 demand models, tests event detection, and ends with a 20,000-run business simulator for large events near Soldier Field. 本仓库包含一套完整的 Chicago TNP 出行数据分析流程。流程从全量 CSV 验证和 MatrixOne 数据表开始，继续完成小时级 H3 客流建模和赛事识别，最后建立 Soldier Field 大型活动的 2 万次业务模拟器。

The public data covers 2022-2024 and contains 243,479,296 completed trips after loading. Large SQL work stays in MatrixOne. Python receives smaller aggregates for analysis and modeling. 公开数据覆盖 2022-2024，载入后共有 243,479,296 条已完成行程。大规模 SQL 计算保留在 MatrixOne 中，Python 只读取较小的聚合结果用于分析和建模。

## Key validated results / 已验证的关键结果

| Area / 方向                             | Recorded result / 已记录结果                                                                                                                                                                        |
|------------------------------------|------------------------------------|
| Full data foundation / 全量数据基础     | 243,479,296 clean and H3-enriched trip rows. / 243,479,296 条清洗并加入 H3 的行程。                                                                                                                 |
| Next-hour H3 demand / 下一小时 H3 客流  | MAE 12.66 pickups and R-squared 0.9412 on 2024. / 2024 测试集 MAE 12.66、R² 0.9412。                                                                                                                |
| Recorded-tip model / 记录小费模型       | Expected-tip MAE improved from about \$1.58 to \$1.46 in the portable model package. / 可移植模型将期望小费 MAE 从约 1.58 美元降至 1.46 美元。                                                      |
| Bears event prediction / Bears 赛事预测 | Known-schedule event model improved pickup MAE by 44.4% and dropoff MAE by 68.5% over a normal-day baseline. / 已知赛程时，赛事模型相对普通日基线将 pickup MAE 改善 44.4%，dropoff MAE 改善 68.5%。 |
| Blind event screening / 赛事盲测筛选    | Top 8 found 5 of 8 games; Top 20 found all 8 games. / Top 8 找到 8 场中的 5 场，Top 20 找到全部 8 场。                                                                                              |
| Event demand generator / 赛事需求生成器 | Validation on eight unseen 2024 games: pickup WAPE 19.15%, dropoff WAPE 22.78%. / 在 8 场未参与训练的 2024 比赛上，pickup WAPE 19.15%，dropoff WAPE 22.78%。                                        |
| Go deployment / Go 部署                 | Python and Go ONNX predictions match within 0.001 on fixed parity cases. / 固定测试样本中，Python 与 Go ONNX 预测差异小于 0.001。                                                                   |

These results are planning and research results. Completed trips are not the same as all ride requests, and simulator profit is contribution profit under stated assumptions rather than audited company net profit. 这些结果用于规划和研究。已完成行程不等同于全部叫车请求；模拟器利润是基于明确假设计算的贡献利润，不是企业审计后的净利润。

## Repository guide / 仓库说明

Run the numbered notebooks in order when rebuilding the full workflow. Notebooks 03-06 are comparison or deployment branches and can be rerun after Notebook 01 has validated the MatrixOne tables. 完整重建时请按编号运行。Notebook 03-06 属于模型对比或部署分支，在 Notebook 01 验证 MatrixOne 数据表后也可以单独运行。

1.  [`01_data_ingestion_validation_and_h3_preparation.ipynb`](01_data_ingestion_validation_and_h3_preparation.ipynb) Validates raw schemas, MatrixOne table lineage, clean rows, H3 coverage, and the shared train/test split. / 验证原始 schema、MatrixOne 表关系、清洗行数、H3 覆盖和统一训练测试切分。
2.  [`02_hourly_h3_demand_forecasting_and_spatial_dynamics.ipynb`](02_hourly_h3_demand_forecasting_and_spatial_dynamics.ipynb) Builds hourly pickup/dropoff data, anomaly sensitivity, multi-horizon forecasts, supply-pressure proxies, and neighbor spread. / 构建小时级上下车数据、异常敏感度、多跨度预测、供给压力代理和周边扩散分析。
3.  [`03_two_stage_tip_prediction_and_llm_tree_comparison.ipynb`](03_two_stage_tip_prediction_and_llm_tree_comparison.ipynb) Predicts recorded-tip probability and positive amount, then compares LLM and tree models on the same yes/no cases. / 预测记录小费概率和正小费金额，再在相同样本上比较大模型与树模型的有无小费判断。
4.  [`04_llm_peak_continuation_benchmark.ipynb`](04_llm_peak_continuation_benchmark.ipynb) Tests whether a confirmed peak remains after 1, 3, or 6 hours using simple baselines, GBM, and an OpenAI model. / 使用简单基线、GBM 和 OpenAI 模型判断已确认高峰在 1、3、6 小时后是否继续。
5.  [`05_chronos2_vs_lightgbm_forecasting_benchmark.ipynb`](05_chronos2_vs_lightgbm_forecasting_benchmark.ipynb) Compares Chronos-2 and LightGBM on the same 24-hour forecast windows. / 在相同的 24 小时预测窗口上比较 Chronos-2 与 LightGBM。
6.  [`06_h3_next_hour_onnx_go_deployment_validation.ipynb`](06_h3_next_hour_onnx_go_deployment_validation.ipynb) Inspects the deployable ONNX package and verifies Python/Go prediction parity. / 检查可部署 ONNX 模型包并验证 Python 与 Go 的预测一致性。
7.  [`07_bears_event_sensitivity_and_2024_blind_detection.ipynb`](07_bears_event_sensitivity_and_2024_blind_detection.ipynb) Learns Bears home-game pickup/dropoff curves, predicts known 2024 games, and performs a schedule-free blind screen. / 学习 Bears 主场比赛上下车曲线，预测已知的 2024 比赛，并进行不提供赛程的盲测筛选。
8.  [`08_event_business_monte_carlo_simulator.ipynb`](08_event_business_monte_carlo_simulator.ipynb) Validates an event-demand generator and compares four fleet sizes with three pricing strategies over 20,000 two-day scenarios. / 验证赛事需求生成器，并在 2 万个连续两天场景中比较四种车队规模和三种定价策略。

## Environment / 环境

### Main notebooks / 主 Notebook

-   Python 3.12 is the validated main environment. / 主环境已在 Python 3.12 验证。
-   Notebook 05 uses Python 3.11 because of its Chronos-2 runtime. / Notebook 05 因 Chronos-2 运行环境使用 Python 3.11。
-   MatrixOne must expose `chicago_tnp.unified_trips_clean` and `chicago_tnp.unified_trips_h3_res9` for the full analysis. / 全量分析需要 MatrixOne 提供这两张表。
-   Go 1.24 or newer is required only for the Go deployment checks. / 只有 Go 部署验证需要 Go 1.24 或更高版本。

Install the common Python environment: 安装通用 Python 环境：

``` bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a separate Python 3.11 environment for Notebook 05: 为 Notebook 05 创建独立的 Python 3.11 环境：

``` bash
python3.11 -m venv .venv-chronos
source .venv-chronos/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-chronos.txt
```

### Shared configuration / 统一配置

``` bash
export CHICAGO_TNP_PROJECT_DIR="/path/to/chicago_tnp_full"
export CHICAGO_TNP_RAW_DIR="/path/to/chicago_tnp_full/raw/full"
export CHICAGO_TNP_REPO_DIR="/path/to/ericsoc"
export MATRIXONE_HOST="127.0.0.1"
export MATRIXONE_PORT="6001"
export MATRIXONE_USER="root"
export MATRIXONE_DATABASE="chicago_tnp"
export MATRIXONE_PASSWORD="your-password"
```

Notebook 03 and Notebook 04 call the OpenAI API. They ask for `OPENAI_API_KEY` at runtime and do not store the key in the Notebook or output files. Review the configured sample size and model before running these paid tests. Notebook 03 和 Notebook 04 会调用 OpenAI API。它们在运行时读取 `OPENAI_API_KEY`，不会将 key 保存到 Notebook 或输出文件。运行付费测试前请先检查样本数量和模型配置。

## Reproduction paths / 复现方式

**Full analytical rebuild / 完整分析重建** Start MatrixOne, prepare the two source CSV files, set the environment variables, and run Notebooks 01-08 in order. Large intermediate results are cached under `CHICAGO_TNP_PROJECT_DIR`. 启动 MatrixOne，准备两个源 CSV，设置环境变量，然后按顺序运行 Notebook 01-08。大型中间结果会缓存到 `CHICAGO_TNP_PROJECT_DIR`。

**Model inference only / 只运行模型推理** MatrixOne and the raw CSV files are not required. Each model folder includes ONNX files, a feature contract, prepared JSON examples, a Go runner, and a local parity test. Read the model folder's `README.md` before supplying new inputs. 不需要 MatrixOne 和原始 CSV。每个模型文件夹都包含 ONNX 文件、特征约定、已准备的 JSON 示例、Go 推理程序和本地一致性测试。使用新输入前请先阅读模型文件夹中的 `README.md`。

## Repository validation / 仓库自检

Run the source-level repository check before publishing: 发布前运行源码级仓库检查：

``` bash
python tools/validate_repository.py
```

Run both Go parity tests: 运行两个 Go 一致性测试：

``` bash
./model_h3_next_hour_onnx_go_deployment/run_go_test.sh
./model_two_stage_tip_onnx_go_deployment/run_go_test.sh
```

The repository check validates Notebook JSON, Python syntax, empty committed outputs, portable paths, model manifests, schemas, and required deliverables. It does not replace a full MatrixOne rerun because the full source tables are not stored in Git. 仓库自检会验证 Notebook JSON、Python 语法、Git 中是否清空运行输出、路径可移植性、模型清单、schema 和必要交付物。由于 Git 中不保存全量源表，它不能替代完整的 MatrixOne 重跑。

## Output and data policy / 输出与数据规则

Notebook outputs are intentionally empty in Git so the repository does not contain machine-specific logs, database credentials, or large generated files. Runtime outputs and caches are written under `CHICAGO_TNP_PROJECT_DIR`. Raw CSV files, MatrixOne data directories, API keys, and local environment files must not be committed. Git 中有意清空 Notebook 运行输出，避免提交机器相关日志、数据库密码或大型生成文件。运行产物和缓存写入 `CHICAGO_TNP_PROJECT_DIR`。原始 CSV、MatrixOne 数据目录、API key 和本地环境文件不得提交。
