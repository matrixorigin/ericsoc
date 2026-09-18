# Bundled Benchmark Data

`simulation/` contains the complete synthetic corpus used to exercise and validate the 30 statistical trace queries.

The corpus was generated with a fixed seed and contains exactly 10,000 traces from 4,430 shared tasks evaluated under two system labels:

- `Astra`: Matrix Origin's Astra agent system label. This is not OpenAI Astra.
- `Codex`: OpenAI Codex system label.

The labels make the paired benchmark examples easy to read. The underlying latency, quality, cost, retry, drift, and annotation behavior is simulated. It is not production telemetry and is not a real performance comparison between the products.

Important files:

- `simulation/canonical_data/`: normalized models, tasks, sessions, traces, observations, generations, tool calls, answers, and annotation events;
- `simulation/reference_answers/`: deterministic JSON answers for Query 01-30;
- `simulation/run_summary.json`: generator parameters, record counts, and query execution summary;
- `simulation/validation_report.json`: canonical relationship checks;
- `simulation/statistical_validation/`: checks that every query recovers its programmed effect, null relationship, range, or structural rule.

Regenerate the same corpus from the repository root:

```bash
python simulation/run_profile.py --profile simulation --output-root benchmark_data --replace
python validation/validate_statistical_truth.py
```
