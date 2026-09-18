# Maintainer Handoff

## Release Boundary

The public package contains the reusable benchmark code, a complete seeded simulation corpus, small synthetic fixtures, source adapters, optional Langfuse tools, and regression tests. It deliberately excludes:

- real Astra and Codex conversation logs;
- API credentials and local environment files;
- personal annotation exports and user identifiers;
- internal reports, presentation scripts, and screenshots;
- local virtual environments and caches.

## First Validation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python validation/validate_benchmark_design.py
python simulation/run_profile.py --profile smoke --replace
python validation/validate_statistical_truth.py
```

Expected outcome:

- all unit tests pass;
- the design validator reports 30 implemented queries and five queries in each family;
- the smoke generator validates all relationships;
- all 30 queries finish successfully.
- the bundled 10,000-trace simulation corpus passes all 30 statistical-truth checks.

## Main Extension Points

### Add a source adapter

Map source records to the canonical tables in `schemas/schema_manifest.json`. Preserve stable IDs, event order, observation parents, model identity, tool outcomes, timestamps, token usage, and errors. Add a synthetic test fixture; do not commit a real user log.

### Add or change a query

Update `query_catalog.json`, the relevant reference implementation, the independent query entry point, the query matrix, and tests together. State the denominator, missing-data rule, tie handling, statistical method, output schema, and evidence IDs.

### Add a simulation factor

Record the parameter in the generated manifest and private ground-truth file. Use a fixed seed and add a validation rule proving that the reference query detects the programmed effect or preserves a programmed null.

### Run a model benchmark

The current package provides questions, evidence data, reference answers, and scoring definitions. It does not ship a provider-specific LLM execution harness. A future runner should give every compared model the same dataset snapshot, query, evidence serialization, output schema, retry policy, and scoring version.

## Public Release Checklist

1. Run all tests and a fresh smoke profile.
2. Search the project for credentials, email addresses, usernames, absolute home paths, and customer text.
3. Confirm non-release output and `.env` files are ignored.
4. Confirm all bundled simulation records remain marked as synthetic and only use the Astra and Codex system labels.
5. Review source-adapter compatibility against the latest source log versions.
6. Review Langfuse API compatibility against the deployed server version.
7. Select and add a repository license.
8. Record the release tag, schema version, generator version, query-catalog checksum, and test result.

## Known Boundaries

- The canonical JSONL layout is a benchmark implementation choice, not a required Langfuse storage layout.
- The synthetic corpus tests controlled statistical behavior; it is not evidence of real product performance.
- Real-log adapters can only support the source formats represented by their regression fixtures.
- The bundled simulation corpus contains exactly 10,000 traces and occupies about 91 MB. Check repository hosting limits before adding larger files or future corpus versions.
- Human-review reliability studies require multiple independent reviewers on overlapping targets.
