# Langfuse Workflow

Langfuse stores and presents traces, observations, experiments, and scores. The benchmark keeps deterministic Python calculations as the reference answers.

## End-to-End Flow

1. Convert source logs to the canonical data model or generate a synthetic corpus.
2. Validate keys, parent-child relationships, timestamps, and synthetic markers.
3. Run reference queries locally and freeze their answers with the dataset version and checksum.
4. Preview the Langfuse upload without `--send`.
5. Upload selected traces with stable event IDs and resumable local state.
6. Verify that remote Trace and Observation IDs match the local selection.
7. Add trace- or observation-level scores in the Langfuse UI.
8. Export manual scores through the Scores API.
9. Join exported scores back to local canonical records and run annotation queries.

## Storage Representation

The canonical JSONL tables are normalized for repeatable Python and SQL analysis. They are one possible physical representation, not a Langfuse requirement. The Langfuse UI reconstructs an observation tree from parent identifiers. Other adapters may accept nested JSON arrays and normalize them before query execution, or execute equivalent queries directly against a nested representation.

## Version Compatibility

The integration targets the Langfuse Python SDK declared in `requirements.txt`. Inventory verification first tries the current Observations API and falls back to the legacy paginated endpoint used by older self-hosted deployments. API compatibility should be rerun whenever the Langfuse server or SDK is upgraded.

## Annotation Semantics

Langfuse scores may target a Trace, Observation, Session, or Dataset Run. The exporter preserves the score name, value, data type, target ID, trace ID, source, author ID, label timestamp, ingestion timestamp, update timestamp, queue ID, comment, and source metadata when available.

The benchmark canonical representation stores one annotation event per row and preserves revision links. A UI update should not be treated as a silent overwrite when annotation history is required for research.
