import unittest

from advanced_reference_queries import QUERY_FUNCTIONS


def make_bundle():
    models = [
        {"model_id": "model-a", "name": "Model A"},
        {"model_id": "model-b", "name": "Model B"},
    ]
    tasks = [
        {"task_id": f"task-{i}", "task_family": "qa", "difficulty": "easy" if i <= 2 else "hard"}
        for i in range(1, 5)
    ]
    sessions, traces, observations, generations, tool_calls, answers = [], [], [], [], [], []
    outcomes = {"model-a": [True, True, False, False], "model-b": [True, False, True, False]}
    for task_index, task in enumerate(tasks, 1):
        for model_index, model in enumerate(models):
            suffix = f"{task_index}-{model_index}"
            session_id, trace_id, root_id = f"session-{suffix}", f"trace-{suffix}", f"root-{suffix}"
            start_hour = (task_index - 1) * 2 + model_index
            start = f"2026-01-01T{start_hour:02d}:00:00Z"
            end = f"2026-01-01T{start_hour:02d}:01:00Z"
            sessions.append({"session_id": session_id, "source": "test", "start_time": start, "end_time": end, "metadata": {}})
            traces.append({
                "trace_id": trace_id, "session_id": session_id, "task_id": task["task_id"],
                "model_id": model["model_id"], "attempt_number": 1,
                "status": "error" if suffix == "4-1" else "success", "start_time": start, "end_time": end,
                "duration_ms": 100 + model_index * 100 + task_index, "cost": 1.0 + model_index,
                "usage": {"input": 10 + task_index, "output": 5}, "metadata": {},
            })
            observations.append({"observation_id": root_id, "trace_id": trace_id, "parent_observation_id": None, "type": "span", "name": "root", "status": "success", "start_time": start, "end_time": end, "metadata": {}})
            generation_id = f"gen-{suffix}"
            observations.append({"observation_id": generation_id, "trace_id": trace_id, "parent_observation_id": root_id, "type": "generation", "name": "llm", "status": "success", "start_time": start, "end_time": end, "metadata": {}})
            generations.append({"observation_id": generation_id, "trace_id": trace_id, "model_id": model["model_id"], "duration_ms": 50, "usage": {"input": 10, "output": 5}, "metadata": {}})
            if task_index <= 2:
                tool_id = f"tool-{suffix}"
                tool_status = "error" if suffix == "1-0" else "success"
                observations.append({"observation_id": tool_id, "trace_id": trace_id, "parent_observation_id": root_id, "type": "tool", "name": "search", "status": tool_status, "start_time": start, "end_time": end, "metadata": {}})
                tool_calls.append({"observation_id": tool_id, "trace_id": trace_id, "tool_name": "search" if task_index == 1 else "python", "tool_index": 1, "status": tool_status, "duration_ms": 20, "metadata": {}})
            answers.append({"answer_id": f"answer-{suffix}", "task_id": task["task_id"], "trace_id": trace_id, "model_id": model["model_id"], "answer_text": "test", "correct": outcomes[model["model_id"]][task_index - 1], "grading_version": "test-v1", "metadata": {}})

    annotations = []
    for task_index in range(1, 5):
        target = f"trace-{task_index}-0"
        for reviewer, values in (("reviewer-a", [True, True, False, False]), ("reviewer-b", [True, False, True, False])):
            annotation_id = f"ann-{reviewer[-1]}-{task_index}"
            annotations.append({
                "annotation_id": annotation_id, "target_type": "trace", "target_id": target, "trace_id": target,
                "label_schema_version": "1.0.0", "label_json": {"answer_correct": values[task_index - 1], "confidence": 0.9},
                "annotator_id": reviewer, "annotator_type": "human", "labeled_at": f"2026-01-02T0{task_index}:00:00Z",
                "ingested_at": f"2026-01-02T0{task_index}:00:05Z", "source": "test",
                "supersedes_annotation_id": None, "retracted": False, "metadata": {},
            })
    annotations.append({
        "annotation_id": "ann-a-1-revision", "target_type": "trace", "target_id": "trace-1-0", "trace_id": "trace-1-0",
        "label_schema_version": "1.0.0", "label_json": {"answer_correct": True, "confidence": 1.0},
        "annotator_id": "reviewer-a", "annotator_type": "human", "labeled_at": "2026-01-02T08:00:00Z",
        "ingested_at": "2026-01-02T08:00:10Z", "source": "test", "supersedes_annotation_id": "ann-a-1",
        "retracted": False, "metadata": {},
    })
    return {
        "models": models, "tasks": tasks, "sessions": sessions, "traces": traces,
        "observations": observations, "generations": generations, "tool_calls": tool_calls,
        "model_answers": answers, "annotations": annotations,
    }


CONFIG = {
    "selected_models": ["model-a", "model-b"],
    "bootstrap_seed": 7,
    "bootstrap_repetitions": 100,
    "permutation_repetitions": 100,
    "minimum_change_point_window": 2,
    "annotation_label_field": "answer_correct",
}


class AdvancedQueryTests(unittest.TestCase):
    def test_all_advanced_queries_return_objects(self):
        bundle = make_bundle()
        for query_id, function in QUERY_FUNCTIONS.items():
            with self.subTest(query_id=query_id):
                self.assertIsInstance(function(bundle, CONFIG), dict)

    def test_paired_accuracy_and_difficulty_interaction(self):
        bundle = make_bundle()
        paired = QUERY_FUNCTIONS["query17"](bundle, CONFIG)
        self.assertEqual(paired["matched_tasks"], 4)
        self.assertEqual(paired["accuracy_a"], 0.5)
        self.assertEqual(paired["accuracy_b"], 0.5)
        self.assertEqual(paired["exact_mcnemar_p_value"], 1.0)

        interaction = QUERY_FUNCTIONS["query20"](bundle, CONFIG)
        self.assertEqual(interaction["interaction_hard_minus_easy"], -1.0)

    def test_pareto_and_annotation_history(self):
        bundle = make_bundle()
        pareto = QUERY_FUNCTIONS["query19"](bundle, CONFIG)
        self.assertEqual(pareto["frontier"], ["model-a"])

        effective = QUERY_FUNCTIONS["query27"](bundle, CONFIG)
        self.assertEqual(effective["effective_label_count"], 8)
        timing = QUERY_FUNCTIONS["query30"](bundle, CONFIG)
        self.assertEqual(timing["revision_events"], 1)


if __name__ == "__main__":
    unittest.main()
