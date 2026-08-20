import json

import pytest

from robot_learner.planning import CheckpointPlanner, PlanError, task_to_dict


class StubLanguageModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.request: tuple[str, str | None] | None = None

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.request = (prompt, system_prompt)
        return self.response


def test_planner_creates_ordered_checkpoints_from_initial_prompt() -> None:
    response = json.dumps({"checkpoints": [
        {"id": "object_found", "name": "Object located",
         "success_predicate": {"name": "object_visible", "parameters": {}},
         "dependencies": [], "recovery_options": ["reobserve the workspace"]},
        {"id": "object_grasped", "name": "Object securely grasped",
         "success_predicate": {"name": "grasp_confirmed",
                               "parameters": {"minimum_confidence": 0.9}},
         "dependencies": ["object_found"],
         "recovery_options": ["release and return to a safe observation pose"]},
    ]})
    model = StubLanguageModel(f"```json\n{response}\n```")
    task = CheckpointPlanner(model).create_task("Pick up the red block")
    assert [checkpoint.id for checkpoint in task.checkpoints] == [
        "object_found", "object_grasped"]
    assert task.checkpoints[1].dependencies == ("object_found",)
    assert '"minimum_confidence": 0.9' in json.dumps(task_to_dict(task))
    assert model.request is not None
    assert "independently verifiable" in (model.request[1] or "")


def test_planner_rejects_forward_dependencies() -> None:
    model = StubLanguageModel(json.dumps({"checkpoints": [{
        "id": "grasped", "name": "Object grasped",
        "success_predicate": {"name": "grasp_confirmed"},
        "dependencies": ["located"],
    }]}))
    with pytest.raises(PlanError, match="unknown or forward dependencies"):
        CheckpointPlanner(model).create_task("Pick up the object")
