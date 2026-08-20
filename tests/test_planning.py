import json
import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from robot_learner.models import Task
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


def test_start_loads_api_key_from_dotenv(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    from robot_learner import cli

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    class StubPlanner:
        def __init__(self, model: object) -> None:
            assert model is not None

        def create_task(self, prompt: str) -> Task:
            assert os.environ["OPENROUTER_API_KEY"] == "from-dotenv"
            return CheckpointPlanner(StubLanguageModel(json.dumps({
                "checkpoints": [{
                    "id": "scene_seen",
                    "name": "Scene seen",
                    "success_predicate": {"name": "scene_visible"},
                }]
            }))).create_task(prompt)

    monkeypatch.setattr(cli, "CheckpointPlanner", StubPlanner)
    config = tmp_path / "config.toml"
    config.write_text(
        """[runtime]
artifact_dir = "artifacts"
database_path = "artifacts/test.db"
dry_run = true

[safety]
authorized_task_ids = []
allow_untrusted_strategies = false
max_velocity_m_s = 0.1
max_force_n = 10.0
max_duration_s = 30.0
workspace_min_m = [-0.5, -0.5, 0.0]
workspace_max_m = [0.5, 0.5, 0.75]
""",
        encoding="utf-8",
    )

    assert cli.create_plan("Observe the scene", config, tmp_path / "plan.json") == 0
