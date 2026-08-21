import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from robot_learner.cli import run_explore
from robot_learner.explore import (
    EXPLORER_SYSTEM_PROMPT,
    ExplorationError,
    SimulationExplorer,
    extract_restricted_script,
    observation_image_paths,
)
from robot_learner.models import Checkpoint, Observation, Predicate, Task, utc_now
from robot_learner.planning import task_from_dict, task_to_dict
from robot_learner.simulation import (
    ScriptValidationError,
    SimulationClient,
    SimulationSpec,
    parse_restricted_script,
)
from robot_learner.simulation_testing import fake_worker_main


class ScriptedLanguageModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[tuple[str, str | None]] = []
        self.images: list[tuple[str, ...]] = []

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        images: Sequence[str | Path] | None = None,
    ) -> str:
        self.prompts.append((prompt, system_prompt))
        self.images.append(tuple(str(item) for item in images or ()))
        if not self.responses:
            raise AssertionError("language model received an extra request")
        return self.responses.pop(0)


def _spec(*, max_script_revisions: int = 3) -> SimulationSpec:
    return SimulationSpec(
        env_id="Fake-v0",
        host="unused:Host",
        worker_timeout_s=10,
        cameras=("work",),
        max_script_revisions=max_script_revisions,
    )


def _task(*checkpoint_ids: str) -> Task:
    checkpoints: list[Checkpoint] = []
    previous = ""
    for checkpoint_id in checkpoint_ids:
        dependencies = (previous,) if previous else ()
        checkpoints.append(
            Checkpoint(
                checkpoint_id,
                checkpoint_id.replace("_", " "),
                Predicate("visible"),
                dependencies,
            )
        )
        previous = checkpoint_id
    return Task("task_cable", "Zip a cable tie in the holder", tuple(checkpoints))


def test_observation_image_paths_skips_non_images(tmp_path: Path) -> None:
    png = tmp_path / "work" / "frame.png"
    txt = tmp_path / "work" / "notes.txt"
    png.parent.mkdir()
    png.write_bytes(b"\x89PNG\r\n")
    txt.write_text("nope\n", encoding="utf-8")
    observation = Observation(
        "obs_1",
        utc_now(),
        {
            "frames": [
                {"camera": "work", "artifact_ref": str(png)},
                {"camera": "notes", "artifact_ref": str(txt)},
            ]
        },
        (str(png), str(txt), str(tmp_path / "missing.png")),
    )
    assert observation_image_paths(observation) == (str(png),)


def test_extract_restricted_script_strips_markdown_fence() -> None:
    source = extract_restricted_script("```python\nsim.run_skill(\"settle\")\n```")
    commands = parse_restricted_script(source)
    assert commands[0].arguments["name"] == "settle"


def test_explorer_persists_observe_skill_observe_script(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(['```\nsim.run_skill("settle")\n```'])
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        report = SimulationExplorer(client, model).explore(_task("scene_ready"))

    script = tmp_path / "scripts" / "scene_ready" / "v1.py"
    assert report.ok is True
    assert report.checkpoints[0].skill == "settle"
    assert script.is_file()
    assert parse_restricted_script(script.read_text(encoding="utf-8"))
    assert 'sim.run_skill("settle")' in script.read_text(encoding="utf-8")
    assert 'sim.observe(cameras=["work"])' in script.read_text(encoding="utf-8")
    prompt, system = model.prompts[0]
    assert system == EXPLORER_SYSTEM_PROMPT
    assert "settle" in prompt
    assert "candidate_ties" in prompt
    assert "Which sim.run_skill(...) next?" in prompt
    assert "Camera frames attached" in prompt
    assert model.images[0]
    assert Path(model.images[0][0]).suffix == ".png"


def test_explorer_rejects_unknown_skill_and_retries(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(
        [
            'sim.run_skill("explode")\n',
            'sim.run_skill("settle")\n',
        ]
    )
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        report = SimulationExplorer(client, model).explore(_task("scene_ready"))

    assert report.ok is True
    assert report.checkpoints[0].attempts[0].ok is False
    assert "catalog" in (report.checkpoints[0].attempts[0].error or "")
    assert (tmp_path / "scripts" / "scene_ready" / "v1.py").is_file()


def test_explorer_restores_after_failed_skill_then_retries(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(
        [
            'sim.run_skill("break")\n',
            'sim.run_skill("settle")\n',
        ]
    )
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        report = SimulationExplorer(client, model).explore(_task("scene_ready"))
        after = client.observe()

    assert report.ok is True
    assert after.context["phase"] == "settle"
    assert after.context["unstable"] is False
    retry_prompt = model.prompts[1][0]
    assert "Previous attempt failed" in retry_prompt
    assert "break" in retry_prompt


def test_explorer_stops_after_max_script_revisions(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(['sim.run_skill("fail")\n'] * 2)
    with SimulationClient(
        _spec(max_script_revisions=2), tmp_path, worker=fake_worker_main
    ) as client:
        report = SimulationExplorer(client, model).explore(
            _task("scene_ready", "target_identified")
        )

    assert report.ok is False
    assert len(report.checkpoints) == 1
    assert len(report.checkpoints[0].attempts) == 2
    assert not (tmp_path / "scripts" / "scene_ready").exists()
    assert model.responses == []


def test_explorer_walks_plan_and_keeps_one_client(tmp_path: Path) -> None:
    model = ScriptedLanguageModel(
        [
            'sim.run_skill("settle")\n',
            'sim.run_skill("identify_tie", idx=1)\n',
        ]
    )
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        report = SimulationExplorer(client, model).explore(
            _task("scene_ready", "target_identified")
        )

    assert report.ok is True
    assert [item.skill for item in report.checkpoints] == ["settle", "identify_tie"]
    second = (tmp_path / "scripts" / "target_identified" / "v1.py").read_text(encoding="utf-8")
    assert "identify_tie" in second
    assert "idx=1" in second
    assert "scene_ready" in model.prompts[1][0]


def test_explorer_rejects_unrestricted_python(tmp_path: Path) -> None:
    with pytest.raises(ScriptValidationError):
        parse_restricted_script("import os\n")
    model = ScriptedLanguageModel(["import os\n", 'sim.run_skill("settle")\n'])
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        report = SimulationExplorer(client, model).explore(_task("scene_ready"))
    assert report.ok is True


def test_example_cabletie_plan_loads() -> None:
    path = Path("examples/cabletie_plan.json")
    task = task_from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert task.checkpoints[0].id == "scene_ready"
    assert task.checkpoints[-1].id == "zipped_placed"


def test_empty_plan_is_rejected(tmp_path: Path) -> None:
    model = ScriptedLanguageModel([])
    with (
        SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client,
        pytest.raises(ExplorationError, match="no checkpoints"),
    ):
        SimulationExplorer(client, model).explore(Task("empty", "none", ()))


def test_run_explore_cli_writes_scripts(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(task_to_dict(_task("scene_ready")), indent=2) + "\n")
    config = tmp_path / "sim.toml"
    config.write_text(
        """
[simulation]
env_id = "Fake-v0"
host = "unused:Host"
cameras = ["work"]
max_script_revisions = 2
""",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    model = ScriptedLanguageModel(['sim.run_skill("settle")\n'])

    assert (
        run_explore(config, plan, output, language_model=model, worker=fake_worker_main) == 0
    )
    assert (output / "scripts" / "scene_ready" / "v1.py").is_file()
    saved = json.loads((output / "explore.json").read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved["checkpoints"][0]["skill"] == "settle"


def test_task_from_dict_round_trips() -> None:
    task = _task("scene_ready", "target_identified")
    loaded = task_from_dict(task_to_dict(task))
    assert loaded.id == task.id
    assert [checkpoint.id for checkpoint in loaded.checkpoints] == [
        "scene_ready",
        "target_identified",
    ]
