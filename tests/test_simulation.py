from pathlib import Path

import pytest

from robot_learner.models import Action, ActionKind
from robot_learner.simulation import (
    GeneratedScript,
    ScriptValidationError,
    SimulationClient,
    SimulationSpec,
    checkpoint_id_for,
    parse_restricted_script,
    resolve_render_backend,
    script_from_actions,
)
from robot_learner.simulation_testing import fake_worker_main


def test_simulation_spec_resolves_piper_path_against_config(tmp_path: Path) -> None:
    meshes = tmp_path / "meshes" / "piper" / "assets"
    meshes.mkdir(parents=True)
    path = tmp_path / "simulation.toml"
    path.write_text(
        """
[simulation]
env_id = "CableTie-v0"
host = "cableties_sim:CableTieHost"
cameras = ["work", "wrist"]
capture_stride = 20

[simulation.env_kwargs]
control_repeat = 2
piper_asset_dir = "meshes/piper/assets"
""",
        encoding="utf-8",
    )

    spec = SimulationSpec.from_toml(path)

    assert spec.env_id == "CableTie-v0"
    assert spec.host == "cableties_sim:CableTieHost"
    assert spec.cameras == ("work", "wrist")
    assert spec.capture_stride == 20
    assert spec.env_kwargs["control_repeat"] == 2
    assert spec.env_kwargs["piper_asset_dir"] == str(meshes)
    assert spec.render_backend == "auto"


def test_simulation_spec_resolves_sibling_robo_wiki_from_configs_dir(tmp_path: Path) -> None:
    project = tmp_path / "robotLearner"
    meshes = tmp_path / "robo-wiki" / "labTesting" / "assets" / "piper" / "assets"
    meshes.mkdir(parents=True)
    config = project / "configs" / "cableties.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
[simulation]
env_id = "CableTie-v0"
host = "cableties_sim:CableTieHost"

[simulation.env_kwargs]
piper_asset_dir = "../robo-wiki/labTesting/assets/piper/assets"
""",
        encoding="utf-8",
    )

    spec = SimulationSpec.from_toml(config)

    assert spec.env_kwargs["piper_asset_dir"] == str(meshes)


def test_simulation_spec_lists_tried_piper_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "simulation.toml"
    path.write_text(
        """
[simulation]
env_id = "CableTie-v0"
host = "cableties_sim:CableTieHost"

[simulation.env_kwargs]
piper_asset_dir = "../robo-wiki/labTesting/assets/piper/assets"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Tried:"):
        SimulationSpec.from_toml(path)


def test_auto_render_backend_is_osmesa_only_on_linux() -> None:
    assert resolve_render_backend("auto", platform="linux") == "osmesa"
    assert resolve_render_backend("auto", platform="darwin") is None
    assert resolve_render_backend("glfw", platform="darwin") == "glfw"
    with pytest.raises(ValueError, match="osmesa is not available"):
        resolve_render_backend("osmesa", platform="darwin")


def test_restricted_script_parses_approved_calls() -> None:
    commands = parse_restricted_script(
        'sim.observe(cameras=["work"])\n'
        'sim.run_skill("thread_tie", nudge_m=[0.003, -0.002, 0.0])\n'
        "sim.stop()\n"
    )

    assert [command.operation for command in commands] == ["observe", "run_skill", "stop"]
    assert commands[1].arguments == {
        "name": "thread_tie",
        "parameters": {"nudge_m": [0.003, -0.002, 0.0]},
    }


def test_restricted_script_keeps_empty_camera_list() -> None:
    commands = parse_restricted_script("sim.observe(cameras=[])\n")
    assert commands[0].arguments == {"cameras": []}


def test_restricted_script_rejects_string_cameras() -> None:
    with pytest.raises(ScriptValidationError, match="list of strings"):
        parse_restricted_script('sim.observe(cameras="work")\n')


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "open('/tmp/result', 'w')",
        "sim.__class__()",
        "for _ in range(3):\n    sim.stop()",
        "x = sim.observe()",
        "sim.run_skill(name)",
        'sim.run_skill("thread_tie", {"nudge_m": [0, 0, 0]}, extra=1)',
    ],
)
def test_restricted_script_rejects_python_escape_surface(source: str) -> None:
    with pytest.raises(ScriptValidationError):
        parse_restricted_script(source)


def test_checkpoint_id_rejects_path_escape() -> None:
    with pytest.raises(ScriptValidationError):
        checkpoint_id_for("/tmp/pwned")
    with pytest.raises(ScriptValidationError):
        checkpoint_id_for("../escape")


def test_script_from_actions_round_trips() -> None:
    source = script_from_actions(
        [
            Action(ActionKind.OBSERVE),
            Action(ActionKind.RUN_SKILL, {"name": "settle", "parameters": {}}),
            Action(ActionKind.OBSERVE),
        ],
        ("work", "wrist"),
    )
    commands = parse_restricted_script(source)
    assert [command.operation for command in commands] == ["observe", "run_skill", "observe"]
    assert commands[1].arguments["name"] == "settle"


def _spec() -> SimulationSpec:
    return SimulationSpec(
        env_id="Fake-v0",
        host="unused:Host",
        worker_timeout_s=10,
        cameras=("work",),
    )


def test_simulation_client_observe_and_skill(tmp_path: Path) -> None:
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        observation = client.observe()
        assert observation.artifact_refs
        assert observation.context["phase"] == "init"
        result = client.execute(Action(ActionKind.RUN_SKILL, {"name": "settle", "parameters": {}}))
        assert result["ok"] is True
        assert result["phase"] == "settle"


def test_simulation_client_marks_unstable_as_failure(tmp_path: Path) -> None:
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        result = client.run_skill("break")
        assert result["ok"] is False
        assert result["unstable"] is True


def test_simulation_client_snapshot_restore(tmp_path: Path) -> None:
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        snap = client.snapshot()
        client.run_skill("settle")
        restored = client.restore(snap)
        assert restored["phase"] == "init"


def test_simulation_client_run_script_and_escape(tmp_path: Path) -> None:
    with SimulationClient(_spec(), tmp_path, worker=fake_worker_main) as client:
        result = client.run_script(
            GeneratedScript(
                checkpoint_id="scene_ready",
                source='sim.observe(cameras=["work"])\nsim.run_skill("settle")\n',
            )
        )
        assert result["results"][-1]["ok"] is True
        assert (tmp_path / "scripts" / "scene_ready" / "v1.py").is_file()
        with pytest.raises(ScriptValidationError):
            client.run_script(GeneratedScript(checkpoint_id="/tmp/pwned", source="sim.stop()\n"))


def test_cli_collects_mixed_flags_in_argv_order() -> None:
    from robot_learner.cli import collect_sim_steps

    steps = collect_sim_steps(
        [
            "simulate",
            "--checkpoint",
            "scene_ready=settle",
            "--section",
            "target_identified=examples/target_identified.py",
            "--skill",
            "identify_tie",
        ]
    )
    assert [kind for kind, _ in steps] == ["checkpoint", "section", "skill"]
