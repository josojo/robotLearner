from pathlib import Path

import pytest

from robot_learner.simulation import (
    ScriptValidationError,
    SimulationSpec,
    parse_restricted_script,
)


def test_simulation_spec_loads_nested_environment_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "simulation.toml"
    path.write_text(
        """
[simulation]
env_id = "CableTie-v0"
registration_modules = ["cableties_sim"]
cameras = ["work", "wrist"]
capture_stride = 20

[simulation.env_kwargs]
control_repeat = 2
""",
        encoding="utf-8",
    )

    spec = SimulationSpec.from_toml(path)

    assert spec.env_id == "CableTie-v0"
    assert spec.cameras == ("work", "wrist")
    assert spec.capture_stride == 20
    assert spec.env_kwargs == {"control_repeat": 2}


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


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "open('/tmp/result', 'w')",
        "sim.__class__()",
        "for _ in range(3):\n    sim.stop()",
        "x = sim.observe()",
        "sim.run_skill(name)",
    ],
)
def test_restricted_script_rejects_python_escape_surface(source: str) -> None:
    with pytest.raises(ScriptValidationError):
        parse_restricted_script(source)
