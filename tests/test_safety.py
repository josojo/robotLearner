from pathlib import Path

import pytest

from robot_learner.config import Settings
from robot_learner.models import Action, ActionKind, SafetyContract, Strategy
from robot_learner.safety import SafetyValidator, SafetyViolation


def test_rejects_move_outside_workspace(tmp_path: Path) -> None:
    contract = SafetyContract(0.1, 10.0, 30.0, (-0.5, -0.5, 0.0), (0.5, 0.5, 0.75))
    settings = Settings(tmp_path, tmp_path / "db", True, frozenset({"task"}), False, contract)
    strategy = Strategy(
        "unsafe",
        "checkpoint",
        (),
        (),
        contract,
        (Action(ActionKind.MOVE_CARTESIAN, {"target_m": [2.0, 0.0, 0.1], "speed_m_s": 0.1}),),
        trusted=True,
    )

    with pytest.raises(SafetyViolation, match="outside"):
        SafetyValidator(settings).validate("task", strategy, human_approved=True)

