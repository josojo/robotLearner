from pathlib import Path
from typing import Any, cast

import cableties_sim  # noqa: F401
import gymnasium as gym
import pytest

PIPER_ASSETS = Path(__file__).parents[2] / "robo-wiki/labTesting/assets/piper/assets"


@pytest.mark.skipif(not PIPER_ASSETS.is_dir(), reason="external PiPER meshes are not fetched")
def test_registered_cabletie_environment_exposes_cameras_and_skills() -> None:
    env = gym.make(
        "CableTie-v0",
        render_mode="rgb_array",
        width=96,
        height=72,
        piper_asset_dir=str(PIPER_ASSETS),
    )
    try:
        observation, info = env.reset(seed=7)
        core = cast(Any, env.unwrapped)

        assert observation["arm_qpos"].shape == (6,)
        assert info["candidate_ties"] == list(range(8))
        assert {"work", "closeup", "overview", "wrist"} <= set(core.camera_names())
        assert "thread_tie" in core.skill_catalog()
        assert core.render_camera("work").shape == (72, 96, 3)
        with pytest.raises(ValueError, match="15 mm"):
            core.run_skill("thread_tie", {"nudge_m": [0.02, 0.0, 0.0]})
    finally:
        env.close()
