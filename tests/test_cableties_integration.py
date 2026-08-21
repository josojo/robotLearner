import os
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

cableties_sim = pytest.importorskip("cableties_sim")

PIPER_ASSETS = Path(__file__).parents[2] / "robo-wiki/labTesting/assets/piper/assets"


@pytest.mark.skipif(not PIPER_ASSETS.is_dir(), reason="external PiPER meshes are not fetched")
def test_cabletie_host_exposes_privileged_state_and_snapshots() -> None:
    host = cableties_sim.CableTieHost(
        seed=7,
        width=96,
        height=72,
        piper_asset_dir=str(PIPER_ASSETS),
    )
    try:
        state = host.privileged_state()
        assert state["candidate_ties"] == list(range(8))
        assert len(state["ties"]) == 8
        assert "hole_in" in state["ties"][0]
        assert {"work", "closeup", "overview", "wrist"} <= set(host.camera_names())
        assert "thread_tie" in host.skill_catalog()
        try:
            rgb = host.render_camera("work")
        except Exception as exc:
            pytest.skip(f"MuJoCo renderer unavailable: {exc}")
        else:
            assert rgb.shape == (72, 96, 3)
        snap = host.snapshot()
        settled = host.run_skill("settle")
        assert settled.name == "settle"
        host.restore(snap)
        assert host.privileged_state()["phase"] == state["phase"]
        rejected = host.run_skill("thread_tie", {"nudge_m": [0.02, 0.0, 0.0]})
        assert rejected.ok is False
        assert "15 mm" in (rejected.error or "")
    finally:
        host.close()
