import json
from pathlib import Path

import pytest

from robot_learner.cli import run_explore, run_review
from robot_learner.planning import task_to_dict
from robot_learner.review import ReviewError, write_review
from robot_learner.simulation_testing import MIN_PNG, fake_worker_main
from test_explore import ScriptedLanguageModel, _task


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MIN_PNG)


def test_write_review_builds_html_player(tmp_path: Path) -> None:
    _write_png(tmp_path / "frames" / "work" / "00000000.png")
    _write_png(tmp_path / "frames" / "work" / "00000001.png")
    _write_png(tmp_path / "frames" / "wrist" / "00000000.png")
    (tmp_path / "frames" / "index.jsonl").write_text(
        json.dumps({"index": 0, "phase": "init"}) + "\n"
        + json.dumps({"index": 1, "phase": "settle"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "explore.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_id": "scene_ready",
                        "skill": "settle",
                        "ok": True,
                        "start_frame": 0,
                        "error": None,
                        "attempts": [
                            {
                                "revision": 1,
                                "skill": "settle",
                                "ok": True,
                                "error": None,
                            }
                        ],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = write_review(tmp_path, encode=False)

    html = artifacts.html.read_text(encoding="utf-8")
    data = json.loads(artifacts.data.read_text(encoding="utf-8"))
    assert artifacts.html.name == "review.html"
    assert "work" in html
    assert "wrist" in html
    assert data["cameras"] == ["work", "wrist"]
    assert data["frames"][0]["phase"] == "init"
    assert data["checkpoints"][0]["id"] == "scene_ready"
    assert "const DATA = " in html


def test_write_review_rejects_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ReviewError, match="no captured frames"):
        write_review(tmp_path)


def test_encode_writes_video_when_possible(tmp_path: Path) -> None:
    _write_png(tmp_path / "frames" / "work" / "00000000.png")
    _write_png(tmp_path / "frames" / "work" / "00000001.png")
    artifacts = write_review(tmp_path, encode=True)
    if artifacts.videos:
        assert artifacts.videos[0].is_file()
        assert artifacts.videos[0].suffix in {".mp4", ".webp", ".gif"}
    else:
        assert artifacts.notes


def test_encode_falls_back_when_ffmpeg_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("robot_learner.review._find_ffmpeg", lambda: None)
    _write_png(tmp_path / "frames" / "work" / "00000000.png")
    _write_png(tmp_path / "frames" / "work" / "00000001.png")
    artifacts = write_review(tmp_path, encode=True)
    assert artifacts.html.is_file()
    if artifacts.videos:
        assert artifacts.videos[0].suffix in {".webp", ".gif"}
    else:
        assert any("ffmpeg" in note for note in artifacts.notes)


def test_review_cli_from_explore_run(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(task_to_dict(_task("scene_ready"))) + "\n")
    config = tmp_path / "sim.toml"
    config.write_text(
        """
[simulation]
env_id = "Fake-v0"
host = "unused:Host"
cameras = ["work"]
""",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    model = ScriptedLanguageModel(['sim.run_skill("settle")\n'])
    assert (
        run_explore(config, plan, output, language_model=model, worker=fake_worker_main) == 0
    )
    assert (output / "review.html").is_file()
    artifacts = write_review(output, encode=False)
    data = json.loads(artifacts.data.read_text(encoding="utf-8"))
    assert data["cameras"] == ["work"]
    assert data["frames"]
    assert run_review(output, encode=False) == 0
