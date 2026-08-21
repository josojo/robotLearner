"""Stdlib-only fake worker for SimulationClient tests."""

from __future__ import annotations

import contextlib
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

# 1x1 RGBA PNG so observe() can feed the vision explorer without PIL.
MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082"
)


def fake_worker_main(connection: Connection, raw_spec: dict[str, Any], artifact_dir: str) -> None:
    """Speak the simulation worker protocol without MuJoCo or PIL."""
    root = Path(artifact_dir)
    frames_dir = root / "frames" / "work"
    frames_dir.mkdir(parents=True, exist_ok=True)
    phase = "init"
    unstable = False
    sim_time = 0.0
    snaps: dict[int, tuple[str, bool, float]] = {}
    snap_i = 0
    frame_index = 0

    def state() -> dict[str, Any]:
        return {
            "simulation_time": sim_time,
            "phase": phase,
            "unstable": unstable,
            "observation": {"arm_qpos": [0.0] * 6},
            "privileged": {
                "time": sim_time,
                "phase": phase,
                "unstable": unstable,
                "candidate_ties": [0, 1],
                "ties": [],
            },
        }

    def capture() -> list[dict[str, Any]]:
        nonlocal frame_index
        destination = frames_dir / f"{frame_index:08d}.png"
        destination.write_bytes(MIN_PNG)
        frame_index += 1
        return [
            {
                "camera": "work",
                "simulation_time": sim_time,
                "phase": phase,
                "width": 8,
                "height": 8,
                "artifact_ref": str(destination),
            }
        ]

    def run_skill(name: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
        nonlocal phase, unstable, sim_time
        if name == "missing":
            return {
                "ok": False,
                "name": name,
                "unstable": unstable,
                "phase": phase,
                "error": "unknown skill",
            }
        if name == "break":
            unstable = True
        phase = name
        sim_time += 1.0
        ok = name != "fail" and not unstable
        return {
            "ok": ok,
            "name": name,
            "unstable": unstable,
            "phase": phase,
            "error": None if ok else "skill failed",
        }

    try:
        connection.send(
            {
                "type": "ready",
                "request_id": None,
                "manifest": {
                    "provider": "fake",
                    "environment_id": raw_spec.get("env_id", "Fake-v0"),
                    "cameras": ["work", "wrist"],
                    "skills": {
                        "settle": {"parameters": {}},
                        "identify_tie": {"parameters": {"idx": "integer|null"}},
                        "fail": {"parameters": {}},
                        "break": {"parameters": {}},
                    },
                    "sensors": ["arm_qpos"],
                },
            }
        )
        while True:
            request = connection.recv()
            request_id = request.get("request_id")
            operation = request.get("operation")
            if operation == "close":
                connection.send({"type": "closed", "request_id": request_id})
                break
            if operation == "stop":
                connection.send({"type": "stopped", "request_id": request_id, "time": sim_time})
                continue
            if operation == "observe":
                connection.send(
                    {
                        "type": "observation",
                        "request_id": request_id,
                        "frames": capture(),
                        "state": state(),
                    }
                )
                continue
            if operation == "run_skill":
                result = run_skill(request.get("name"), request.get("parameters"))
                connection.send(
                    {
                        "type": "skill_result",
                        "request_id": request_id,
                        "result": result,
                        "state": state(),
                    }
                )
                continue
            if operation == "snapshot":
                snap_i += 1
                snaps[snap_i] = (phase, unstable, sim_time)
                connection.send(
                    {
                        "type": "snapshot",
                        "request_id": request_id,
                        "snapshot_id": snap_i,
                    }
                )
                continue
            if operation == "restore":
                phase, unstable, sim_time = snaps[int(request["snapshot_id"])]
                connection.send(
                    {
                        "type": "restored",
                        "request_id": request_id,
                        "state": state()["privileged"],
                    }
                )
                continue
            if operation == "run_script":
                results = []
                for raw in request.get("commands", []):
                    op = raw.get("operation")
                    if op == "observe":
                        results.append({"operation": "observe", "ok": True, "frames": capture()})
                    elif op == "run_skill":
                        arguments = raw["arguments"]
                        skill = run_skill(arguments["name"], arguments.get("parameters"))
                        ok = bool(skill.get("ok")) and not bool(skill.get("unstable"))
                        results.append(
                            {
                                "operation": "run_skill",
                                "name": raw["arguments"]["name"],
                                "ok": ok,
                                "error": skill.get("error"),
                            }
                        )
                        if not ok:
                            break
                    else:
                        results.append({"operation": "stop", "ok": True})
                        break
                connection.send(
                    {
                        "type": "script_result",
                        "request_id": request_id,
                        "checkpoint_id": request.get("checkpoint_id"),
                        "version": request.get("version"),
                        "results": results,
                        "state": state(),
                        "stride_frames": [],
                    }
                )
                continue
            connection.send(
                {
                    "type": "error",
                    "error": f"unknown worker operation {operation!r}",
                    "request_id": request_id,
                }
            )
    except EOFError:
        pass
    finally:
        with contextlib.suppress(OSError):
            connection.close()
