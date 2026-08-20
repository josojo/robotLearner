"""Isolated simulation loading, camera capture, and restricted scripts."""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import multiprocessing as mp
import os
import tomllib
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


class SimulationError(RuntimeError):
    """The simulation worker or environment rejected an operation."""


class ScriptValidationError(ValueError):
    """Generated script source is outside the restricted simulation grammar."""


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    env_id: str
    registration_modules: tuple[str, ...]
    seed: int | None = None
    worker_timeout_s: float = 300.0
    max_script_revisions: int = 3
    cameras: tuple[str, ...] = ()
    frame_width: int = 480
    frame_height: int = 360
    capture_stride: int = 40
    render_backend: str = "osmesa"
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path) -> SimulationSpec:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        data = raw.get("simulation")
        if not isinstance(data, dict):
            raise ValueError("configuration needs a [simulation] table")
        env_id = data.get("env_id")
        modules = data.get("registration_modules", [])
        cameras = data.get("cameras", [])
        kwargs = data.get("env_kwargs", {})
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("simulation.env_id must be a non-empty string")
        if not isinstance(modules, list) or not all(isinstance(x, str) for x in modules):
            raise ValueError("simulation.registration_modules must be strings")
        if not isinstance(cameras, list) or not all(isinstance(x, str) for x in cameras):
            raise ValueError("simulation.cameras must be strings")
        if not isinstance(kwargs, dict) or not _is_json(kwargs):
            raise ValueError("simulation.env_kwargs must contain JSON values")
        spec = cls(
            env_id=env_id,
            registration_modules=tuple(modules),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            worker_timeout_s=float(data.get("worker_timeout_s", 300)),
            max_script_revisions=int(data.get("max_script_revisions", 3)),
            cameras=tuple(cameras),
            frame_width=int(data.get("frame_width", 480)),
            frame_height=int(data.get("frame_height", 360)),
            capture_stride=int(data.get("capture_stride", 40)),
            render_backend=str(data.get("render_backend", "osmesa")),
            env_kwargs=dict(kwargs),
        )
        if spec.worker_timeout_s <= 0 or spec.capture_stride <= 0:
            raise ValueError("worker timeout and capture stride must be positive")
        if spec.frame_width <= 0 or spec.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return spec


@dataclass(frozen=True, slots=True)
class ScriptCommand:
    operation: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GeneratedScript:
    checkpoint_id: str
    source: str
    version: int = 1
    parent_version: int | None = None


def parse_restricted_script(source: str) -> tuple[ScriptCommand, ...]:
    """Accept only literal calls to sim.observe/run_skill/stop."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ScriptValidationError(f"invalid Python syntax: {exc.msg}") from exc
    commands: list[ScriptCommand] = []
    allowed = {"observe", "run_skill", "stop"}
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ScriptValidationError("scripts may contain only sim method calls")
        call = statement.value
        if not isinstance(call.func, ast.Attribute):
            raise ScriptValidationError("calls must target sim")
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "sim":
            raise ScriptValidationError("calls must target sim")
        operation = call.func.attr
        if operation not in allowed:
            raise ScriptValidationError(f"sim.{operation} is not permitted")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ScriptValidationError("keyword expansion is not permitted")
        try:
            positional = [ast.literal_eval(arg) for arg in call.args]
            keywords = {str(k.arg): ast.literal_eval(k.value) for k in call.keywords}
        except (ValueError, TypeError) as exc:
            raise ScriptValidationError("arguments must be JSON-like literals") from exc
        arguments = _normalize_call(operation, positional, keywords)
        if not _is_json(arguments):
            raise ScriptValidationError("arguments must be JSON values")
        commands.append(ScriptCommand(operation, arguments))
    if not commands:
        raise ScriptValidationError("script must contain at least one command")
    return tuple(commands)


def _normalize_call(
    operation: str, positional: list[Any], keywords: dict[str, Any]
) -> dict[str, Any]:
    if operation == "observe":
        if len(positional) > 1 or (positional and "cameras" in keywords):
            raise ScriptValidationError("observe accepts one cameras argument")
        cameras = positional[0] if positional else keywords.pop("cameras", None)
        if keywords:
            raise ScriptValidationError("unexpected observe arguments")
        return {} if cameras is None else {"cameras": cameras}
    if operation == "run_skill":
        if not positional or len(positional) > 2:
            raise ScriptValidationError("run_skill needs a skill name")
        if keywords and set(keywords) != {"parameters"}:
            # Ergonomic form: sim.run_skill("thread_tie", nudge_m=[...]).
            parameters = keywords
        else:
            parameters = keywords.get("parameters", positional[1] if len(positional) == 2 else {})
        if not isinstance(positional[0], str) or not isinstance(parameters, dict):
            raise ScriptValidationError("run_skill needs a string and parameter object")
        return {"name": positional[0], "parameters": parameters}
    if positional or keywords:
        raise ScriptValidationError("stop accepts no arguments")
    return {}


class SimulationClient:
    """Parent-side owner of one spawned Gymnasium simulation worker."""

    def __init__(self, spec: SimulationSpec, artifact_dir: Path) -> None:
        self.spec = spec
        self.artifact_dir = artifact_dir.resolve()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_worker_main,
            args=(child, asdict(spec), str(self.artifact_dir)),
            name=f"robot-learner-{spec.env_id}",
        )
        self._process.start()
        child.close()
        ready = self._receive()
        if ready.get("type") != "ready":
            self.close(force=True)
            raise SimulationError(str(ready.get("error", "worker did not become ready")))
        self.manifest = ready["manifest"]

    def observe(self, cameras: tuple[str, ...] | None = None) -> dict[str, Any]:
        return self._request("observe", cameras=list(cameras or self.spec.cameras))

    def run_script(self, script: GeneratedScript) -> dict[str, Any]:
        commands = [asdict(command) for command in parse_restricted_script(script.source)]
        script_dir = self.artifact_dir / "scripts" / script.checkpoint_id
        script_dir.mkdir(parents=True, exist_ok=True)
        destination = script_dir / f"v{script.version}.py"
        destination.write_text(script.source.rstrip() + "\n", encoding="utf-8")
        return self._request(
            "run_script",
            checkpoint_id=script.checkpoint_id,
            version=script.version,
            commands=commands,
        )

    def stop(self) -> dict[str, Any]:
        return self._request("stop")

    def close(self, *, force: bool = False) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.is_alive() and not force:
            try:
                self._request("close")
            except SimulationError:
                force = True
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()

    def __enter__(self) -> SimulationClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if not self._process.is_alive():
            raise SimulationError("simulation worker is not running")
        self._connection.send({"operation": operation, **payload})
        response = self._receive()
        if response.get("type") == "error":
            raise SimulationError(str(response.get("error", "simulation operation failed")))
        return response

    def _receive(self) -> dict[str, Any]:
        if not self._connection.poll(self.spec.worker_timeout_s):
            raise SimulationError("simulation worker timed out")
        response = self._connection.recv()
        if not isinstance(response, dict):
            raise SimulationError("simulation worker returned an invalid response")
        return response


def _worker_main(connection: Connection, raw_spec: dict[str, Any], artifact_dir: str) -> None:
    env: Any = None
    try:
        spec = SimulationSpec(**raw_spec)
        os.environ["MUJOCO_GL"] = spec.render_backend
        for module in spec.registration_modules:
            importlib.import_module(module)
        import gymnasium as gym
        from PIL import Image

        kwargs = dict(spec.env_kwargs)
        kwargs.update(width=spec.frame_width, height=spec.frame_height)
        env = gym.make(spec.env_id, render_mode="rgb_array", **kwargs)
        core = env.unwrapped
        _, info = env.reset(seed=spec.seed)
        available = tuple(core.camera_names())
        cameras = spec.cameras or available
        missing = set(cameras) - set(available)
        if missing:
            raise ValueError(f"unknown configured cameras: {sorted(missing)}")
        if not cameras:
            raise ValueError("environment exposes no named cameras")
        root = Path(artifact_dir)
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_index = 0

        def capture(phase: str, selected: tuple[str, ...] = cameras) -> list[dict[str, Any]]:
            nonlocal frame_index
            results = []
            for camera in selected:
                rgb = core.render_camera(camera)
                destination = frames_dir / camera / f"{frame_index:08d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(destination)
                results.append(
                    {
                        "camera": camera,
                        "simulation_time": float(core.data.time),
                        "phase": phase,
                        "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                        "artifact_ref": str(destination),
                    }
                )
            frame_index += 1
            return results

        callback_count = 0

        def on_step(_: Any) -> None:
            nonlocal callback_count
            callback_count += 1
            if callback_count % spec.capture_stride == 0:
                capture(str(core.env.phase))

        core.set_step_callback(on_step)
        manifest = core.capability_manifest()
        manifest.update(
            {
                "configured_cameras": list(cameras),
                "seed": spec.seed,
                "initial_info": _jsonable(info),
            }
        )
        connection.send({"type": "ready", "manifest": manifest})
        while True:
            request = connection.recv()
            operation = request.get("operation")
            if operation == "close":
                connection.send({"type": "closed"})
                break
            if operation == "stop":
                connection.send({"type": "stopped", "time": float(core.data.time)})
                continue
            if operation == "observe":
                selected = tuple(request.get("cameras") or cameras)
                unknown = set(selected) - set(available)
                if unknown:
                    raise ValueError(f"unknown cameras: {sorted(unknown)}")
                connection.send(
                    {
                        "type": "observation",
                        "frames": capture(str(core.env.phase), selected),
                        "state": _state(core),
                    }
                )
                continue
            if operation == "run_script":
                results = []
                for raw in request.get("commands", []):
                    command = ScriptCommand(**raw)
                    if command.operation == "observe":
                        selected = tuple(command.arguments.get("cameras") or cameras)
                        results.append(
                            {
                                "operation": "observe",
                                "frames": capture(str(core.env.phase), selected),
                            }
                        )
                    elif command.operation == "run_skill":
                        name = command.arguments["name"]
                        ok = core.run_skill(name, command.arguments.get("parameters"))
                        results.append({"operation": "run_skill", "name": name, "ok": ok})
                        if not ok or bool(core.env.unstable):
                            break
                    else:
                        results.append({"operation": "stop"})
                        break
                connection.send(
                    {
                        "type": "script_result",
                        "checkpoint_id": request.get("checkpoint_id"),
                        "version": request.get("version"),
                        "results": results,
                        "state": _state(core),
                        "frames": capture(str(core.env.phase)),
                    }
                )
                continue
            raise ValueError(f"unknown worker operation {operation!r}")
    except EOFError:
        pass
    except Exception as exc:
        with contextlib.suppress(BrokenPipeError, EOFError):
            connection.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
        connection.close()


def _state(core: Any) -> dict[str, Any]:
    observation = core.observation()
    return {
        "simulation_time": float(core.data.time),
        "phase": str(core.env.phase),
        "unstable": bool(core.env.unstable),
        "observation": _jsonable(observation),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True
