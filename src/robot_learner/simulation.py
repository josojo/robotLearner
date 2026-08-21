"""Isolated simulation host, camera capture, and restricted scripts."""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import multiprocessing as mp
import os
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from robot_learner.models import (
    Action,
    ActionKind,
    JSONValue,
    Observation,
    new_id,
    utc_now,
)


class SimulationError(RuntimeError):
    """The simulation worker or environment rejected an operation."""


class ScriptValidationError(ValueError):
    """Generated script source is outside the restricted simulation grammar."""


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    env_id: str
    host: str = "cableties_sim:CableTieHost"
    seed: int | None = None
    worker_timeout_s: float = 300.0
    max_script_revisions: int = 3
    cameras: tuple[str, ...] = ()
    frame_width: int = 480
    frame_height: int = 360
    capture_stride: int = 40
    render_backend: str = "osmesa"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    config_dir: Path | None = None

    @classmethod
    def from_toml(cls, path: Path) -> SimulationSpec:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        data = raw.get("simulation")
        if not isinstance(data, dict):
            raise ValueError("configuration needs a [simulation] table")
        env_id = data.get("env_id")
        cameras = data.get("cameras", [])
        kwargs = data.get("env_kwargs", {})
        host = data.get("host", "cableties_sim:CableTieHost")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("simulation.env_id must be a non-empty string")
        if not isinstance(host, str) or ":" not in host:
            raise ValueError("simulation.host must be module:Class")
        if not isinstance(cameras, list) or not all(isinstance(x, str) for x in cameras):
            raise ValueError("simulation.cameras must be strings")
        if not isinstance(kwargs, dict) or not _is_json(kwargs):
            raise ValueError("simulation.env_kwargs must contain JSON values")
        resolved = dict(kwargs)
        asset = resolved.get("piper_asset_dir")
        if isinstance(asset, str) and asset:
            raw_path = Path(asset).expanduser()
            if not raw_path.is_absolute():
                resolved["piper_asset_dir"] = str((path.parent / raw_path).resolve())
        spec = cls(
            env_id=env_id,
            host=host,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            worker_timeout_s=float(data.get("worker_timeout_s", 300)),
            max_script_revisions=int(data.get("max_script_revisions", 3)),
            cameras=tuple(cameras),
            frame_width=int(data.get("frame_width", 480)),
            frame_height=int(data.get("frame_height", 360)),
            capture_stride=int(data.get("capture_stride", 40)),
            render_backend=str(data.get("render_backend", "osmesa")),
            env_kwargs=resolved,
            config_dir=path.parent.resolve(),
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


def script_from_actions(actions: list[Action], cameras: tuple[str, ...]) -> str:
    """Compile an explored action trace into a restricted checkpoint script."""
    lines: list[str] = []
    camera_literal = json.dumps(list(cameras))
    for action in actions:
        if action.kind is ActionKind.OBSERVE:
            selected = action.parameters.get("cameras")
            if isinstance(selected, list):
                lines.append(f"sim.observe(cameras={json.dumps(selected)})")
            else:
                lines.append(f"sim.observe(cameras={camera_literal})")
        elif action.kind is ActionKind.RUN_SKILL:
            name = action.parameters.get("name")
            if not isinstance(name, str):
                raise ScriptValidationError("run_skill action is missing a name")
            params = action.parameters.get("parameters") or {}
            if not isinstance(params, dict) or not params:
                lines.append(f"sim.run_skill({json.dumps(name)})")
            else:
                keywords = ", ".join(
                    f"{key}={json.dumps(value)}" for key, value in params.items()
                )
                lines.append(f"sim.run_skill({json.dumps(name)}, {keywords})")
        elif action.kind is ActionKind.STOP:
            lines.append("sim.stop()")
    if not lines:
        raise ScriptValidationError("no actions to compile")
    return "\n".join(lines) + "\n"


def _normalize_call(
    operation: str, positional: list[Any], keywords: dict[str, Any]
) -> dict[str, Any]:
    if operation == "observe":
        if len(positional) > 1 or (positional and "cameras" in keywords):
            raise ScriptValidationError("observe accepts one cameras argument")
        cameras = positional[0] if positional else keywords.pop("cameras", None)
        if keywords:
            raise ScriptValidationError("unexpected observe arguments")
        if cameras is None:
            return {}
        if not isinstance(cameras, list) or not all(isinstance(item, str) for item in cameras):
            raise ScriptValidationError("cameras must be a list of strings")
        return {"cameras": cameras}
    if operation == "run_skill":
        if not positional or not isinstance(positional[0], str):
            raise ScriptValidationError("run_skill needs a skill name")
        if len(positional) > 2:
            raise ScriptValidationError("run_skill needs a skill name")
        if len(positional) == 2 and keywords:
            raise ScriptValidationError("run_skill cannot mix a mapping and keywords")
        if keywords and set(keywords) != {"parameters"}:
            parameters = keywords
        else:
            parameters = keywords.get("parameters", positional[1] if len(positional) == 2 else {})
        if not isinstance(parameters, dict):
            raise ScriptValidationError("run_skill needs a string and parameter object")
        return {"name": positional[0], "parameters": parameters}
    if positional or keywords:
        raise ScriptValidationError("stop accepts no arguments")
    return {}


def checkpoint_id_for(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value or value in {".", ".."}:
        raise ScriptValidationError(f"invalid checkpoint id {value!r}")
    return value


class SimulationClient:
    """Parent-side RobotAdapter over one spawned simulation host."""

    def __init__(
        self,
        spec: SimulationSpec,
        artifact_dir: Path,
        *,
        worker: Callable[..., None] | None = None,
    ) -> None:
        self.spec = spec
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._request_id = 0
        self._closed = False
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        target = worker or _worker_main
        self._process: BaseProcess | None = context.Process(
            target=target,
            args=(child, asdict(spec), str(self.artifact_dir)),
            name=f"robot-learner-{spec.env_id}",
        )
        self._process.start()
        child.close()
        try:
            ready = self._receive()
        except Exception:
            self.close(force=True)
            raise
        if ready.get("type") != "ready":
            self.close(force=True)
            raise SimulationError(str(ready.get("error", "worker did not become ready")))
        self.manifest = ready["manifest"]

    def observe(self, cameras: tuple[str, ...] | None = None) -> Observation:
        selected = list(self.spec.cameras if cameras is None else cameras)
        payload = self._request("observe", cameras=selected)
        frames = payload.get("frames") or []
        refs = tuple(str(frame["artifact_ref"]) for frame in frames if "artifact_ref" in frame)
        context: dict[str, JSONValue] = {
            "phase": payload.get("state", {}).get("phase"),
            "unstable": payload.get("state", {}).get("unstable"),
            "privileged": payload.get("state", {}).get("privileged") or {},
            "observation": payload.get("state", {}).get("observation") or {},
            "frames": frames,
        }
        return Observation(new_id("obs"), utc_now(), context, refs)

    def execute(self, action: Action) -> dict[str, JSONValue]:
        if action.kind is ActionKind.OBSERVE:
            cameras = action.parameters.get("cameras")
            selected = tuple(str(item) for item in cameras) if isinstance(cameras, list) else None
            observation = self.observe(selected)
            return {
                "ok": True,
                "observation_id": observation.id,
                "artifact_refs": list(observation.artifact_refs),
                "phase": observation.context.get("phase"),
                "unstable": observation.context.get("unstable"),
            }
        if action.kind is ActionKind.RUN_SKILL:
            name = action.parameters.get("name")
            if not isinstance(name, str):
                raise SimulationError("run_skill requires a name")
            parameters = action.parameters.get("parameters") or {}
            if not isinstance(parameters, dict):
                raise SimulationError("run_skill parameters must be an object")
            return self.run_skill(name, parameters)
        if action.kind is ActionKind.STOP:
            self.stop()
            return {"ok": True}
        raise SimulationError(f"unsupported simulation action {action.kind}")

    def run_skill(
        self, name: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, JSONValue]:
        payload = self._request("run_skill", name=name, parameters=parameters or {})
        result = payload.get("result") or {}
        ok = bool(result.get("ok")) and not bool(result.get("unstable"))
        return {
            "ok": ok,
            "name": name,
            "unstable": bool(result.get("unstable")),
            "phase": result.get("phase"),
            "error": result.get("error"),
            "state": payload.get("state") or {},
        }

    def snapshot(self) -> int:
        payload = self._request("snapshot")
        return int(payload["snapshot_id"])

    def restore(self, snapshot_id: int) -> dict[str, JSONValue]:
        payload = self._request("restore", snapshot_id=int(snapshot_id))
        return payload.get("state") or {}

    def run_script(self, script: GeneratedScript) -> dict[str, Any]:
        checkpoint_id = checkpoint_id_for(script.checkpoint_id)
        commands = [asdict(command) for command in parse_restricted_script(script.source)]
        script_dir = self.artifact_dir / "scripts" / checkpoint_id
        script_dir.mkdir(parents=True, exist_ok=True)
        destination = script_dir / f"v{script.version}.py"
        destination.write_text(script.source.rstrip() + "\n", encoding="utf-8")
        return self._request(
            "run_script",
            checkpoint_id=checkpoint_id,
            version=script.version,
            commands=commands,
        )

    def persist_script(self, script: GeneratedScript) -> Path:
        checkpoint_id = checkpoint_id_for(script.checkpoint_id)
        script_dir = self.artifact_dir / "scripts" / checkpoint_id
        script_dir.mkdir(parents=True, exist_ok=True)
        destination = script_dir / f"v{script.version}.py"
        destination.write_text(script.source.rstrip() + "\n", encoding="utf-8")
        return destination

    def stop(self) -> None:
        if self._process is None or not self._process.is_alive():
            return
        self._request("stop")

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        process = getattr(self, "_process", None)
        if process is None:
            self._closed = True
            return
        try:
            if process.is_alive() and not force:
                try:
                    self._send({"operation": "close"})
                    if self._connection.poll(5):
                        self._connection.recv()
                except (BrokenPipeError, EOFError, OSError, SimulationError):
                    force = True
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        finally:
            with contextlib.suppress(OSError):
                self._connection.close()
            self._process = None
            self._closed = True

    def __enter__(self) -> SimulationClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(self, payload: dict[str, Any]) -> int:
        self._request_id += 1
        payload = {**payload, "request_id": self._request_id}
        self._connection.send(payload)
        return self._request_id

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if self._process is None or not self._process.is_alive():
            raise SimulationError("simulation worker is not running")
        request_id = self._send({"operation": operation, **payload})
        response = self._receive()
        if response.get("type") == "error":
            raise SimulationError(str(response.get("error", "simulation operation failed")))
        if response.get("request_id") not in (None, request_id):
            raise SimulationError("simulation worker returned a stale response")
        return response

    def _receive(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.spec.worker_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close(force=True)
                raise SimulationError("simulation worker timed out")
            if self._connection.poll(min(0.2, remaining)):
                break
            if self._process is None or not self._process.is_alive():
                raise SimulationError("simulation worker exited")
        try:
            response = self._connection.recv()
        except (EOFError, OSError) as exc:
            self.close(force=True)
            raise SimulationError("simulation worker closed the pipe") from exc
        if not isinstance(response, dict):
            raise SimulationError("simulation worker returned an invalid response")
        return response


def _spec_from_worker_payload(raw_spec: dict[str, Any]) -> SimulationSpec:
    data = dict(raw_spec)
    data.pop("config_dir", None)
    cameras = data.get("cameras")
    if isinstance(cameras, list):
        data["cameras"] = tuple(cameras)
    return SimulationSpec(**data)


def _worker_main(connection: Connection, raw_spec: dict[str, Any], artifact_dir: str) -> None:
    host: Any = None
    try:
        spec = _spec_from_worker_payload(raw_spec)
        os.environ["MUJOCO_GL"] = spec.render_backend
        module_name, _, class_name = spec.host.partition(":")
        host_cls = getattr(importlib.import_module(module_name), class_name)
        from PIL import Image

        kwargs = dict(spec.env_kwargs)
        kwargs.update(
            seed=spec.seed,
            width=spec.frame_width,
            height=spec.frame_height,
        )
        host = host_cls(**kwargs)
        available = tuple(host.camera_names())
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
        stride_log: list[dict[str, Any]] = []

        def capture(phase: str, selected: tuple[str, ...]) -> list[dict[str, Any]]:
            nonlocal frame_index
            results = []
            for camera in selected:
                rgb = host.render_camera(camera)
                destination = frames_dir / camera / f"{frame_index:08d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(destination)
                results.append(
                    {
                        "camera": camera,
                        "simulation_time": float(host.data.time),
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
                privileged = host.privileged_state()
                frames = capture(str(privileged.get("phase")), cameras)
                stride_log.append(
                    {
                        "simulation_time": privileged.get("time"),
                        "phase": privileged.get("phase"),
                        "frames": frames,
                    }
                )

        host.set_step_callback(on_step)
        manifest = host.capability_manifest()
        manifest.update(
            {
                "configured_cameras": list(cameras),
                "seed": spec.seed,
                "max_script_revisions": spec.max_script_revisions,
                "initial_state": _jsonable(host.privileged_state()),
            }
        )
        connection.send({"type": "ready", "manifest": manifest, "request_id": None})
        while True:
            request = connection.recv()
            request_id = request.get("request_id")
            operation = request.get("operation")
            if operation == "close":
                connection.send({"type": "closed", "request_id": request_id})
                break
            if operation == "stop":
                connection.send(
                    {
                        "type": "stopped",
                        "request_id": request_id,
                        "time": float(host.data.time),
                    }
                )
                continue
            if operation == "observe":
                selected = _selected_cameras(request.get("cameras"), cameras, available)
                state = _state(host)
                connection.send(
                    {
                        "type": "observation",
                        "request_id": request_id,
                        "frames": capture(str(state["phase"]), selected),
                        "state": state,
                    }
                )
                continue
            if operation == "run_skill":
                result = _skill_dict(host.run_skill(request.get("name"), request.get("parameters")))
                connection.send(
                    {
                        "type": "skill_result",
                        "request_id": request_id,
                        "result": result,
                        "state": _state(host),
                    }
                )
                continue
            if operation == "snapshot":
                connection.send(
                    {
                        "type": "snapshot",
                        "request_id": request_id,
                        "snapshot_id": int(host.snapshot()),
                    }
                )
                continue
            if operation == "restore":
                state = host.restore(int(request["snapshot_id"]))
                connection.send(
                    {
                        "type": "restored",
                        "request_id": request_id,
                        "state": _jsonable(state),
                    }
                )
                continue
            if operation == "run_script":
                results: list[dict[str, Any]] = []
                for raw in request.get("commands", []):
                    command = ScriptCommand(**raw)
                    try:
                        if command.operation == "observe":
                            if "cameras" in command.arguments:
                                selected = _selected_cameras(
                                    command.arguments.get("cameras"), cameras, available
                                )
                            else:
                                selected = cameras
                            results.append(
                                {
                                    "operation": "observe",
                                    "ok": True,
                                    "frames": capture(str(_state(host)["phase"]), selected),
                                }
                            )
                        elif command.operation == "run_skill":
                            skill = _skill_dict(
                                host.run_skill(
                                    command.arguments["name"],
                                    command.arguments.get("parameters"),
                                )
                            )
                            ok = bool(skill.get("ok")) and not bool(skill.get("unstable"))
                            results.append(
                                {
                                    "operation": "run_skill",
                                    "name": command.arguments["name"],
                                    "ok": ok,
                                    "error": skill.get("error"),
                                }
                            )
                            if not ok:
                                break
                        else:
                            results.append({"operation": "stop", "ok": True})
                            break
                    except (TypeError, ValueError) as exc:
                        results.append(
                            {
                                "operation": command.operation,
                                "ok": False,
                                "error": str(exc),
                            }
                        )
                        break
                state = _state(host)
                connection.send(
                    {
                        "type": "script_result",
                        "request_id": request_id,
                        "checkpoint_id": request.get("checkpoint_id"),
                        "version": request.get("version"),
                        "results": results,
                        "state": state,
                        "stride_frames": list(stride_log),
                    }
                )
                stride_log.clear()
                continue
            raise ValueError(f"unknown worker operation {operation!r}")
    except EOFError:
        pass
    except Exception as exc:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if host is not None:
            with contextlib.suppress(Exception):
                host.close()
        with contextlib.suppress(OSError):
            connection.close()


def _selected_cameras(
    requested: Any, default: tuple[str, ...], available: tuple[str, ...]
) -> tuple[str, ...]:
    if requested is None:
        return default
    selected = tuple(requested)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown cameras: {sorted(unknown)}")
    return selected


def _skill_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "as_dict"):
        return dict(result.as_dict())
    if isinstance(result, dict):
        return result
    return {"ok": bool(result), "unstable": False, "error": None}


def _state(host: Any) -> dict[str, Any]:
    privileged = host.privileged_state()
    observation = host.observation()
    return {
        "simulation_time": float(host.data.time),
        "phase": str(privileged.get("phase")),
        "unstable": bool(privileged.get("unstable")),
        "observation": _jsonable(observation),
        "privileged": _jsonable(privileged),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True
