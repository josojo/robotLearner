"""Simulation explorer: catalog + privileged state + camera frames → one run_skill."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_learner.models import (
    Action,
    ActionKind,
    Checkpoint,
    JSONValue,
    Observation,
    Task,
)
from robot_learner.ports import LanguageModel
from robot_learner.simulation import (
    GeneratedScript,
    ScriptCommand,
    ScriptValidationError,
    SimulationClient,
    SimulationError,
    parse_restricted_script,
    script_from_actions,
)

EXPLORER_SYSTEM_PROMPT = """You choose the next restricted simulation action for a robot checkpoint.

Camera frames from the current observation are attached as images when available.
Use those pictures together with the privileged state to choose the skill.

Return ONLY a restricted script. The only legal statements are literal calls:
  sim.run_skill("skill_name")
  sim.run_skill("skill_name", param=value)

Rules:
- Emit exactly one sim.run_skill(...) that advances the current checkpoint.
- Use only skill names from the provided catalog.
- Arguments must be JSON literals (strings, numbers, booleans, lists, objects, null).
- Do not call sim.observe, sim.stop, or any other name.
- Do not import, assign, loop, comment, or write prose.
- Do not emit raw joint or Cartesian motion; skills are the only legal actions.
"""


class ExplorationError(RuntimeError):
    """The explorer could not produce a valid checkpoint script."""


@dataclass(frozen=True, slots=True)
class CheckpointAttempt:
    revision: int
    source: str
    ok: bool
    error: str | None = None
    phase: JSONValue = None
    skill: str | None = None
    state: JSONValue = None


@dataclass(frozen=True, slots=True)
class CheckpointExploration:
    checkpoint_id: str
    ok: bool
    skill: str | None
    script_path: str | None
    attempts: tuple[CheckpointAttempt, ...]
    start_frame: int | None = None
    end_frame: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExplorationReport:
    task_id: str
    goal: str
    ok: bool
    checkpoints: tuple[CheckpointExploration, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "ok": self.ok,
            "checkpoints": [
                {
                    "checkpoint_id": item.checkpoint_id,
                    "ok": item.ok,
                    "skill": item.skill,
                    "script_path": item.script_path,
                    "start_frame": item.start_frame,
                    "end_frame": item.end_frame,
                    "error": item.error,
                    "attempts": [
                        {
                            "revision": attempt.revision,
                            "source": attempt.source,
                            "ok": attempt.ok,
                            "error": attempt.error,
                            "phase": attempt.phase,
                            "skill": attempt.skill,
                            "state": attempt.state,
                        }
                        for attempt in item.attempts
                    ],
                }
                for item in self.checkpoints
            ],
        }


class SimulationExplorer:
    """Walk a plan on one SimulationClient and persist a script per checkpoint."""

    def __init__(self, simulation: SimulationClient, language_model: LanguageModel) -> None:
        self._simulation = simulation
        self._language_model = language_model
        self._catalog = _skill_catalog(simulation.manifest)

    def explore(self, task: Task) -> ExplorationReport:
        if not task.checkpoints:
            raise ExplorationError("plan has no checkpoints")
        completed: list[tuple[Checkpoint, str]] = []
        records: list[CheckpointExploration] = []
        for checkpoint in task.checkpoints:
            record = self._explore_checkpoint(task, checkpoint, completed)
            records.append(record)
            if not record.ok or record.skill is None:
                break
            completed.append((checkpoint, record.skill))
        return ExplorationReport(
            task_id=task.id,
            goal=task.goal,
            ok=bool(records) and len(records) == len(task.checkpoints) and records[-1].ok,
            checkpoints=tuple(records),
        )

    def _explore_checkpoint(
        self,
        task: Task,
        checkpoint: Checkpoint,
        completed: list[tuple[Checkpoint, str]],
    ) -> CheckpointExploration:
        before = self._simulation.observe()
        start_frame = _frame_index(before)
        end_frame = start_frame
        snapshot_id = self._simulation.snapshot()
        attempts: list[CheckpointAttempt] = []
        revisions = max(1, int(self._simulation.spec.max_script_revisions))
        for revision in range(1, revisions + 1):
            prompt = _user_prompt(
                task,
                checkpoint,
                completed,
                self._catalog,
                before,
                attempts,
            )
            images = observation_image_paths(before)
            raw = self._language_model.complete(
                prompt,
                system_prompt=EXPLORER_SYSTEM_PROMPT,
                images=images or None,
            )
            try:
                source = extract_restricted_script(raw)
                command = _exactly_one_run_skill(parse_restricted_script(source), self._catalog)
            except ScriptValidationError as exc:
                attempts.append(
                    CheckpointAttempt(
                        revision=revision,
                        source=raw.strip(),
                        ok=False,
                        error=str(exc),
                        phase=before.context.get("phase"),
                        state=_compact_state(before),
                    )
                )
                continue
            source = _command_source(command)
            action = _run_skill_action(command)
            try:
                result = self._simulation.execute(action)
            except SimulationError as exc:
                self._simulation.restore(snapshot_id)
                before = self._simulation.observe()
                attempts.append(
                    CheckpointAttempt(
                        revision=revision,
                        source=source,
                        ok=False,
                        error=str(exc),
                        phase=before.context.get("phase"),
                        skill=str(command.arguments["name"]),
                        state=_compact_state(before),
                    )
                )
                continue
            after = self._simulation.observe()
            end_frame = _frame_index(after) or end_frame
            failure = _failure_reason(result, after)
            if failure is None:
                compiled = script_from_actions(
                    [Action(ActionKind.OBSERVE), action, Action(ActionKind.OBSERVE)],
                    self._simulation.spec.cameras,
                )
                path = self._simulation.persist_script(
                    GeneratedScript(checkpoint_id=checkpoint.id, source=compiled, version=1)
                )
                attempts.append(
                    CheckpointAttempt(
                        revision=revision,
                        source=source,
                        ok=True,
                        phase=after.context.get("phase"),
                        skill=str(command.arguments["name"]),
                        state=_compact_state(after),
                    )
                )
                return CheckpointExploration(
                    checkpoint_id=checkpoint.id,
                    ok=True,
                    skill=str(command.arguments["name"]),
                    script_path=str(path),
                    attempts=tuple(attempts),
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            self._simulation.restore(snapshot_id)
            before = self._simulation.observe()
            attempts.append(
                CheckpointAttempt(
                    revision=revision,
                    source=source,
                    ok=False,
                    error=failure,
                    phase=after.context.get("phase"),
                    skill=str(command.arguments["name"]),
                    state=_compact_state(after),
                )
            )
        last = attempts[-1] if attempts else None
        last_skill = next((item.skill for item in reversed(attempts) if item.skill), None)
        return CheckpointExploration(
            checkpoint_id=checkpoint.id,
            ok=False,
            skill=last_skill,
            script_path=None,
            attempts=tuple(attempts),
            start_frame=start_frame,
            end_frame=end_frame,
            error=last.error if last else "no skill attempts",
        )


_IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


def _frame_index(observation: Observation) -> int | None:
    for ref in observation.artifact_refs:
        stem = Path(str(ref)).stem
        if stem.isdigit():
            return int(stem)
    return None


def observation_image_paths(observation: Observation) -> tuple[str, ...]:
    """Filesystem paths of observe-camera images, in capture order."""
    paths: list[str] = []
    seen: set[str] = set()
    frames = observation.context.get("frames")
    candidates: list[object] = []
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict):
                candidates.append(frame.get("artifact_ref"))
    candidates.extend(observation.artifact_refs)
    for raw in candidates:
        if not isinstance(raw, str) or raw in seen:
            continue
        path = Path(raw)
        if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.is_file():
            continue
        seen.add(raw)
        paths.append(raw)
    return tuple(paths)


def extract_restricted_script(text: str) -> str:
    """Strip markdown fences so parse_restricted_script can see the calls."""
    candidate = text.strip()
    if not candidate:
        raise ScriptValidationError("model returned an empty script")
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:python|py)?\s*", "", candidate, count=1)
        candidate = re.sub(r"\s*```$", "", candidate, count=1)
    return candidate.strip() + "\n"


def _skill_catalog(manifest: dict[str, Any]) -> dict[str, Any]:
    skills = manifest.get("skills")
    if not isinstance(skills, dict) or not skills:
        raise ExplorationError("simulation manifest does not advertise any skills")
    catalog: dict[str, Any] = {}
    for name, spec in skills.items():
        if not isinstance(name, str) or not name:
            continue
        catalog[name] = spec if isinstance(spec, dict) else {}
    if not catalog:
        raise ExplorationError("simulation manifest does not advertise any skills")
    return catalog


def _exactly_one_run_skill(
    commands: tuple[ScriptCommand, ...], catalog: dict[str, Any]
) -> ScriptCommand:
    if any(command.operation != "run_skill" for command in commands):
        raise ScriptValidationError("explorer scripts may contain only sim.run_skill(...)")
    if len(commands) != 1:
        raise ScriptValidationError("explorer scripts must contain exactly one sim.run_skill(...)")
    command = commands[0]
    name = command.arguments.get("name")
    if not isinstance(name, str) or name not in catalog:
        raise ScriptValidationError(f"skill {name!r} is not in the simulation catalog")
    return command


def _run_skill_action(command: ScriptCommand) -> Action:
    parameters = command.arguments.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ScriptValidationError("run_skill parameters must be an object")
    return Action(
        ActionKind.RUN_SKILL,
        {"name": str(command.arguments["name"]), "parameters": parameters},
    )


def _command_source(command: ScriptCommand) -> str:
    return script_from_actions([_run_skill_action(command)], ())


def _failure_reason(result: dict[str, JSONValue], after: Observation) -> str | None:
    if result.get("ok") and not after.context.get("unstable"):
        return None
    parts: list[str] = []
    name = result.get("name")
    if isinstance(name, str) and name:
        parts.append(name)
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        parts.append(error.strip())
    elif not result.get("ok"):
        parts.append("skill returned ok=false")
    if after.context.get("unstable") or result.get("unstable"):
        parts.append("scene unstable")
    state = _compact_state(after)
    for key in ("phase", "held_tie", "current_tie", "tie_z"):
        value = state.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "skill failed"


def _compact_state(observation: Observation) -> dict[str, JSONValue]:
    privileged = observation.context.get("privileged")
    if not isinstance(privileged, dict):
        privileged = {}
    current = privileged.get("current_tie")
    tie_z: JSONValue = None
    ties = privileged.get("ties")
    if isinstance(current, int) and isinstance(ties, list):
        for item in ties:
            if isinstance(item, dict) and item.get("index") == current:
                z_value = item.get("z")
                if isinstance(z_value, (int, float)):
                    tie_z = float(z_value)
                break
    return {
        "phase": observation.context.get("phase"),
        "unstable": observation.context.get("unstable"),
        "held_tie": privileged.get("held_tie"),
        "current_tie": current,
        "tie_z": tie_z,
        "tcp": privileged.get("tcp"),
        "holder_locked": privileged.get("holder_locked"),
        "candidate_ties": privileged.get("candidate_ties"),
    }


def _user_prompt(
    task: Task,
    checkpoint: Checkpoint,
    completed: list[tuple[Checkpoint, str]],
    catalog: dict[str, Any],
    observation: Observation,
    attempts: list[CheckpointAttempt],
) -> str:
    done_ids = {item.id for item, _ in completed}
    remaining = [
        item.id
        for item in task.checkpoints
        if item.id not in done_ids and item.id != checkpoint.id
    ]
    sections = [
        f"Task: {task.goal}",
        f"Current checkpoint: {checkpoint.id}",
        f"Checkpoint name: {checkpoint.name}",
        "Success predicate: "
        + json.dumps(
            {
                "name": checkpoint.success_predicate.name,
                "parameters": checkpoint.success_predicate.parameters,
            },
            sort_keys=True,
        ),
        "Completed checkpoints: "
        + (
            json.dumps(
                [{"id": item.id, "skill": skill} for item, skill in completed],
                sort_keys=True,
            )
            if completed
            else "none"
        ),
        "Remaining checkpoints: " + (", ".join(remaining) if remaining else "none"),
        "Skill catalog:\n" + json.dumps(catalog, indent=2, sort_keys=True),
        "Privileged state:\n"
        + json.dumps(_privileged_payload(observation), indent=2, sort_keys=True, default=str),
        _camera_section(observation),
    ]
    if attempts:
        last = attempts[-1]
        sections.append(
            "Previous attempt failed. Restore already happened; pick a different skill.\n"
            + json.dumps(
                {
                    "revision": last.revision,
                    "source": last.source,
                    "error": last.error,
                    "phase": last.phase,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    sections.append("Which sim.run_skill(...) next?")
    return "\n\n".join(sections)


def _camera_section(observation: Observation) -> str:
    images = observation_image_paths(observation)
    if not images:
        return "Camera frames: none attached"
    labels: list[str] = []
    frames = observation.context.get("frames")
    by_ref: dict[str, str] = {}
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict) and isinstance(frame.get("artifact_ref"), str):
                camera = frame.get("camera")
                if isinstance(camera, str) and camera:
                    by_ref[str(frame["artifact_ref"])] = camera
    for path in images:
        camera = by_ref.get(path) or Path(path).parent.name
        labels.append(f"{camera}: {Path(path).name}")
    return "Camera frames attached in order:\n" + "\n".join(f"- {item}" for item in labels)


def _privileged_payload(observation: Observation) -> dict[str, JSONValue]:
    privileged = observation.context.get("privileged")
    if not isinstance(privileged, dict):
        privileged = {}
    return {
        "phase": observation.context.get("phase"),
        "unstable": observation.context.get("unstable"),
        "privileged": privileged,
    }
