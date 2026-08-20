"""Create a validated checkpoint graph from a natural-language task prompt."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from robot_learner.models import Checkpoint, JSONValue, Predicate, Task, new_id
from robot_learner.ports import LanguageModel


class PlanError(ValueError):
    """The language model returned an invalid checkpoint plan."""


PLANNER_SYSTEM_PROMPT = """You are a checkpoint planner for a robot learning harness.
Turn the user's task into small, ordered, independently verifiable checkpoints.

Return ONLY one JSON object with this shape:
{
  "checkpoints": [
    {
      "id": "short_snake_case_id",
      "name": "Human-readable achieved state",
      "success_predicate": {
        "name": "observable_snake_case_condition",
        "parameters": {}
      },
      "dependencies": ["earlier_checkpoint_id"],
      "recovery_options": ["bounded recovery description"],
      "fallback": "stop_for_human_review"
    }
  ]
}

Rules:
- Describe achieved states, not vague activities.
- Use fine-grained checkpoints; do not merge distinct manipulation steps.
- Every success predicate must be independently observable after the step.
- Keep the graph linear unless the task clearly requires branching.
- Dependencies may refer only to checkpoints earlier in the list.
- Do not emit executable robot code, coordinates, force limits, or arbitrary commands.
- Use stop_for_human_review when safe recovery cannot be stated.
"""

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class CheckpointPlanner:
    """Language-model-backed implementation of the planner boundary."""

    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def create_task(self, prompt: str) -> Task:
        goal = prompt.strip()
        if not goal:
            raise ValueError("task prompt must not be empty")
        response = self._language_model.complete(goal, system_prompt=PLANNER_SYSTEM_PROMPT)
        checkpoints = _parse_checkpoints(_extract_json_object(response))
        return Task(id=new_id("task"), goal=goal, checkpoints=checkpoints)


def _extract_json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, count=1)
        candidate = re.sub(r"\s*```$", "", candidate, count=1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PlanError(f"planner did not return valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise PlanError("planner response must be a JSON object")
    return value


def _parse_checkpoints(raw: Mapping[str, Any]) -> tuple[Checkpoint, ...]:
    items = raw.get("checkpoints")
    if not isinstance(items, list) or not items:
        raise PlanError("plan must contain a non-empty checkpoints list")
    checkpoints: list[Checkpoint] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise PlanError(f"checkpoint {index + 1} must be an object")
        checkpoint = _parse_checkpoint(item, index)
        if checkpoint.id in seen:
            raise PlanError(f"duplicate checkpoint id: {checkpoint.id}")
        unknown = set(checkpoint.dependencies) - seen
        if unknown:
            names = ", ".join(sorted(unknown))
            raise PlanError(
                f"checkpoint {checkpoint.id} has unknown or forward dependencies: {names}"
            )
        seen.add(checkpoint.id)
        checkpoints.append(checkpoint)
    return tuple(checkpoints)


def _parse_checkpoint(raw: Mapping[str, Any], index: int) -> Checkpoint:
    checkpoint_id = _required_string(raw, "id", index)
    if not _IDENTIFIER.fullmatch(checkpoint_id):
        raise PlanError(f"checkpoint id must be snake_case: {checkpoint_id!r}")
    name = _required_string(raw, "name", index)
    predicate_raw = raw.get("success_predicate")
    if not isinstance(predicate_raw, dict):
        raise PlanError(f"checkpoint {checkpoint_id} needs a success_predicate object")
    predicate_name = _required_string(predicate_raw, "name", index)
    if not _IDENTIFIER.fullmatch(predicate_name):
        raise PlanError(f"predicate name must be snake_case: {predicate_name!r}")
    parameters = predicate_raw.get("parameters", {})
    if not isinstance(parameters, dict) or not _is_json_value(parameters):
        raise PlanError(f"checkpoint {checkpoint_id} predicate parameters must be JSON")
    dependencies = _string_tuple(raw.get("dependencies", ()), "dependencies", checkpoint_id)
    recoveries = _string_tuple(raw.get("recovery_options", ()), "recovery_options", checkpoint_id)
    fallback = raw.get("fallback", "stop_for_human_review")
    if not isinstance(fallback, str) or not fallback.strip():
        raise PlanError(f"checkpoint {checkpoint_id} fallback must be a non-empty string")
    return Checkpoint(
        id=checkpoint_id,
        name=name,
        success_predicate=Predicate(predicate_name, parameters),
        dependencies=dependencies,
        recovery_options=recoveries,
        fallback=fallback.strip(),
    )


def _required_string(raw: Mapping[str, Any], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"checkpoint {index + 1} needs a non-empty {key}")
    return value.strip()


def _string_tuple(value: object, key: str, checkpoint_id: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise PlanError(f"checkpoint {checkpoint_id} {key} must be a list of strings")
    return tuple(item.strip() for item in value)


def _is_json_value(value: object) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def task_to_dict(task: Task) -> dict[str, JSONValue]:
    """Convert a generated task into its stable artifact representation."""
    return {
        "id": task.id,
        "goal": task.goal,
        "constraints": list(task.constraints),
        "input_refs": list(task.input_refs),
        "checkpoints": [
            {
                "id": checkpoint.id,
                "name": checkpoint.name,
                "success_predicate": {
                    "name": checkpoint.success_predicate.name,
                    "parameters": checkpoint.success_predicate.parameters,
                },
                "dependencies": list(checkpoint.dependencies),
                "recovery_options": list(checkpoint.recovery_options),
                "fallback": checkpoint.fallback,
            }
            for checkpoint in task.checkpoints
        ],
    }
