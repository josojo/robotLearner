"""Typed, serializable domain models for the harness boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeAlias
from uuid import uuid4

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class ActionKind(StrEnum):
    OBSERVE = "observe"
    MOVE_CARTESIAN = "move_cartesian"
    GRASP = "grasp"
    RELEASE = "release"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    WAIT = "wait"
    STOP = "stop"


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    captured_at: datetime
    context: dict[str, JSONValue] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Predicate:
    name: str
    parameters: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    parameters: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyContract:
    max_velocity_m_s: float
    max_force_n: float
    max_duration_s: float
    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    requires_human_approval: bool = True


@dataclass(frozen=True, slots=True)
class Strategy:
    id: str
    checkpoint_id: str
    required_skills: tuple[str, ...]
    preconditions: tuple[Predicate, ...]
    safety_contract: SafetyContract
    actions: tuple[Action, ...]
    version: int = 1
    parent_strategy_id: str | None = None
    trusted: bool = False


@dataclass(frozen=True, slots=True)
class Checkpoint:
    id: str
    name: str
    success_predicate: Predicate
    dependencies: tuple[str, ...] = ()
    recovery_options: tuple[str, ...] = ()
    fallback: str = "stop_for_human_review"


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    goal: str
    checkpoints: tuple[Checkpoint, ...]
    constraints: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationResult:
    outcome: Outcome
    reason: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ActionRecord:
    action: Action
    started_at: datetime
    finished_at: datetime
    response: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    id: str
    task_id: str
    checkpoint_id: str
    strategy_id: str
    started_at: datetime
    finished_at: datetime
    start_observation: Observation
    actions: tuple[ActionRecord, ...]
    observations: tuple[Observation, ...]
    verification: VerificationResult
    safety_events: tuple[str, ...] = ()

