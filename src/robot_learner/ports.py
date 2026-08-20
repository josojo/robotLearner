"""Extension interfaces for hardware and intelligence providers."""

from collections.abc import Sequence
from typing import Protocol

from robot_learner.models import (
    Action,
    Checkpoint,
    ExecutionTrace,
    JSONValue,
    Observation,
    Strategy,
    Task,
    VerificationResult,
)


class RobotAdapter(Protocol):
    def observe(self) -> Observation: ...

    def execute(self, action: Action) -> dict[str, JSONValue]: ...

    def stop(self) -> None: ...


class Verifier(Protocol):
    def evaluate(self, checkpoint: Checkpoint, trace: ExecutionTrace) -> VerificationResult: ...


class LanguageModel(Protocol):
    """A narrow text-generation boundary used by the orchestration harness."""

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str: ...


class StrategyRepository(Protocol):
    def candidates(self, checkpoint_id: str, observation: Observation) -> Sequence[Strategy]: ...

    def record(self, strategy: Strategy, trace: ExecutionTrace) -> None: ...


class Planner(Protocol):
    def propose(self, task: Task, observation: Observation) -> Sequence[Checkpoint]: ...


class Synthesizer(Protocol):
    def create(
        self,
        checkpoint: Checkpoint,
        observation: Observation,
        related: Sequence[Strategy],
    ) -> Strategy: ...

