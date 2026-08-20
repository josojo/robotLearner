"""Interruptible execution of validated action sequences."""

from threading import Event

from robot_learner.models import Action, ActionKind, ActionRecord, JSONValue, Observation, utc_now
from robot_learner.ports import RobotAdapter
from robot_learner.safety import ValidatedStrategy


class ExecutionStopped(RuntimeError):
    pass


class StrategyExecutor:
    def __init__(self, adapter: RobotAdapter) -> None:
        self._adapter = adapter
        self._stop_requested = Event()

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._adapter.stop()

    def run(
        self, validated: ValidatedStrategy
    ) -> tuple[tuple[ActionRecord, ...], tuple[Observation, ...]]:
        records: list[ActionRecord] = []
        observations: list[Observation] = []
        for action in validated.strategy.actions:
            if self._stop_requested.is_set():
                raise ExecutionStopped("execution interrupted")
            started_at = utc_now()
            response = self._adapter.execute(action)
            records.append(ActionRecord(action, started_at, utc_now(), response))
            if action.kind is ActionKind.OBSERVE:
                observations.append(self._adapter.observe())
        return tuple(records), tuple(observations)


class DryRunRobot:
    """Deterministic adapter for demos and tests; never controls hardware."""

    def __init__(self) -> None:
        self.stopped = False

    def observe(self) -> Observation:
        from robot_learner.models import new_id

        return Observation(new_id("obs"), utc_now(), {"mode": "dry_run", "visible": True})

    def execute(self, action: Action) -> dict[str, JSONValue]:
        if self.stopped:
            raise ExecutionStopped("dry-run adapter is stopped")
        return {"accepted": True, "dry_run": True}

    def stop(self) -> None:
        self.stopped = True

