"""Observable orchestration of one checkpoint transition."""

from dataclasses import replace

from robot_learner.executor import StrategyExecutor
from robot_learner.models import (
    Checkpoint,
    ExecutionTrace,
    Outcome,
    Strategy,
    Task,
    VerificationResult,
    new_id,
    utc_now,
)
from robot_learner.ports import RobotAdapter, Verifier
from robot_learner.safety import SafetyValidator
from robot_learner.tracing import JsonTraceRecorder


class LearningHarness:
    def __init__(
        self,
        adapter: RobotAdapter,
        safety: SafetyValidator,
        verifier: Verifier,
        recorder: JsonTraceRecorder,
    ) -> None:
        self._adapter = adapter
        self._safety = safety
        self._verifier = verifier
        self._recorder = recorder

    def run_checkpoint(
        self,
        task: Task,
        checkpoint: Checkpoint,
        strategy: Strategy,
        *,
        human_approved: bool = False,
    ) -> ExecutionTrace:
        if strategy.checkpoint_id != checkpoint.id:
            raise ValueError("strategy does not target the requested checkpoint")
        validated = self._safety.validate(task.id, strategy, human_approved=human_approved)
        started_at = utc_now()
        start_observation = self._adapter.observe()
        actions, observations = StrategyExecutor(self._adapter).run(validated)
        provisional = ExecutionTrace(
            id=new_id("exec"),
            task_id=task.id,
            checkpoint_id=checkpoint.id,
            strategy_id=strategy.id,
            started_at=started_at,
            finished_at=utc_now(),
            start_observation=start_observation,
            actions=actions,
            observations=observations,
            verification=VerificationResult(Outcome.UNCERTAIN, "verification pending", 0.0),
        )
        trace = replace(provisional, verification=self._verifier.evaluate(checkpoint, provisional))
        self._recorder.record(trace)
        return trace

