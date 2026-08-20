"""Dry-run demonstration entry point."""

import argparse
from pathlib import Path

from robot_learner.config import load_settings
from robot_learner.executor import DryRunRobot
from robot_learner.harness import LearningHarness
from robot_learner.models import Action, ActionKind, Checkpoint, Predicate, Strategy, Task
from robot_learner.safety import SafetyValidator
from robot_learner.tracing import JsonTraceRecorder
from robot_learner.verification import ContextPredicateVerifier


def run_demo(config_path: Path) -> int:
    settings = load_settings(config_path)
    checkpoint = Checkpoint("observe_scene", "Observe scene", Predicate("visible"))
    task = Task("demo_pick_and_place", "Observe a pick-and-place scene", (checkpoint,))
    strategy = Strategy(
        id="strategy_observe_v1",
        checkpoint_id=checkpoint.id,
        required_skills=("observe",),
        preconditions=(),
        safety_contract=settings.safety_contract,
        actions=(Action(ActionKind.OBSERVE),),
        trusted=True,
    )
    harness = LearningHarness(
        DryRunRobot(),
        SafetyValidator(settings),
        ContextPredicateVerifier(),
        JsonTraceRecorder(settings.artifact_dir),
    )
    trace = harness.run_checkpoint(task, checkpoint, strategy, human_approved=True)
    print(f"{trace.id}: {trace.verification.outcome.value} ({trace.verification.reason})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="robot-learner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the hardware-free demo")
    demo.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    args = parser.parse_args()
    if args.command == "demo":
        return run_demo(args.config)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

