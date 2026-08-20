"""Dry-run demonstration entry point."""

import argparse
import json
from pathlib import Path

from robot_learner.config import load_settings
from robot_learner.executor import DryRunRobot
from robot_learner.harness import LearningHarness
from robot_learner.models import Action, ActionKind, Checkpoint, Predicate, Strategy, Task
from robot_learner.openrouter import OpenRouterLanguageModel
from robot_learner.planning import CheckpointPlanner, PlanError, task_to_dict
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


def create_plan(prompt: str, config_path: Path, output: Path | None = None) -> int:
    settings = load_settings(config_path)
    task = CheckpointPlanner(OpenRouterLanguageModel.from_env()).create_task(prompt)
    destination = output or settings.artifact_dir / "plans" / f"{task.id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(task_to_dict(task), indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(task.checkpoints)} checkpoints for: {task.goal}")
    for number, checkpoint in enumerate(task.checkpoints, 1):
        print(f"  {number:02d} {checkpoint.id}: {checkpoint.name}")
    print(f"Plan: {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="robot-learner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the hardware-free demo")
    demo.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    start = subparsers.add_parser("start", help="create checkpoints from an initial prompt")
    start.add_argument("prompt", help="natural-language robot task")
    start.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    start.add_argument("--output", type=Path, help="write the generated task JSON here")
    args = parser.parse_args()
    if args.command == "demo":
        return run_demo(args.config)
    if args.command == "start":
        try:
            return create_plan(args.prompt, args.config, args.output)
        except (PlanError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

