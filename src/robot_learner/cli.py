"""Dry-run demonstration entry point."""

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from robot_learner.config import load_settings
from robot_learner.executor import DryRunRobot
from robot_learner.harness import LearningHarness
from robot_learner.models import Action, ActionKind, Checkpoint, Predicate, Strategy, Task
from robot_learner.openrouter import OpenRouterLanguageModel
from robot_learner.planning import CheckpointPlanner, PlanError, task_to_dict
from robot_learner.safety import SafetyValidator
from robot_learner.simulation import GeneratedScript, SimulationClient, SimulationSpec
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
    load_dotenv(Path.cwd() / ".env")
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


def inspect_simulation(config_path: Path) -> int:
    spec = SimulationSpec.from_toml(config_path)
    run_dir = Path("artifacts") / "simulation-inspect"
    with SimulationClient(spec, run_dir) as simulation:
        print(json.dumps(simulation.manifest, indent=2, sort_keys=True))
    return 0


def run_simulation(
    config_path: Path,
    sections: list[str],
    checkpoints: list[str],
    output: Path | None,
) -> int:
    from robot_learner.models import new_id

    spec = SimulationSpec.from_toml(config_path)
    run_dir = output or Path("artifacts") / "simulation-runs" / new_id("sim")
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with SimulationClient(spec, run_dir) as simulation:
        (run_dir / "manifest.json").write_text(
            json.dumps(simulation.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        requested: list[tuple[str, str]] = []
        for item in sections:
            checkpoint_id, separator, raw_path = item.partition("=")
            if not separator or not checkpoint_id or not raw_path:
                raise ValueError("--section must use CHECKPOINT_ID=SCRIPT.py")
            requested.append((checkpoint_id, Path(raw_path).read_text(encoding="utf-8")))
        for item in checkpoints:
            checkpoint_id, separator, skill = item.partition("=")
            if not separator or not checkpoint_id or not skill:
                raise ValueError("--checkpoint must use CHECKPOINT_ID=SKILL")
            camera_literal = json.dumps(list(spec.cameras))
            source = (
                f"sim.observe(cameras={camera_literal})\n"
                f"sim.run_skill({json.dumps(skill)})\n"
                f"sim.observe(cameras={camera_literal})\n"
            )
            requested.append((checkpoint_id, source))
        for checkpoint_id, source in requested:
            script = GeneratedScript(
                checkpoint_id=checkpoint_id,
                source=source,
                version=1,
            )
            result = simulation.run_script(script)
            records.append(result)
            skill_fail = any(
                row.get("operation") == "run_skill" and not row.get("ok")
                for row in result.get("results", [])
            )
            print(f"{checkpoint_id}: {'FAIL' if skill_fail else 'PASS'}")
            if skill_fail:
                break
    (run_dir / "run.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Run: {run_dir}")
    return 1 if records and any(
        any(row.get("operation") == "run_skill" and not row.get("ok")
            for row in record.get("results", []))
        for record in records
    ) else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="robot-learner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the hardware-free demo")
    demo.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    start = subparsers.add_parser("start", help="create checkpoints from an initial prompt")
    start.add_argument("prompt", help="natural-language robot task")
    start.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    start.add_argument("--output", type=Path, help="write the generated task JSON here")
    simulation_info = subparsers.add_parser(
        "simulation-info", help="start a simulation worker and print its capabilities"
    )
    simulation_info.add_argument(
        "--simulation-config", type=Path, default=Path("configs/cableties.toml")
    )
    simulate = subparsers.add_parser(
        "simulate", help="execute restricted checkpoint scripts in a simulation worker"
    )
    simulate.add_argument(
        "--simulation-config", type=Path, default=Path("configs/cableties.toml")
    )
    simulate.add_argument(
        "--section",
        action="append",
        default=[],
        help="checkpoint and restricted script as CHECKPOINT_ID=SCRIPT.py; repeat in order",
    )
    simulate.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="generate and execute a camera-grounded script as CHECKPOINT_ID=SKILL",
    )
    simulate.add_argument("--output", type=Path, help="simulation artifact directory")
    args = parser.parse_args()
    if args.command == "demo":
        return run_demo(args.config)
    if args.command == "start":
        try:
            return create_plan(args.prompt, args.config, args.output)
        except (PlanError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    if args.command == "simulation-info":
        return inspect_simulation(args.simulation_config)
    if args.command == "simulate":
        try:
            if not args.section and not args.checkpoint:
                parser.error("simulate requires at least one --section or --checkpoint")
            return run_simulation(
                args.simulation_config, args.section, args.checkpoint, args.output
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
