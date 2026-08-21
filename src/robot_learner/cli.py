"""Dry-run demonstration entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from robot_learner.config import load_settings
from robot_learner.executor import DryRunRobot
from robot_learner.explore import ExplorationError, SimulationExplorer
from robot_learner.harness import LearningHarness
from robot_learner.models import Action, ActionKind, Checkpoint, Predicate, Strategy, Task
from robot_learner.openrouter import OpenRouterLanguageModel
from robot_learner.planning import CheckpointPlanner, PlanError, task_from_dict, task_to_dict
from robot_learner.ports import LanguageModel
from robot_learner.review import ReviewError, write_review
from robot_learner.safety import SafetyValidator
from robot_learner.simulation import (
    GeneratedScript,
    ScriptValidationError,
    SimulationClient,
    SimulationError,
    SimulationSpec,
    script_from_actions,
)
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
    destination.write_text(json.dumps(task_to_dict(task), indent=2, sort_keys=True) + "\n")
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


def collect_sim_steps(argv: list[str]) -> list[tuple[str, str]]:
    """Preserve mixed --do/--skill/--section/--checkpoint order from argv."""
    steps: list[tuple[str, str]] = []
    flags = {"--do", "--skill", "--section", "--checkpoint"}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in flags and index + 1 < len(argv):
            steps.append((token[2:], argv[index + 1]))
            index += 2
            continue
        index += 1
    return steps


def _parse_skill(raw: str) -> tuple[str, dict[str, Any]]:
    name, separator, rest = raw.partition(":")
    if not name:
        raise ValueError("skill name is empty")
    if not separator:
        return name, {}
    loaded = json.loads(rest)
    if not isinstance(loaded, dict):
        raise ValueError("skill parameters must be a JSON object")
    return name, loaded


def run_simulation(config_path: Path, argv: list[str], output: Path | None) -> int:
    from robot_learner.models import new_id

    spec = SimulationSpec.from_toml(config_path)
    steps = collect_sim_steps(argv)
    if not steps:
        raise ValueError("simulate requires at least one --do, --skill, --section, or --checkpoint")
    run_dir = output or Path("artifacts") / "simulation-runs" / new_id("sim")
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    explored: list[Action] = []
    failed = False
    simulation: SimulationClient | None = None
    try:
        simulation = SimulationClient(spec, run_dir)
        (run_dir / "manifest.json").write_text(
            json.dumps(simulation.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for kind, value in steps:
            if kind == "do" and value == "observe":
                observation = simulation.observe()
                explored.append(Action(ActionKind.OBSERVE))
                records.append(_observation_record(observation))
                print(f"observe: {observation.context.get('phase')}")
            elif kind == "do" and value.startswith("observe:"):
                cameras = tuple(item for item in value.split(":", 1)[1].split(",") if item)
                observation = simulation.observe(cameras)
                explored.append(Action(ActionKind.OBSERVE, {"cameras": list(cameras)}))
                records.append(_observation_record(observation))
                print(f"observe: {observation.context.get('phase')}")
            elif kind == "do" and value == "snapshot":
                snapshot_id = simulation.snapshot()
                records.append({"operation": "snapshot", "ok": True, "snapshot_id": snapshot_id})
                print(f"snapshot: {snapshot_id}")
            elif kind == "do" and (value == "restore" or value.startswith("restore=")):
                raw_id = value.partition("=")[2] or value.partition(":")[2]
                snapshot_id = int(raw_id) if raw_id else 1
                state = simulation.restore(snapshot_id)
                records.append(
                    {"operation": "restore", "ok": True, "snapshot_id": snapshot_id, "state": state}
                )
                print(f"restore: {snapshot_id} phase={state.get('phase')}")
            elif kind in {"skill", "do"} and (
                kind == "skill" or value.startswith("skill=") or value.startswith("skill:")
            ):
                raw_skill = value.split("=", 1)[-1] if kind == "do" else value
                if kind == "do" and value.startswith("skill:"):
                    raw_skill = value.split(":", 1)[1]
                name, parameters = _parse_skill(raw_skill)
                action = Action(
                    ActionKind.RUN_SKILL, {"name": name, "parameters": parameters}
                )
                result = simulation.execute(action)
                explored.append(action)
                records.append({"operation": "run_skill", **result})
                ok = bool(result.get("ok"))
                print(f"{name}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    failed = True
                    break
            elif kind == "section":
                checkpoint_id, separator, raw_path = value.partition("=")
                if not separator or not checkpoint_id or not raw_path:
                    raise ValueError("--section must use CHECKPOINT_ID=SCRIPT.py")
                source = Path(raw_path).read_text(encoding="utf-8")
                result = simulation.run_script(
                    GeneratedScript(checkpoint_id=checkpoint_id, source=source)
                )
                records.append(result)
                if _script_failed(result):
                    print(f"{checkpoint_id}: FAIL")
                    failed = True
                    break
                print(f"{checkpoint_id}: PASS")
            elif kind == "checkpoint":
                checkpoint_id, separator, skill = value.partition("=")
                if not separator or not checkpoint_id or not skill:
                    raise ValueError("--checkpoint must use CHECKPOINT_ID=SKILL")
                name, parameters = _parse_skill(skill)
                before = simulation.observe()
                records.append(_observation_record(before))
                action = Action(
                    ActionKind.RUN_SKILL, {"name": name, "parameters": parameters}
                )
                result = simulation.execute(action)
                after = simulation.observe()
                records.append({"operation": "run_skill", **result})
                records.append(_observation_record(after))
                actions = [
                    Action(ActionKind.OBSERVE),
                    action,
                    Action(ActionKind.OBSERVE),
                ]
                source = script_from_actions(actions, spec.cameras)
                simulation.persist_script(
                    GeneratedScript(checkpoint_id=checkpoint_id, source=source)
                )
                explored.extend(actions)
                ok = bool(result.get("ok"))
                print(f"{checkpoint_id}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    failed = True
                    break
            else:
                raise ValueError(f"unrecognized simulate step {kind}={value}")
        if explored:
            compiled = script_from_actions(explored, spec.cameras)
            (run_dir / "explored.py").write_text(compiled, encoding="utf-8")
    except Exception:
        failed = True
        raise
    finally:
        if simulation is not None:
            simulation.close()
        (run_dir / "run.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _print_review(run_dir)
        print(f"Run: {run_dir}")
    return 1 if failed else 0


def _observation_record(observation: Any) -> dict[str, Any]:
    return {
        "operation": "observe",
        "ok": True,
        "observation_id": observation.id,
        "artifact_refs": list(observation.artifact_refs),
        "phase": observation.context.get("phase"),
        "unstable": observation.context.get("unstable"),
    }


def _script_failed(result: dict[str, Any]) -> bool:
    if result.get("state", {}).get("unstable"):
        return True
    return any(not row.get("ok", True) for row in result.get("results", []))


def load_plan(path: Path) -> Task:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"plan is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise PlanError("plan must be a JSON object")
    return task_from_dict(raw)


def run_explore(
    simulation_config: Path,
    plan_path: Path,
    output: Path | None,
    *,
    language_model: LanguageModel | None = None,
    worker: Callable[..., None] | None = None,
) -> int:
    from robot_learner.models import new_id

    load_dotenv(Path.cwd() / ".env")
    spec = SimulationSpec.from_toml(simulation_config)
    task = load_plan(plan_path)
    run_dir = output or Path("artifacts") / "simulation-runs" / new_id("explore")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = language_model or OpenRouterLanguageModel.from_env()
    report = None
    simulation: SimulationClient | None = None
    try:
        simulation = SimulationClient(spec, run_dir, worker=worker)
        (run_dir / "manifest.json").write_text(
            json.dumps(simulation.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = SimulationExplorer(simulation, model).explore(task)
        for item in report.checkpoints:
            status = "PASS" if item.ok else "FAIL"
            detail = f" ({item.skill})" if item.skill else ""
            print(f"{item.checkpoint_id}: {status}{detail}")
        return 0 if report.ok else 1
    finally:
        if simulation is not None:
            simulation.close()
        if report is not None:
            (run_dir / "explore.json").write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _print_review(run_dir)
        print(f"Run: {run_dir}")


def _print_review(run_dir: Path, *, encode: bool = False) -> None:
    try:
        artifacts = write_review(run_dir, encode=encode)
    except ReviewError:
        return
    _report_review(artifacts)


def _report_review(artifacts: Any) -> None:
    print(f"Watch in a browser: {artifacts.html}")
    for video in artifacts.videos:
        print(f"Video: {video}")
    for note in artifacts.notes:
        print(f"Note: {note}")


def run_review(run_dir: Path, *, encode: bool = True, open_browser: bool = False) -> int:
    artifacts = write_review(run_dir, encode=encode)
    _report_review(artifacts)
    if encode and not artifacts.videos:
        print("Video: install ffmpeg (brew install ffmpeg) or the simulation extra for GIF/WebP")
    if open_browser:
        import webbrowser

        webbrowser.open(artifacts.html.resolve().as_uri())
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
    simulation_info = subparsers.add_parser(
        "simulation-info", help="start a simulation worker and print its capabilities"
    )
    simulation_info.add_argument(
        "--simulation-config", type=Path, default=Path("configs/cableties.toml")
    )
    simulate = subparsers.add_parser(
        "simulate",
        help="explore a simulation host (observe/run_skill) and compile restricted scripts",
    )
    simulate.add_argument(
        "--simulation-config", type=Path, default=Path("configs/cableties.toml")
    )
    simulate.add_argument(
        "--do",
        action="append",
        default=[],
        help="observe | observe:cam,cam | skill=NAME | skill=NAME:{json} | snapshot | restore=ID",
    )
    simulate.add_argument("--skill", action="append", default=[], help="run a named skill")
    simulate.add_argument(
        "--section",
        action="append",
        default=[],
        help="replay a restricted script as CHECKPOINT_ID=SCRIPT.py",
    )
    simulate.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="explore CHECKPOINT_ID=SKILL and persist the compiled script",
    )
    simulate.add_argument("--output", type=Path, help="simulation artifact directory")
    explore = subparsers.add_parser(
        "explore",
        help="walk a checkpoint plan and persist one restricted script per checkpoint",
    )
    explore.add_argument(
        "--simulation-config", type=Path, default=Path("configs/cableties.toml")
    )
    explore.add_argument("--plan", type=Path, required=True, help="task JSON from `start`")
    explore.add_argument("--output", type=Path, help="simulation artifact directory")
    review = subparsers.add_parser(
        "review",
        help="build a watchable replay from captured simulation frames",
    )
    review.add_argument("run_dir", type=Path, help="simulation run directory")
    review.add_argument(
        "--no-video",
        action="store_true",
        help="write review.html only, skip MP4/GIF/WebP encoding",
    )
    review.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        default=True,
        help="open review.html in a browser (default)",
    )
    review.add_argument(
        "--no-open",
        action="store_false",
        dest="open_browser",
        help="do not open a browser",
    )
    args = parser.parse_args()
    if args.command == "demo":
        return run_demo(args.config)
    if args.command == "start":
        try:
            return create_plan(args.prompt, args.config, args.output)
        except (PlanError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    sim_errors = (
        OSError,
        ValueError,
        RuntimeError,
        SimulationError,
        ScriptValidationError,
        ExplorationError,
        PlanError,
        ReviewError,
        EOFError,
    )
    if args.command == "simulation-info":
        try:
            return inspect_simulation(args.simulation_config)
        except sim_errors as exc:
            parser.error(str(exc))
    if args.command == "simulate":
        try:
            return run_simulation(args.simulation_config, sys.argv[1:], args.output)
        except sim_errors as exc:
            parser.error(str(exc))
    if args.command == "explore":
        try:
            return run_explore(args.simulation_config, args.plan, args.output)
        except sim_errors as exc:
            parser.error(str(exc))
    if args.command == "review":
        try:
            return run_review(
                args.run_dir, encode=not args.no_video, open_browser=args.open_browser
            )
        except sim_errors as exc:
            parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
