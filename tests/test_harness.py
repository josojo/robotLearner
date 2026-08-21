from pathlib import Path

from robot_learner.config import Settings
from robot_learner.executor import DryRunRobot
from robot_learner.harness import LearningHarness
from robot_learner.models import Action, ActionKind, Checkpoint, Outcome, Predicate, Strategy, Task
from robot_learner.safety import SafetyValidator
from robot_learner.tracing import JsonTraceRecorder
from robot_learner.verification import ContextPredicateVerifier


class StubLanguageModel:
    def __init__(self) -> None:
        self.request: tuple[str, str | None] | None = None

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        images: object = None,
    ) -> str:
        self.request = (prompt, system_prompt)
        return "inspect the scene"


def test_checkpoint_runs_and_records_trace(tmp_path: Path) -> None:
    contract = _settings(tmp_path).safety_contract
    checkpoint = Checkpoint("seen", "Scene seen", Predicate("visible"))
    strategy = Strategy(
        "observe_v1",
        checkpoint.id,
        ("observe",),
        (),
        contract,
        (Action(ActionKind.OBSERVE),),
        trusted=True,
    )
    settings = _settings(tmp_path)
    harness = LearningHarness(
        DryRunRobot(),
        SafetyValidator(settings),
        ContextPredicateVerifier(),
        JsonTraceRecorder(settings.artifact_dir),
    )

    trace = harness.run_checkpoint(
        Task("allowed", "see the scene", (checkpoint,)),
        checkpoint,
        strategy,
        human_approved=True,
    )

    assert trace.verification.outcome is Outcome.SUCCESS
    assert (tmp_path / "traces" / f"{trace.id}.json").exists()


def test_harness_consults_configured_language_model(tmp_path: Path) -> None:
    model = StubLanguageModel()
    settings = _settings(tmp_path)
    harness = LearningHarness(
        DryRunRobot(),
        SafetyValidator(settings),
        ContextPredicateVerifier(),
        JsonTraceRecorder(settings.artifact_dir),
        model,
    )

    response = harness.consult_model("What should happen next?", system_prompt="Be cautious")

    assert response == "inspect the scene"
    assert model.request == ("What should happen next?", "Be cautious")


def _settings(path: Path) -> Settings:
    from robot_learner.models import SafetyContract

    contract = SafetyContract(0.1, 10.0, 30.0, (-0.5, -0.5, 0.0), (0.5, 0.5, 0.75))
    return Settings(path, path / "test.db", True, frozenset({"allowed"}), False, contract)

