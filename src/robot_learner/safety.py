"""Static safety validation before an action reaches an adapter."""

from dataclasses import dataclass

from robot_learner.config import Settings
from robot_learner.models import ActionKind, Strategy


class SafetyViolation(ValueError):
    """Raised when a strategy is outside its authorized safety boundary."""


@dataclass(frozen=True, slots=True)
class ValidatedStrategy:
    strategy: Strategy


class SafetyValidator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(
        self, task_id: str, strategy: Strategy, *, human_approved: bool = False
    ) -> ValidatedStrategy:
        if task_id not in self._settings.authorized_task_ids:
            raise SafetyViolation(f"task {task_id!r} is not authorized")
        if not strategy.trusted and not self._settings.allow_untrusted_strategies:
            raise SafetyViolation("untrusted strategy is not permitted")
        if strategy.safety_contract.requires_human_approval and not human_approved:
            raise SafetyViolation("strategy requires explicit human approval")
        configured = self._settings.safety_contract
        declared = strategy.safety_contract
        if declared.max_velocity_m_s > configured.max_velocity_m_s:
            raise SafetyViolation("strategy velocity exceeds configured maximum")
        if declared.max_force_n > configured.max_force_n:
            raise SafetyViolation("strategy force exceeds configured maximum")
        if declared.max_duration_s > configured.max_duration_s:
            raise SafetyViolation("strategy duration exceeds configured maximum")
        self._validate_actions(strategy)
        return ValidatedStrategy(strategy)

    @staticmethod
    def _validate_actions(strategy: Strategy) -> None:
        for action in strategy.actions:
            if action.kind is ActionKind.MOVE_CARTESIAN:
                raw_speed = action.parameters.get("speed_m_s")
                if not isinstance(raw_speed, int | float):
                    raise SafetyViolation("move speed must be numeric")
                speed = float(raw_speed)
                if speed <= 0 or speed > strategy.safety_contract.max_velocity_m_s:
                    raise SafetyViolation("move speed is absent, non-positive, or too high")
                target = action.parameters.get("target_m")
                if not isinstance(target, list) or len(target) != 3:
                    raise SafetyViolation("move target_m must contain three coordinates")
                for value, lower, upper in zip(
                    target,
                    strategy.safety_contract.workspace_min_m,
                    strategy.safety_contract.workspace_max_m,
                    strict=True,
                ):
                    if not isinstance(value, int | float):
                        raise SafetyViolation("move coordinates must be numeric")
                    coordinate = float(value)
                    if not lower <= coordinate <= upper:
                        raise SafetyViolation("move target is outside the declared workspace")

