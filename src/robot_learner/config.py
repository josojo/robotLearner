"""TOML configuration loading."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from robot_learner.models import SafetyContract


@dataclass(frozen=True, slots=True)
class Settings:
    artifact_dir: Path
    database_path: Path
    dry_run: bool
    authorized_task_ids: frozenset[str]
    allow_untrusted_strategies: bool
    safety_contract: SafetyContract


def _vector(value: object, key: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must contain exactly three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def load_settings(path: Path) -> Settings:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    runtime = data["runtime"]
    safety = data["safety"]
    return Settings(
        artifact_dir=Path(runtime["artifact_dir"]),
        database_path=Path(runtime["database_path"]),
        dry_run=bool(runtime["dry_run"]),
        authorized_task_ids=frozenset(str(item) for item in safety["authorized_task_ids"]),
        allow_untrusted_strategies=bool(safety["allow_untrusted_strategies"]),
        safety_contract=SafetyContract(
            max_velocity_m_s=float(safety["max_velocity_m_s"]),
            max_force_n=float(safety["max_force_n"]),
            max_duration_s=float(safety["max_duration_s"]),
            workspace_min_m=_vector(safety["workspace_min_m"], "workspace_min_m"),
            workspace_max_m=_vector(safety["workspace_max_m"], "workspace_max_m"),
            requires_human_approval=True,
        ),
    )

