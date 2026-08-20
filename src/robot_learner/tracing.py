"""Append-only JSON trace artifacts."""

import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from robot_learner.models import ExecutionTrace


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class JsonTraceRecorder:
    def __init__(self, artifact_dir: Path) -> None:
        self._trace_dir = artifact_dir / "traces"

    def record(self, trace: ExecutionTrace) -> Path:
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        destination = self._trace_dir / f"{trace.id}.json"
        destination.write_text(
            json.dumps(asdict(trace), default=_json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

