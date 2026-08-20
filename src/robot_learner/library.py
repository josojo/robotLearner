"""Minimal append-only SQLite execution history."""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from robot_learner.models import ExecutionTrace, Strategy


class SQLiteStrategyLibrary:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                context_json TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )"""
        )

    def record(self, strategy: Strategy, trace: ExecutionTrace) -> None:
        context = trace.start_observation.context
        payload = json.dumps(asdict(trace), default=str, sort_keys=True)
        with self._connection:
            self._connection.execute(
                "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    trace.id,
                    strategy.id,
                    strategy.checkpoint_id,
                    trace.verification.outcome.value,
                    json.dumps(context, sort_keys=True),
                    payload,
                ),
            )

    def outcome_counts(self, strategy_id: str) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT outcome, COUNT(*) FROM executions WHERE strategy_id = ? GROUP BY outcome",
            (strategy_id,),
        )
        return {str(outcome): int(count) for outcome, count in rows}

