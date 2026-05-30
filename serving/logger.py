"""SQLite prediction logger for housing approvals forecasts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite_utils

__all__ = ["PredictionLogger"]

_DEFAULT_DB_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "data/predictions.db"))
_TABLE = "predictions"


class PredictionLogger:
    """Logs every forecast request to a SQLite table for drift monitoring."""

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite_utils.Database(str(db_path))
        self._ensure_table()

    def _ensure_table(self) -> None:
        if _TABLE not in self._db.table_names():
            self._db[_TABLE].create({
                "id": int,
                "timestamp": str,
                "lga_code": str,
                "horizon_quarters": int,
                "predicted_values": str,
                "model_version": str,
                "actual_values": str,
            }, pk="id")

    def log(
        self,
        lga_code: str,
        horizon_quarters: int,
        predicted_values: list[float],
        model_version: str,
        actual_values: Optional[list[float]] = None,
    ) -> int:
        """Insert a prediction record; return the new row id."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lga_code": lga_code,
            "horizon_quarters": horizon_quarters,
            "predicted_values": str(predicted_values),
            "model_version": model_version,
            "actual_values": str(actual_values) if actual_values else None,
        }
        self._db[_TABLE].insert(record)
        return self._db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_recent(self, n: int = 100) -> list[dict]:
        """Return the n most recent prediction records."""
        rows = list(self._db.execute(
            f"SELECT * FROM {_TABLE} ORDER BY timestamp DESC LIMIT ?", [n]
        ).fetchall())
        cols = [col[1] for col in self._db.execute(f"PRAGMA table_info({_TABLE})").fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def get_recent_by_lga(self, lga_code: str, n: int = 100) -> list[dict]:
        """Return recent predictions for a specific LGA."""
        rows = list(self._db.execute(
            f"SELECT * FROM {_TABLE} WHERE lga_code = ? ORDER BY timestamp DESC LIMIT ?",
            [lga_code, n],
        ).fetchall())
        cols = [col[1] for col in self._db.execute(f"PRAGMA table_info({_TABLE})").fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def total_count(self) -> int:
        return self._db.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
