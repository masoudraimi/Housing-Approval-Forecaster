"""Tests for serving/logger.py: SQLite write/read and schema validation."""

import tempfile
from pathlib import Path

import pytest

from serving.logger import PredictionLogger


@pytest.fixture
def tmp_logger(tmp_path):
    return PredictionLogger(db_path=tmp_path / "test_predictions.db")


def test_log_and_read(tmp_logger):
    row_id = tmp_logger.log(
        lga_code="LGA12345",
        horizon_quarters=4,
        predicted_values=[100.0, 110.0, 105.0, 115.0],
        model_version="1",
    )
    assert row_id == 1
    records = tmp_logger.get_recent(n=10)
    assert len(records) == 1
    assert records[0]["lga_code"] == "LGA12345"


def test_total_count(tmp_logger):
    assert tmp_logger.total_count() == 0
    tmp_logger.log("LGA001", 4, [100.0], "1")
    tmp_logger.log("LGA002", 4, [200.0], "1")
    assert tmp_logger.total_count() == 2


def test_get_recent_by_lga(tmp_logger):
    tmp_logger.log("LGA001", 4, [100.0], "1")
    tmp_logger.log("LGA002", 4, [200.0], "1")
    tmp_logger.log("LGA001", 4, [105.0], "1")
    records = tmp_logger.get_recent_by_lga("LGA001")
    assert len(records) == 2
    assert all(r["lga_code"] == "LGA001" for r in records)


def test_log_with_actuals(tmp_logger):
    tmp_logger.log("LGA001", 4, [100.0, 110.0], "1", actual_values=[95.0, 108.0])
    records = tmp_logger.get_recent(1)
    assert records[0]["actual_values"] is not None


def test_schema_columns(tmp_logger):
    tmp_logger.log("LGA001", 4, [100.0], "1")
    records = tmp_logger.get_recent(1)
    expected_keys = {"lga_code", "horizon_quarters", "predicted_values", "model_version", "timestamp"}
    assert expected_keys.issubset(set(records[0].keys()))
