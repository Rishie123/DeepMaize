"""Unit tests for DeepMaize season sequence construction."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.build_maize_dataset import _build_sequences, parse_months


def test_default_months_are_kharif_example():
    assert parse_months("6,7,8,9,10,11,12") == [6, 7, 8, 9, 10, 11, 12]


def test_sequences_assign_months_to_calendar_sowing_year():
    rows = []
    for month in range(6, 12):
        rows.append({"DISTRICT_ID": "MH_A", "YEAR": 2020, "MONTH": month, "rain": month})
    frame, features = _build_sequences(pd.DataFrame(rows), list(range(6, 12)))
    assert features == ["rain"]
    assert len(frame) == 1
    assert frame.iloc[0]["sowing_year"] == 2020
    assert np.asarray(frame.iloc[0]["feature_sequence"]).shape == (6, 1)


def test_incomplete_sequence_is_excluded():
    rows = [
        {"DISTRICT_ID": "MH_A", "YEAR": 2020, "MONTH": month, "rain": month}
        for month in [6, 7, 8, 9, 10]
    ]
    frame, _ = _build_sequences(pd.DataFrame(rows), list(range(6, 12)))
    assert frame.empty


def test_duplicate_district_month_fails():
    row = {"DISTRICT_ID": "MH_A", "YEAR": 2020, "MONTH": 6, "rain": 20}
    with pytest.raises(ValueError, match="duplicate"):
        _build_sequences(pd.DataFrame([row, row]), [6])


def test_month_validation():
    with pytest.raises(ValueError):
        parse_months("6,7,13")
