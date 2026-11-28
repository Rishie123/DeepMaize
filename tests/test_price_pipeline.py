"""Tests for the market-level modal-price forecasting pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.prices.prepare_price_panel import prepare_price_panel
from src.prices.build_price_dataset import build_price_dataset


def test_prepare_panel_aggregates_modal_prices_and_market_arrivals(tmp_path: Path):
    prices_path = tmp_path / "prices.csv"
    quantities_path = tmp_path / "quantities.csv"
    output_path = tmp_path / "monthly.csv"

    pd.DataFrame([
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Nagpur APMC", "market_id": 123,
            "commodity": "Maize", "variety": "Yellow",
            "date": "2020-06-01", "min_price": 1400,
            "max_price": 1800, "modal_price": 1600,
        },
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Nagpur APMC", "market_id": 123,
            "commodity": "Maize", "variety": "Yellow",
            "date": "2020-06-15", "min_price": 1500,
            "max_price": 1900, "modal_price": 1800,
        },
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Kalamna APMC", "market_id": 124,
            "commodity": "Maize", "variety": "Yellow",
            "date": "2020-06-15", "min_price": 1300,
            "max_price": 1700, "modal_price": 1500,
        },
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Nagpur APMC", "market_id": 123,
            "commodity": "Maize", "variety": "Yellow",
            "date": "2020-06-20", "min_price": 1900,
            "max_price": 1500, "modal_price": 1600,
        },
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Nagpur APMC", "market_id": 123,
            "commodity": "Wheat", "variety": "Local",
            "date": "2020-06-20", "min_price": 1800,
            "max_price": 2200, "modal_price": 2000,
        },
    ]).to_csv(prices_path, index=False)

    pd.DataFrame([
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Nagpur APMC", "market_id": 123,
            "commodity": "Maize", "date": "2020-06-01", "arrival_qty": 10,
        },
        {
            "state": "Maharashtra", "district": "Nagpur",
            "market": "Nagpur APMC", "market_id": 123,
            "commodity": "Maize", "date": "2020-06-15", "arrival_qty": 20,
        },
    ]).to_csv(quantities_path, index=False)

    panel, metadata = prepare_price_panel(
        prices_path, output_path, quantities_path
    )
    assert len(panel) == 2
    nagpur = panel.loc[panel["market_id"].eq(123)].iloc[0] if "market_id" in panel else panel.loc[panel["market"].eq("Nagpur APMC")].iloc[0]
    assert np.isclose(nagpur["modal_price"], 1700)
    assert nagpur["arrival_qty"] == 30
    assert metadata["quality"]["rows_removed_for_min_modal_max_order"] == 1
    assert panel["commodity"].str.lower().eq("maize").all()
    assert output_path.exists()
    assert output_path.with_suffix(".metadata.json").exists()


def _make_monthly_panel(path: Path):
    rows = []
    for series_index, (state, district, market, variety, base) in enumerate([
        ("Maharashtra", "Nagpur", "Nagpur APMC", "Yellow", 1500),
        ("Madhya Pradesh", "Indore", "Indore APMC", "Hybrid", 1600),
    ]):
        for month_index, period in enumerate(pd.date_range("2017-01-01", "2024-12-01", freq="MS")):
            seasonal = 100 * np.sin(2 * np.pi * (period.month - 1) / 12)
            trend = month_index * 2
            rows.append({
                "series_id": f"{state}|{district}|id:{100 + series_index}|maize|{variety.lower()}",
                "state": state,
                "district": district,
                "market": market,
                "variety": variety,
                "period": period,
                "modal_price": base + seasonal + trend,
                "min_price": base - 80 + seasonal + trend,
                "max_price": base + 100 + seasonal + trend,
                "arrival_qty": 50 + month_index,
                "reporting_days": 8,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_price_dataset_uses_only_prior_consecutive_months_and_chronological_splits(tmp_path: Path):
    panel_path = tmp_path / "panel.csv"
    _make_monthly_panel(panel_path)
    output_dir = tmp_path / "dataset"

    metadata = build_price_dataset(
        panel_path, output_dir, window=12, train_end=2020, val_end=2021
    )

    assert metadata["task"] == "monthly_maize_mandi_modal_price_forecasting"
    assert metadata["forecast_horizon_months"] == 1
    assert metadata["lookback_months"] == 12
    assert metadata["split_rows"]["train"] > 0
    assert metadata["split_rows"]["val"] > 0
    assert metadata["split_rows"]["test"] > 0
    assert metadata["unique_series_per_split"]["test"] == 2

    with np.load(output_dir / "test.npz", allow_pickle=False) as test:
        assert test["X"].shape[1:] == (12, 9)
        assert np.all(test["y_price"] > 0)
        assert np.all(test["last_price"] > 0)
        # A full 12-month lookback gives the same-calendar-month prior year
        # as the first (oldest) row of each sequence.
        assert np.all(np.isfinite(test["seasonal_naive_price"]))
        assert min(test["period"]) >= "2022-01"

    with np.load(output_dir / "train.npz", allow_pickle=False) as train:
        assert max(train["period"]) <= "2020-12"
    with np.load(output_dir / "val.npz", allow_pickle=False) as val:
        assert min(val["period"]) >= "2021-01"
        assert max(val["period"]) <= "2021-12"


def test_price_dataset_does_not_bridge_a_missing_month(tmp_path: Path):
    panel_path = tmp_path / "panel_gap.csv"
    _make_monthly_panel(panel_path)
    panel = pd.read_csv(panel_path)
    key = panel["series_id"].iloc[0]
    panel = panel[~(
        panel["series_id"].eq(key) & panel["period"].eq("2020-04-01")
    )]
    panel.to_csv(panel_path, index=False)

    output_dir = tmp_path / "dataset_gap"
    metadata = build_price_dataset(
        panel_path, output_dir, window=12, train_end=2020, val_end=2021
    )
    with np.load(output_dir / "test.npz", allow_pickle=False) as test:
        # Market 1 cannot contribute May 2020 (its previous month is missing).
        # It resumes after 12 new consecutive observations have accumulated.
        affected = test["series_id"] == key
        assert not np.any(affected & (test["period"] == "2020-05"))
        assert metadata["data_quality"]["windowed_samples"] > 0


def test_price_dataset_fails_when_market_identity_is_missing(tmp_path: Path):
    path = tmp_path / "prices_no_market.csv"
    pd.DataFrame([{
        "state": "Maharashtra", "district": "Nagpur",
        "date": "2020-06-01", "commodity": "Maize", "modal_price": 1500,
    }]).to_csv(path, index=False)
    try:
        prepare_price_panel(path, tmp_path / "out.csv")
    except ValueError as exc:
        assert "market_id or market" in str(exc)
    else:
        raise AssertionError("Expected market identity to be required")
