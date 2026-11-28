"""Build leakage-safe rolling-window datasets for one-month-ahead price forecasts.

Each sample predicts a market × commodity × variety's modal price in month t
using the previous N complete consecutive months only. No yield data, soil
matrix, current-month arrival, or future-month prices are inputs.

Run from repository root:
    python src/prices/build_price_dataset.py \
      --panel data/processed/maize_market_monthly.csv \
      --output data/datasets/maize_price \
      --window 12 --train-end 2019 --val-end 2021
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "log_modal_price", "log_min_price", "log_max_price",
    "log_arrival_qty", "arrival_missing",
    "history_month_sin", "history_month_cos",
    "forecast_month_sin", "forecast_month_cos",
]


def _to_float(row: pd.Series, key: str, fallback: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else fallback


def _month_features(history: pd.DataFrame, forecast_month: int) -> np.ndarray:
    """Transform prior months to numeric, scale-stable model features."""
    rows = []
    target_angle = 2 * np.pi * (forecast_month - 1) / 12
    for _, row in history.iterrows():
        modal = max(_to_float(row, "modal_price"), 0.0)
        minimum = _to_float(row, "min_price", modal)
        maximum = _to_float(row, "max_price", modal)
        arrival_value = pd.to_numeric(
            pd.Series([row.get("arrival_qty")]), errors="coerce"
        ).iloc[0]
        arrival_missing = float(pd.isna(arrival_value))
        arrival = 0.0 if arrival_missing else max(float(arrival_value), 0.0)
        month = int(pd.Timestamp(row["period"]).month)
        angle = 2 * np.pi * (month - 1) / 12
        rows.append([
            np.log1p(modal), np.log1p(max(minimum, 0.0)),
            np.log1p(max(maximum, 0.0)), np.log1p(arrival),
            arrival_missing, np.sin(angle), np.cos(angle),
            np.sin(target_angle), np.cos(target_angle),
        ])
    return np.asarray(rows, dtype=np.float32)


def build_price_dataset(
    panel_path: Path,
    output_dir: Path,
    *,
    window: int = 12,
    train_end: int = 2019,
    val_end: int = 2021,
    minimum_reporting_days: int = 1,
) -> dict:
    """Create consecutive monthly windows and chronological target-month splits."""
    panel = pd.read_csv(panel_path)
    required = {
        "series_id", "period", "modal_price", "state", "district",
        "market", "variety",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Monthly price panel missing columns: {sorted(missing)}")
    if window < 2:
        raise ValueError("window must be at least 2 months")
    if train_end >= val_end:
        raise ValueError("train-end must be earlier than val-end")

    panel["period"] = pd.to_datetime(panel["period"], errors="coerce")
    panel["modal_price"] = pd.to_numeric(panel["modal_price"], errors="coerce")
    if "reporting_days" not in panel.columns:
        panel["reporting_days"] = 1
    panel["reporting_days"] = pd.to_numeric(panel["reporting_days"], errors="coerce").fillna(0)
    panel = panel.dropna(subset=["period", "modal_price", "series_id"])
    panel = panel[panel["modal_price"].gt(0)].copy()
    panel = panel[panel["reporting_days"].ge(minimum_reporting_days)].copy()
    if panel.duplicated(["series_id", "period"]).any():
        raise ValueError("Panel has duplicate series_id × period records")
    panel = panel.sort_values(["series_id", "period"]).reset_index(drop=True)

    samples: list[dict] = []
    # Indexing by Period makes a missing calendar month explicit. We do not
    # shift across gaps and then mislabel an 8-week forecast as 1 month ahead.
    for series_id, group in panel.groupby("series_id", sort=True):
        by_period = {
            pd.Timestamp(row["period"]).to_period("M"): row
            for _, row in group.iterrows()
        }
        for target_period, target_row in sorted(by_period.items()):
            history_periods = [
                target_period - offset
                for offset in range(window, 0, -1)
            ]
            if not all(period in by_period for period in history_periods):
                continue
            history = pd.DataFrame([by_period[p] for p in history_periods])
            target_price = float(target_row["modal_price"])
            if not np.isfinite(target_price) or target_price <= 0:
                continue
            x = _month_features(history, int(target_period.month))
            last_price = float(history.iloc[-1]["modal_price"])
            seasonal_price = float(history.iloc[0]["modal_price"]) if window >= 12 else np.nan
            samples.append({
                "X": x,
                "y_log": np.float32(np.log1p(target_price)),
                "y_price": np.float32(target_price),
                "last_price": np.float32(last_price),
                "seasonal_naive_price": np.float32(seasonal_price),
                "period": str(target_period),
                "year": int(target_period.year),
                "series_id": str(series_id),
                "state": str(target_row["state"]),
                "district": str(target_row["district"]),
                "market": str(target_row["market"]),
                "variety": str(target_row["variety"]),
            })

    if not samples:
        raise ValueError(
            f"No samples with {window} consecutive history months. "
            "Check market-level coverage, monthly continuity and the selected dates."
        )
    series_vocab = {
        value: index + 1
        for index, value in enumerate(sorted({s["series_id"] for s in samples}))
    }
    for sample in samples:
        sample["series_index"] = series_vocab[sample["series_id"]]

    split_masks = {
        "train": lambda s: s["year"] <= train_end,
        "val": lambda s: train_end < s["year"] <= val_end,
        "test": lambda s: s["year"] > val_end,
    }
    split_samples = {
        name: [sample for sample in samples if mask(sample)]
        for name, mask in split_masks.items()
    }
    empty = [name for name, rows in split_samples.items() if not rows]
    if empty:
        years = sorted({sample["year"] for sample in samples})
        raise ValueError(
            f"Empty chronological split(s): {empty}. Sample target years={years}; "
            "adjust train-end/val-end after inspecting source coverage."
        )

    # Standardise only the historical features used in training samples.
    scaler = StandardScaler().fit(
        np.concatenate([sample["X"] for sample in split_samples["train"]], axis=0)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

    split_years, split_rows, split_series = {}, {}, {}
    for name, rows in split_samples.items():
        x = np.stack([scaler.transform(sample["X"]) for sample in rows]).astype(np.float32)
        payload = {
            "X": x,
            "y_log": np.asarray([s["y_log"] for s in rows], dtype=np.float32),
            "y_price": np.asarray([s["y_price"] for s in rows], dtype=np.float32),
            "last_price": np.asarray([s["last_price"] for s in rows], dtype=np.float32),
            "seasonal_naive_price": np.asarray(
                [s["seasonal_naive_price"] for s in rows], dtype=np.float32
            ),
            "series_index": np.asarray([s["series_index"] for s in rows], dtype=np.int64),
            "period": np.asarray([s["period"] for s in rows], dtype="U10"),
            "series_id": np.asarray([s["series_id"] for s in rows], dtype="U512"),
            "state": np.asarray([s["state"] for s in rows], dtype="U80"),
            "district": np.asarray([s["district"] for s in rows], dtype="U120"),
            "market": np.asarray([s["market"] for s in rows], dtype="U160"),
            "variety": np.asarray([s["variety"] for s in rows], dtype="U120"),
        }
        np.savez_compressed(output_dir / f"{name}.npz", **payload)
        split_years[name] = sorted({int(s["year"]) for s in rows})
        split_rows[name] = len(rows)
        split_series[name] = len({s["series_id"] for s in rows})

    metadata = {
        "project": "DeepMaize",
        "task": "monthly_maize_mandi_modal_price_forecasting",
        "prediction_unit": "market × variety × target month",
        "target": "monthly median of reported daily modal_price",
        "target_unit": "provider-reported price unit; confirm source metadata",
        "forecast_horizon_months": 1,
        "lookback_months": window,
        "feature_columns": FEATURE_COLUMNS,
        "series_vocabulary": series_vocab,
        "n_series": len(series_vocab),
        "split_years": split_years,
        "split_rows": split_rows,
        "unique_series_per_split": split_series,
        "data_quality": {
            "monthly_panel_rows": int(len(panel)),
            "series_in_monthly_panel": int(panel["series_id"].nunique()),
            "windowed_samples": len(samples),
            "complete_consecutive_windows": len(samples),
        },
        "split_rule": {
            "train_target_year_max": train_end,
            "validation_target_year_min": train_end + 1,
            "validation_target_year_max": val_end,
            "test_target_year_min": val_end + 1,
        },
        "leakage_controls": [
            "Target month price is never included in its own input window.",
            "Arrival quantity is included only for historical months.",
            "All input scalers are fitted on training samples only.",
            "Splits are chronological by target month.",
            "Missing months are not forward-filled or bridged.",
        ],
        "limitations": [
            "This is an observational wholesale market quotation, not farm-gate price.",
            "Forecast is one calendar month ahead from the last complete month.",
            "New markets absent from training map to unknown embedding index 0.",
            "Availability and reporting frequency differ between mandis.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/datasets/maize_price"))
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--train-end", type=int, default=2019)
    parser.add_argument("--val-end", type=int, default=2021)
    parser.add_argument("--minimum-reporting-days", type=int, default=1)
    args = parser.parse_args()
    metadata = build_price_dataset(
        args.panel, args.output, window=args.window,
        train_end=args.train_end, val_end=args.val_end,
        minimum_reporting_days=args.minimum_reporting_days,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
