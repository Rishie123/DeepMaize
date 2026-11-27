"""Evaluate the persistence forecast for the held-out DeepMaize test split.

The persistence forecast is y_hat(district, year) = yield(district, year - 1).
The input CSV stores past_yield after training-set standardisation, so this
script reverses that transformation before comparing against the raw Yield
target. It reports the baseline only; it does not fit or select a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST = REPO_ROOT / "data/datasets/maize_kharif/test.csv"
DEFAULT_SCALER = REPO_ROOT / "data/datasets/maize_kharif/past_yield_scaler.joblib"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/maize_persistence_test_predictions.csv"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return MAE, RMSE and R² with guards for empty/constant targets."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays of equal shape")
    residual = actual - predicted
    variance = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / variance if variance > 0 else float("nan")
    return {
        "n": int(actual.size),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "R2": r2,
    }


def evaluate_persistence(test_csv: Path, scaler_path: Path) -> tuple[pd.DataFrame, dict]:
    """Restore lagged yields to source units and score the test persistence baseline."""
    frame = pd.read_csv(test_csv)
    required = {"DISTRICT_ID", "sowing_year", "past_yield", "Yield"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test CSV is missing required fields: {sorted(missing)}")
    scaler = joblib.load(scaler_path)
    frame = frame.copy()
    frame["persistence_prediction"] = scaler.inverse_transform(
        frame[["past_yield"]].to_numpy(dtype=float)
    ).reshape(-1)
    frame["persistence_residual"] = (
        frame["Yield"].astype(float) - frame["persistence_prediction"]
    )

    metrics = {"all_test_years": regression_metrics(
        frame["Yield"].to_numpy(), frame["persistence_prediction"].to_numpy()
    )}
    for year, group in frame.groupby("sowing_year", sort=True):
        metrics[str(int(year))] = regression_metrics(
            group["Yield"].to_numpy(), group["persistence_prediction"].to_numpy()
        )
    return frame, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions, metrics = evaluate_persistence(args.test, args.scaler)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Prediction-level baseline results saved to {args.output}")


if __name__ == "__main__":
    main()
