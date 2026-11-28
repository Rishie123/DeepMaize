"""Train and evaluate a pooled LSTM for one-month-ahead maize mandi prices.

The validation split selects the best checkpoint. The held-out test split is
used once for final reporting, never for model or epoch selection.
Metrics and predictions are reported in the original price unit (as supplied
by Agmarknet/CEDA), not in log space.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.prices.model import PriceLSTM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path, device: torch.device) -> tuple[TensorDataset, dict]:
    """Load one prepared split from NPZ without enabling pickle loading."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    tensors = TensorDataset(
        torch.tensor(arrays["X"], dtype=torch.float32, device=device),
        torch.tensor(arrays["series_index"], dtype=torch.long, device=device),
        torch.tensor(arrays["y_log"], dtype=torch.float32, device=device),
    )
    return tensors, arrays


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate price-unit errors with stable guards for percentage metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.where(denominator > 0, 2 * np.abs(error) / denominator, 0))
    variance = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - np.sum(error ** 2) / variance if variance > 0 else float("nan")
    return {
        "n": int(len(actual)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(r2),
        "sMAPE_percent": float(smape * 100),
    }


@torch.no_grad()
def predict(model: PriceLSTM, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    for sequence, series_index, _ in loader:
        output = model(sequence.to(device), series_index.to(device))
        # The model predicts log(1 + price), with no target scaler.
        predictions.append(torch.expm1(output).clamp_min(0).cpu().numpy())
    return np.concatenate(predictions)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = args.dataset
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    train_data, train_arrays = load_split(data_dir / "train.npz", device)
    val_data, val_arrays = load_split(data_dir / "val.npz", device)
    test_data, test_arrays = load_split(data_dir / "test.npz", device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = PriceLSTM(
        input_dim=int(train_arrays["X"].shape[-1]),
        n_series=int(metadata["n_series"]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "price_lstm.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for sequence, series_index, y_log in train_loader:
            sequence = sequence.to(device)
            series_index = series_index.to(device)
            y_log = y_log.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence, series_index)
            loss = loss_fn(prediction, y_log)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_log_pred = predict(model, train_eval_loader, device)
        val_log_pred = predict(model, val_loader, device)
        # Predictions are unscaled prices; score on the natural price scale.
        train_scores = metrics(train_arrays["y_price"], train_log_pred)
        val_scores = metrics(val_arrays["y_price"], val_log_pred)
        history.append({
            "epoch": epoch,
            "train_loss_log_mse": float(np.mean(batch_losses)),
            "train_rmse": train_scores["RMSE"],
            "validation_rmse": val_scores["RMSE"],
            "validation_mae": val_scores["MAE"],
        })
        print(
            f"Epoch {epoch:03d} | train RMSE {train_scores['RMSE']:.2f} | "
            f"validation RMSE {val_scores['RMSE']:.2f}"
        )
        if val_scores["RMSE"] < best_val_rmse:
            best_val_rmse = val_scores["RMSE"]
            best_epoch = epoch
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": int(train_arrays["X"].shape[-1]),
                    "n_series": int(metadata["n_series"]),
                    "hidden_dim": args.hidden_dim,
                    "embedding_dim": args.embedding_dim,
                    "dropout": args.dropout,
                },
                "epoch": epoch,
                "validation_rmse": best_val_rmse,
            }, checkpoint_path)
        else:
            wait += 1
            if wait >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_prediction = predict(model, test_loader, device)
    actual = test_arrays["y_price"]

    persistence = test_arrays["last_price"]
    seasonal = test_arrays["seasonal_naive_price"]
    valid_seasonal = np.isfinite(seasonal)
    report = {
        "task": metadata["task"],
        "target": metadata["target"],
        "forecast_horizon_months": metadata["forecast_horizon_months"],
        "lookback_months": metadata["lookback_months"],
        "unit": "same as modal_price input; verify provider unit",
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "test": {
            "PriceLSTM": metrics(actual, test_prediction),
            "last_observed_price_persistence": metrics(actual, persistence),
            "same_month_last_year": (
                metrics(actual[valid_seasonal], seasonal[valid_seasonal])
                if valid_seasonal.any() else None
            ),
        },
        "selection_rule": "minimum validation RMSE; test set not used for selection",
    }
    prediction_frame = pd.DataFrame({
        "period": test_arrays["period"],
        "state": test_arrays["state"],
        "district": test_arrays["district"],
        "market": test_arrays["market"],
        "variety": test_arrays["variety"],
        "series_id": test_arrays["series_id"],
        "actual_modal_price": actual,
        "predicted_modal_price": test_prediction,
        "persistence_prediction": persistence,
        "seasonal_naive_prediction": seasonal,
    })
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/maize_price"))
    parser.add_argument("--output", type=Path, default=Path("experiments/maize_price_lstm"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
