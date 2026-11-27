"""Convert the maize sequence splits into the flattened schema for tree baselines.

The original Random Forest / XGBoost / LightGBM trainers expect numeric columns
named temporal_*, past_yield and optional soil_*. This utility converts the
same split files produced by build_maize_dataset.py without changing labels or
split membership.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOIL_FEATURES = [
    "soil_organic_carbon", "soil_ph", "bulk_density", "sand_content",
    "silt_content", "clay_content", "cation_exchange_capacity",
    "total_nitrogen",
]
SOIL_DEPTHS = ["0_5cm", "5_15cm", "15_30cm", "30_60cm", "60_100cm", "100_200cm"]


def _parse_nested(value, field: str) -> np.ndarray:
    """Parse a list stored in a CSV cell and fail clearly if it is malformed."""
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        array = np.asarray(parsed, dtype=np.float32)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"Could not parse {field} cell: {exc}") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array


def flatten_split(
    input_path: Path,
    output_path: Path,
    feature_names: list[str],
    include_soil: bool | None = None,
) -> pd.DataFrame:
    """Flatten temporal and optional soil matrices into tree-model columns."""
    frame = pd.read_csv(input_path)
    required = {"DISTRICT_ID", "sowing_year", "feature_sequence", "past_yield", "Yield"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{input_path} missing required fields: {sorted(missing)}")
    if frame.duplicated(["DISTRICT_ID", "sowing_year"]).any():
        raise ValueError(f"{input_path} has duplicate district/sowing_year rows")

    soil_available = "soil_features" in frame.columns
    if include_soil is True and not soil_available:
        raise ValueError("include_soil=True but the split has no soil_features column")
    if include_soil is False:
        soil_available = False

    rows = []
    expected_features = len(feature_names)
    for _, row in frame.iterrows():
        sequence = _parse_nested(row["feature_sequence"], "feature_sequence")
        if sequence.ndim != 2 or sequence.shape[1] != expected_features:
            raise ValueError(
                f"{row['DISTRICT_ID']} {row['sowing_year']}: expected "
                f"(months, {expected_features}) sequence, got {sequence.shape}"
            )
        record = {
            "DISTRICT_ID": row["DISTRICT_ID"],
            "district": row["DISTRICT_ID"],
            "sowing_year": int(row["sowing_year"]),
            "year": int(row["sowing_year"]),
            "past_yield": float(row["past_yield"]),
            "yield_value": float(row["Yield"]),
        }
        for month_index in range(sequence.shape[0]):
            for feature_index, feature_name in enumerate(feature_names):
                column = f"temporal_m{month_index + 1:02d}_{feature_name}"
                record[column] = float(sequence[month_index, feature_index])

        if soil_available:
            soil = _parse_nested(row["soil_features"], "soil_features")
            if soil.shape != (len(SOIL_FEATURES), len(SOIL_DEPTHS)):
                raise ValueError(
                    f"{row['DISTRICT_ID']}: expected soil matrix "
                    f"{(len(SOIL_FEATURES), len(SOIL_DEPTHS))}, got {soil.shape}"
                )
            for property_index, property_name in enumerate(SOIL_FEATURES):
                for depth_index, depth_name in enumerate(SOIL_DEPTHS):
                    column = f"{property_name}_{depth_name}"
                    record[column] = float(
                        soil[property_index, depth_index]
                    )
        rows.append(record)

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No rows in split file: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def build_flat_datasets(sequence_dir: Path, output_dir: Path) -> dict:
    """Flatten train/validation/test splits using feature names in metadata."""
    metadata_path = sequence_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata.get("feature_columns")
    if not feature_names:
        raise ValueError("metadata.json does not contain feature_columns")

    outputs = {}
    for split in ["train", "val", "test"]:
        outputs[split] = flatten_split(
            sequence_dir / f"{split}.csv",
            output_dir / f"{split}_rf_flat.csv",
            feature_names,
        )
    return {
        "output_dir": str(output_dir),
        "feature_columns": feature_names,
        "split_rows": {name: len(frame) for name, frame in outputs.items()},
        "soil_included": "soil_features" in pd.read_csv(sequence_dir / "train.csv").columns,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("../../data/datasets/maize_kharif"))
    parser.add_argument("--output", type=Path, default=Path("../../data/datasets/rf_flat_maize"))
    args = parser.parse_args()
    summary = build_flat_datasets(args.input, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
