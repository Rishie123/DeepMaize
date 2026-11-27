"""Build leakage-safe district-year Kharif maize sequence datasets.

Expected inputs
---------------
Monthly features CSV:
    DISTRICT_ID, YEAR, MONTH, followed by numeric feature columns.
Yield CSV:
    DISTRICT_ID, sowing_year, Yield
Optional soil CSV:
    DISTRICT_ID, soil_features
where soil_features is a Python-literal 2-D list shaped
(n_soil_properties, n_depths), as produced by data/soilgrids/wrangle.py.

Important design decisions
--------------------------
* Crop year is defined by sowing_year. A monthly observation is assigned to
  the calendar year in which it occurs. For the default monsoon calendar,
  June-November belong to the same calendar-year Kharif season.
* The crop calendar is configurable; do not assume every district has the
  same sowing/harvest dates without agronomic validation.
* District-year sequences are retained only when every configured month is
  present. This prevents variable-length sequences caused by silent gaps.
* Scalers are fitted on training years only and re-used unchanged elsewhere.
* The first labelled year for each district is removed because lagged yield
  is unavailable. Missing intermediate years are not bridged.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_months(value: str) -> list[int]:
    """Parse a comma-separated month list and validate calendar month values."""
    months = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not months or len(set(months)) != len(months):
        raise ValueError("months must be a non-empty list without duplicates")
    if any(month < 1 or month > 12 for month in months):
        raise ValueError("months must be integers from 1 to 12")
    return months


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Raise an actionable error when an input does not match the schema."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _build_sequences(
    monthly: pd.DataFrame, months: list[int], year_offset: int = 0
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate monthly rows into fixed-length district-season sequences.

    year_offset is 0 for standard Kharif seasons: June-November of calendar
    year Y are assigned to sowing_year Y. It is exposed for crop-calendar
    edge cases, but must be justified from the district-specific calendar.
    """
    monthly = monthly.copy()
    monthly["YEAR"] = pd.to_numeric(monthly["YEAR"], errors="raise").astype(int)
    monthly["MONTH"] = pd.to_numeric(monthly["MONTH"], errors="raise").astype(int)
    if monthly.duplicated(["DISTRICT_ID", "YEAR", "MONTH"]).any():
        raise ValueError("Monthly input contains duplicate district/year/month rows")

    monthly = monthly[monthly["MONTH"].isin(months)].copy()
    monthly["sowing_year"] = monthly["YEAR"] + year_offset
    feature_cols = [
        col for col in monthly.columns
        if col not in {"DISTRICT_ID", "YEAR", "MONTH", "sowing_year"}
    ]
    if not feature_cols:
        raise ValueError("No feature columns found in monthly feature input")

    records = []
    expected = set(months)
    for (district_id, sowing_year), group in monthly.groupby(
        ["DISTRICT_ID", "sowing_year"], sort=True
    ):
        group = group.sort_values("MONTH")
        observed = set(group["MONTH"].astype(int))
        if observed != expected:
            continue
        values = group[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            continue
        records.append({
            "DISTRICT_ID": district_id,
            "sowing_year": int(sowing_year),
            "feature_sequence": values.tolist(),
        })
    return pd.DataFrame(records), feature_cols


def _parse_soil(value: object) -> np.ndarray:
    """Parse and validate one soil matrix from a CSV string or list."""
    matrix = np.asarray(ast.literal_eval(value) if isinstance(value, str) else value,
                        dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Each soil_features value must be a finite 2-D matrix")
    return matrix


def _fit_transform_sequences(
    frames: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Fit temporal and lagged-yield scalers on train only, apply to all splits."""
    train = frames["train"]
    if train.empty:
        raise ValueError("Training split is empty; cannot fit scalers")
    train_sequences = np.stack(train["feature_sequence"].to_numpy()).astype(np.float32)
    sequence_scaler = StandardScaler().fit(
        train_sequences.reshape(-1, train_sequences.shape[-1])
    )
    for split, frame in frames.items():
        if frame.empty:
            continue
        array = np.stack(frame["feature_sequence"].to_numpy()).astype(np.float32)
        n, steps, features = array.shape
        transformed = sequence_scaler.transform(array.reshape(-1, features))
        frame.loc[:, "feature_sequence"] = list(
            transformed.reshape(n, steps, features).tolist()
        )
    joblib.dump(sequence_scaler, output_dir / "temporal_scaler.joblib")

    yield_scaler = StandardScaler().fit(train[["past_yield"]])
    for frame in frames.values():
        if not frame.empty:
            frame.loc[:, "past_yield"] = yield_scaler.transform(
                frame[["past_yield"]]
            ).ravel()
    joblib.dump(yield_scaler, output_dir / "past_yield_scaler.joblib")


def _fit_transform_soil(
    frames: dict[str, pd.DataFrame], soil: pd.DataFrame, output_dir: Path
) -> int:
    """Join soil matrices and scale each property using training districts only."""
    soil = soil.copy()
    soil["soil_features_parsed"] = soil["soil_features"].map(_parse_soil)
    if soil["DISTRICT_ID"].duplicated().any():
        raise ValueError("Soil input must contain one row per DISTRICT_ID")
    shapes = {value.shape for value in soil["soil_features_parsed"]}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent soil matrix shapes: {sorted(shapes)}")

    for split, frame in frames.items():
        frames[split] = frame.merge(
            soil[["DISTRICT_ID", "soil_features_parsed"]],
            on="DISTRICT_ID", how="left", validate="many_to_one"
        )
    # Soil is district-level and static in this design. Fit using each
    # training district once, rather than weighting districts by the number
    # of historical yield years they happen to contribute.
    training_districts = set(frames["train"]["DISTRICT_ID"].unique())
    train_matrices = soil.loc[
        soil["DISTRICT_ID"].isin(training_districts), "soil_features_parsed"
    ].tolist()
    if not train_matrices:
        raise ValueError("No soil matrices available for training districts")
    train_values = np.stack(train_matrices)
    # Matrix convention: [soil_property, depth]. Standardise each property
    # across training districts and all depths, then preserve CNN orientation.
    n_districts, n_properties, n_depths = train_values.shape
    scaler = StandardScaler().fit(
        train_values.transpose(0, 2, 1).reshape(-1, n_properties)
    )
    for frame in frames.values():
        scaled_values = []
        for matrix in frame["soil_features_parsed"]:
            if not isinstance(matrix, np.ndarray):
                raise ValueError(
                    "Missing soil matrix for a district-year record; "
                    "supply complete soil coverage or run without soil features"
                )
            scaled = scaler.transform(matrix.T).T
            scaled_values.append(scaled.tolist())
        frame.loc[:, "soil_features"] = scaled_values
        frame.drop(columns=["soil_features_parsed"], inplace=True)
    joblib.dump(scaler, output_dir / "soil_scaler.joblib")
    return n_properties * n_depths


def build_dataset(
    monthly_path: Path, yield_path: Path, output_dir: Path,
    months: list[int], train_end: int, val_end: int,
    soil_path: Path | None = None, year_offset: int = 0,
) -> dict:
    """Create fixed-month sequences, district-wise yield lags, splits and scalers."""
    monthly = pd.read_csv(monthly_path)
    yields = pd.read_csv(yield_path)
    _validate_columns(monthly, ["DISTRICT_ID", "YEAR", "MONTH"], "Monthly features")
    _validate_columns(yields, ["DISTRICT_ID", "sowing_year", "Yield"], "Yield data")
    if yields.duplicated(["DISTRICT_ID", "sowing_year"]).any():
        raise ValueError("Yield data has duplicate district/sowing_year rows")

    sequences, feature_cols = _build_sequences(monthly, months, year_offset)
    yields = yields.copy()
    yields["sowing_year"] = pd.to_numeric(yields["sowing_year"], errors="raise").astype(int)
    yields["Yield"] = pd.to_numeric(yields["Yield"], errors="coerce")
    labelled = sequences.merge(
        yields[["DISTRICT_ID", "sowing_year", "Yield"]],
        on=["DISTRICT_ID", "sowing_year"], how="inner", validate="one_to_one"
    )
    labelled = labelled.dropna(subset=["Yield"]).sort_values(
        ["DISTRICT_ID", "sowing_year"]
    ).reset_index(drop=True)

    # A lag is valid only if the immediately preceding sowing year exists.
    previous = labelled[["DISTRICT_ID", "sowing_year", "Yield"]].copy()
    previous["sowing_year"] += 1
    previous = previous.rename(columns={"Yield": "past_yield"})
    labelled = labelled.merge(
        previous, on=["DISTRICT_ID", "sowing_year"],
        how="left", validate="one_to_one"
    ).dropna(subset=["past_yield"]).reset_index(drop=True)

    years = sorted(labelled["sowing_year"].unique().tolist())
    train_years = [year for year in years if year <= train_end]
    val_years = [year for year in years if train_end < year <= val_end]
    test_years = [year for year in years if year > val_end]
    if not train_years or not val_years or not test_years:
        raise ValueError(
            "Chronological splits must each contain at least one year. "
            f"Available years={years}; train_end={train_end}; val_end={val_end}"
        )

    frames = {
        "train": labelled[labelled.sowing_year.isin(train_years)].copy(),
        "val": labelled[labelled.sowing_year.isin(val_years)].copy(),
        "test": labelled[labelled.sowing_year.isin(test_years)].copy(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    if soil_path is not None:
        soil = pd.read_csv(soil_path)
        _validate_columns(soil, ["DISTRICT_ID", "soil_features"], "Soil features")
        soil_dim = _fit_transform_soil(frames, soil, output_dir)
    else:
        soil_dim = 0

    _fit_transform_sequences(frames, output_dir)

    for split, frame in frames.items():
        # Keep only columns expected by the original Dataset class plus IDs.
        keep = ["DISTRICT_ID", "sowing_year", "feature_sequence",
                "Yield", "past_yield"]
        if soil_path is not None:
            keep.append("soil_features")
        frame[keep].to_csv(output_dir / f"{split}.csv", index=False)

    metadata = {
        "project": "DeepMaize",
        "crop": "maize",
        "season": "Kharif",
        "target_region": ["Maharashtra", "Madhya Pradesh", "Chhattisgarh"],
        "crop_calendar_months": months,
        "year_offset": year_offset,
        "feature_columns": feature_cols,
        "soil_matrix_shape": (
            list(_parse_soil(pd.read_csv(soil_path).iloc[0]["soil_features"]).shape)
            if soil_path is not None and not pd.read_csv(soil_path).empty else None
        ),
        "soil_flat_dimension": soil_dim,
        "split_years": {
            "train": [int(x) for x in train_years],
            "validation": [int(x) for x in val_years],
            "test": [int(x) for x in test_years],
        },
        "notes": [
            "Calendar is configurable and must be validated against local agronomy.",
            "All scalers are fitted using the training split only.",
            "District identifiers must be stable across yield, weather and soil inputs.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly", type=Path, required=True,
                        help="District/month feature CSV")
    parser.add_argument("--yields", type=Path, required=True,
                        help="District-year yield CSV")
    parser.add_argument("--soil", type=Path, default=None,
                        help="Optional district soil feature CSV")
    parser.add_argument("--output", type=Path, default=Path("../../data/datasets/maize_kharif"))
    parser.add_argument("--months", default="6,7,8,9,10,11",
                        help="Comma-separated Kharif months; defaults to June-November")
    parser.add_argument("--train-end", type=int, default=2018)
    parser.add_argument("--val-end", type=int, default=2021)
    parser.add_argument("--year-offset", type=int, default=0)
    args = parser.parse_args()
    metadata = build_dataset(
        args.monthly, args.yields, args.output, parse_months(args.months),
        args.train_end, args.val_end, args.soil, args.year_offset
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
