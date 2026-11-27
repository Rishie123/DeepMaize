"""Combine daily weather and MODIS vegetation features into monthly district rows.

An outer date join is intentional: daily weather and 16-day satellite composites
do not share every observation date. Monthly means are calculated independently
per variable. A month with no observation for a required variable remains NaN
and is rejected by the sequence builder instead of disappearing silently.
"""
import argparse
from pathlib import Path
import pandas as pd


def make_monthly_features(
    weather_path: Path, vegetation_path: Path, output_path: Path
) -> pd.DataFrame:
    """Merge daily weather/vegetation tables and calculate district-month means."""
    weather = pd.read_csv(weather_path)
    vegetation = pd.read_csv(vegetation_path)

    weather_date = "DATE" if "DATE" in weather.columns else "Date"
    vegetation_date = "DATE" if "DATE" in vegetation.columns else "date"
    for frame, date_col, label in [
        (weather, weather_date, "weather"),
        (vegetation, vegetation_date, "vegetation"),
    ]:
        required = {"DISTRICT_ID", date_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} input is missing columns: {sorted(missing)}")
        frame[date_col] = pd.to_datetime(frame[date_col], errors="raise").dt.normalize()
        if frame.duplicated(["DISTRICT_ID", date_col]).any():
            raise ValueError(f"{label} input has duplicate district/date rows")

    weather = weather.rename(columns={weather_date: "DATE"})
    vegetation = vegetation.rename(columns={vegetation_date: "DATE"})
    weather_features = [c for c in weather.columns if c not in {"DISTRICT_ID", "DATE"}]
    vegetation_features = [c for c in vegetation.columns if c not in {"DISTRICT_ID", "DATE"}]
    overlap = set(weather_features) & set(vegetation_features)
    if overlap:
        raise ValueError(f"Weather and vegetation columns overlap: {sorted(overlap)}")

    for col in weather_features:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    for col in vegetation_features:
        vegetation[col] = pd.to_numeric(vegetation[col], errors="coerce")

    merged = weather.merge(
        vegetation, on=["DISTRICT_ID", "DATE"], how="outer",
        validate="one_to_one", sort=False
    )
    merged["YEAR"] = merged["DATE"].dt.year
    merged["MONTH"] = merged["DATE"].dt.month
    feature_cols = weather_features + vegetation_features
    monthly = merged.groupby(
        ["DISTRICT_ID", "YEAR", "MONTH"], as_index=False
    )[feature_cols].mean()
    monthly = monthly[["DISTRICT_ID", "YEAR", "MONTH"] + feature_cols]
    monthly = monthly.sort_values(
        ["DISTRICT_ID", "YEAR", "MONTH"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_path, index=False)
    return monthly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", type=Path, default=Path(
        "weather_features_central_india.csv"
    ))
    parser.add_argument("--vegetation", type=Path, required=True,
                        help="Dated MODIS NDVI/EVI CSV with DISTRICT_ID")
    parser.add_argument("--output", type=Path, default=Path(
        "../../data/processed/maize_monthly_features.csv"
    ))
    args = parser.parse_args()
    result = make_monthly_features(args.weather, args.vegetation, args.output)
    print(f"Wrote {len(result):,} district-month rows to {args.output}")


if __name__ == "__main__":
    main()
