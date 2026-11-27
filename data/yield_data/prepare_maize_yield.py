"""Prepare long-format district × season × crop yield rows for DeepMaize.

Input schema (column names configurable):
    State, District, Crop, Season, Year, Yield

Yield units are inherited from the source table unless --yield-multiplier is
provided. Record the source units and conversion in your experiment notes.
"""
import argparse
import re
from pathlib import Path
import pandas as pd


TARGET_STATES = {"MAHARASHTRA", "MADHYA PRADESH", "CHHATTISGARH"}


def normalise_label(series: pd.Series) -> pd.Series:
    """Uppercase text and collapse repeated whitespace for stable join keys."""
    return series.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def extract_sowing_year(value: object) -> int:
    """Read the starting calendar year from e.g. 2019, 2019-20 or 2019-2020."""
    match = re.match(r"^\s*(\d{4})", str(value))
    if not match:
        raise ValueError(f"Cannot parse sowing/season year from {value!r}")
    return int(match.group(1))


def prepare_yield(
    input_path: Path,
    output_path: Path,
    state_col: str = "State",
    district_col: str = "District",
    crop_col: str = "Crop",
    season_col: str = "Season",
    year_col: str = "Year",
    yield_col: str = "Yield",
    yield_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Filter Kharif maize observations and write model-compatible yield rows."""
    frame = pd.read_csv(input_path)
    required = {state_col, district_col, crop_col, season_col, year_col, yield_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Yield input is missing columns: {sorted(missing)}")

    frame["_state"] = normalise_label(frame[state_col])
    frame["_district"] = normalise_label(frame[district_col])
    frame["_crop"] = normalise_label(frame[crop_col])
    frame["_season"] = normalise_label(frame[season_col])
    frame = frame[
        frame["_state"].isin(TARGET_STATES)
        & frame["_crop"].str.contains("MAIZE", regex=False)
        & frame["_season"].str.contains("KHARIF", regex=False)
    ].copy()
    frame["sowing_year"] = frame[year_col].map(extract_sowing_year)
    frame["Yield"] = pd.to_numeric(frame[yield_col], errors="coerce") * yield_multiplier
    frame = frame.dropna(subset=["Yield"])
    frame = frame[frame["Yield"] > 0].copy()
    frame["DISTRICT_ID"] = frame["_state"] + "_" + frame["_district"]

    result = frame[["DISTRICT_ID", "sowing_year", "Yield"]].copy()
    if result.duplicated(["DISTRICT_ID", "sowing_year"]).any():
        duplicates = result.loc[
            result.duplicated(["DISTRICT_ID", "sowing_year"], keep=False),
            ["DISTRICT_ID", "sowing_year"]
        ].drop_duplicates()
        raise ValueError(
            "Multiple Kharif maize yield records per district/year. "
            f"Aggregate explicitly before modelling. Examples: {duplicates.head().to_dict('records')}"
        )
    result = result.sort_values(["DISTRICT_ID", "sowing_year"]).reset_index(drop=True)
    if result.empty:
        raise ValueError(
            "No Kharif maize rows found for Maharashtra, Madhya Pradesh or Chhattisgarh"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "../../data/yield_data/maize_kharif_central_india.csv"
    ))
    parser.add_argument("--state-col", default="State")
    parser.add_argument("--district-col", default="District")
    parser.add_argument("--crop-col", default="Crop")
    parser.add_argument("--season-col", default="Season")
    parser.add_argument("--year-col", default="Year")
    parser.add_argument("--yield-col", default="Yield")
    parser.add_argument("--yield-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    result = prepare_yield(
        args.input, args.output, args.state_col, args.district_col,
        args.crop_col, args.season_col, args.year_col, args.yield_col,
        args.yield_multiplier
    )
    print(f"Wrote {len(result):,} Kharif maize yield rows to {args.output}")


if __name__ == "__main__":
    main()
