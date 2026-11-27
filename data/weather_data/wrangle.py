"""Prepare raw NASA POWER output for the DeepMaize feature pipeline."""
import argparse
from pathlib import Path
import pandas as pd


def wrangle_weather(input_file: Path, output_file: Path) -> pd.DataFrame:
    """Create a stable state-qualified district key and standard date column."""
    frame = pd.read_csv(input_file)
    if "DISTRICT_ID" not in frame.columns:
        required = {"State", "District"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Weather input is missing columns: {sorted(missing)}")
        frame["DISTRICT_ID"] = (
            frame["State"].astype(str).str.strip().str.upper()
            + "_" + frame["District"].astype(str).str.strip()
        )
    date_col = "Date" if "Date" in frame.columns else "DATE"
    if date_col not in frame.columns:
        raise ValueError("Weather input must contain Date or DATE")
    frame["DATE"] = pd.to_datetime(frame[date_col], errors="raise").dt.strftime("%Y-%m-%d")
    drop_cols = [col for col in ["State", "District", "Date"] if col in frame.columns]
    frame = frame.drop(columns=drop_cols)
    if frame.duplicated(["DISTRICT_ID", "DATE"]).any():
        raise ValueError("Weather input contains duplicate district/date rows")
    frame.to_csv(output_file, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NASA POWER combined CSV")
    parser.add_argument("--output", type=Path, default=Path("weather_data_central_india.csv"))
    args = parser.parse_args()
    result = wrangle_weather(args.input, args.output)
    print(f"Wrote {len(result):,} weather rows to {args.output}")


if __name__ == "__main__":
    main()
