"""Derive daily vapour-pressure-deficit features for DeepMaize."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def calculate_svp(temp_c: pd.Series) -> pd.Series:
    """Saturation vapour pressure (kPa), using the Tetens approximation."""
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def derive_weather_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Add maximum/minimum VPD without changing the original weather columns."""
    frame = pd.read_csv(input_path)
    required = {"T2M_MAX", "T2M_MIN", "RH2M"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Weather input is missing required columns: {sorted(missing)}")

    # NASA POWER reports RH2M in percent. The original paper uses this
    # temperature/RH-derived VPD approach for the two daily VPD features.
    svp_max = calculate_svp(pd.to_numeric(frame["T2M_MAX"], errors="coerce"))
    svp_min = calculate_svp(pd.to_numeric(frame["T2M_MIN"], errors="coerce"))
    rh_fraction = pd.to_numeric(frame["RH2M"], errors="coerce") / 100.0
    actual_vp = ((svp_max + svp_min) / 2.0) * rh_fraction
    frame["max_vpd"] = svp_max - actual_vp
    frame["min_vpd"] = svp_min - actual_vp
    frame.to_csv(output_path, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("weather_data_central_india.csv"))
    parser.add_argument("--output", type=Path, default=Path("weather_features_central_india.csv"))
    args = parser.parse_args()
    result = derive_weather_features(args.input, args.output)
    print(f"Wrote {len(result):,} weather rows with derived VPD to {args.output}")


if __name__ == "__main__":
    main()
