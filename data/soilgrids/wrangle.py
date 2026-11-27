"""Wrangle SoilGrids downloads into paper-ordered district soil tensors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Explicit ordering is critical: CNN channel positions must have stable meaning.
SOIL_FEATURE_ORDER = [
    "Soil Organic Carbon",
    "Soil pH",
    "Bulk Density",
    "Sand Content",
    "Silt Content",
    "Clay Content",
    "Cation Exchange Capacity",
    "Total Nitrogen",
]
DEPTH_ORDER = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


def wrangle_soil_data(input_file: Path) -> pd.DataFrame:
    """Create one finite [8 properties, 6 depths] matrix per district."""
    frame = pd.read_csv(input_file)
    required = {"DISTRICT_ID", "feature_name", "depth", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Soil input missing columns: {sorted(missing)}")
    if frame.duplicated(["DISTRICT_ID", "feature_name", "depth"]).any():
        raise ValueError("Soil input contains duplicate district/property/depth records")

    unknown_features = set(frame["feature_name"].dropna()) - set(SOIL_FEATURE_ORDER)
    unknown_depths = set(frame["depth"].dropna()) - set(DEPTH_ORDER)
    if unknown_features or unknown_depths:
        raise ValueError(
            f"Unexpected SoilGrids labels. Features={unknown_features}; depths={unknown_depths}"
        )

    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    rows = []
    for district_id, district in frame.groupby("DISTRICT_ID", sort=True):
        matrix = district.pivot(index="feature_name", columns="depth", values="value")
        matrix = matrix.reindex(index=SOIL_FEATURE_ORDER, columns=DEPTH_ORDER)
        values = matrix.to_numpy(dtype=np.float32)
        if values.shape != (8, 6) or not np.isfinite(values).all():
            missing_cells = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"{district_id}: incomplete/non-finite soil matrix; "
                f"{missing_cells} of 48 property-depth cells missing"
            )
        rows.append({"DISTRICT_ID": district_id, "soil_features": values.tolist()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "downloaded_data_wcs/district_soil_data_central_india.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "downloaded_data_wcs/district_soil_data_central_india_wrangled.csv"
    ))
    args = parser.parse_args()
    frame = wrangle_soil_data(args.input)
    frame.to_csv(args.output, index=False)
    print(f"Wrote soil matrices for {len(frame)} districts to {args.output}")


if __name__ == "__main__":
    main()
