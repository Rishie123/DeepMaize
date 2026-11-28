"""Convert downloaded Agmarknet observations into a market-month price panel.

The modelling entity is market × commodity × variety. The monthly target is
the median of the reported daily modal prices in a calendar month. Price rows
are never joined to crop yield labels. Arrival quantity is an explanatory
series and is aggregated separately to market × commodity × month before
being joined to each matching variety series.

Run from the repository root:
    python src/prices/prepare_price_panel.py \
        --prices data/market_data/agmarknet/agmarknet_prices.csv \
        --quantities data/market_data/agmarknet/agmarknet_quantities.csv \
        --output data/processed/maize_market_monthly.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalise_text(value: object) -> str:
    """Normalise a market label without attempting fuzzy district matching."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _required(frame: pd.DataFrame, names: set[str], label: str) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _market_identity(frame: pd.DataFrame) -> pd.Series:
    """Prefer provider market IDs; use normalised names only as a fallback."""
    market_id = frame.get("market_id", pd.Series(index=frame.index, dtype=object))
    market_name = frame.get("market", pd.Series(index=frame.index, dtype=object))
    has_id = market_id.notna() & market_id.astype(str).str.strip().ne("")
    if market_name.isna().all() and not has_id.any():
        raise ValueError(
            "No market_id or market field exists. Re-download using --level market; "
            "district-level aggregates are not suitable for mandi-level forecasting."
        )
    id_key = "id:" + market_id.fillna("").astype(str).str.strip()
    name_key = "name:" + market_name.map(normalise_text)
    return pd.Series(np.where(has_id, id_key, name_key), index=frame.index)


def _standardise_geography(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Add stable internal keys while retaining source labels and numeric IDs."""
    frame = frame.copy()
    _required(frame, {"state", "district", "date"}, label)
    frame["state"] = frame["state"].astype(str).str.strip()
    frame["district"] = frame["district"].astype(str).str.strip()
    frame["MARKET_KEY"] = _market_identity(frame)
    frame["market"] = frame.get("market", pd.Series(index=frame.index, dtype=object))
    market_id = frame.get("market_id", pd.Series(index=frame.index, dtype=object))
    missing_market_name = frame["market"].isna() | frame["market"].astype(str).str.strip().eq("")
    frame.loc[missing_market_name, "market"] = (
        "Market " + market_id.loc[missing_market_name].astype(str)
    )
    frame["commodity"] = frame.get(
        "commodity", pd.Series("Maize", index=frame.index)
    ).fillna("Maize").astype(str).str.strip()
    frame["variety"] = frame.get(
        "variety", pd.Series("UNSPECIFIED", index=frame.index)
    ).fillna("UNSPECIFIED").astype(str).str.strip()
    frame["commodity_key"] = frame["commodity"].map(normalise_text)
    frame["variety_key"] = frame["variety"].map(normalise_text)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["period"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    return frame


def prepare_price_panel(
    prices_path: Path,
    output_path: Path,
    quantities_path: Path | None = None,
    *,
    commodity_filter: str = "maize",
) -> tuple[pd.DataFrame, dict]:
    """Create one monthly record per market, commodity and variety.

    Daily modal prices are median-aggregated within month to reduce sensitivity
    to a small number of unusual daily reports. Monthly arrivals are summed
    across daily records at market × commodity level. The same commodity-level
    arrival total can appear on multiple variety rows; never sum it across
    varieties.
    """
    raw_prices = pd.read_csv(prices_path)
    _required(raw_prices, {"modal_price", "date", "state", "district"}, "Prices")
    prices = _standardise_geography(raw_prices, "Prices")
    prices["modal_price"] = pd.to_numeric(prices["modal_price"], errors="coerce")
    for column in ("min_price", "max_price"):
        prices[column] = pd.to_numeric(
            prices.get(column, pd.Series(np.nan, index=prices.index)),
            errors="coerce",
        )

    # The downloaded commodity is requested explicitly, but filter again so
    # a mixed or hand-concatenated file cannot silently contaminate maize runs.
    filter_key = normalise_text(commodity_filter)
    prices = prices[prices["commodity_key"].str.contains(
        re.escape(filter_key), na=False
    )].copy()
    if prices.empty:
        raise ValueError(
            f"No price rows matched commodity filter {commodity_filter!r}"
        )

    total_price_rows = len(prices)
    prices = prices[prices["modal_price"].gt(0)].copy()
    # Official data-entry validation expects min <= modal <= max. If both
    # bounds exist, flag and exclude rows that violate this check.
    has_bounds = prices["min_price"].notna() & prices["max_price"].notna()
    valid_order = (
        prices["min_price"].le(prices["modal_price"])
        & prices["modal_price"].le(prices["max_price"])
    )
    invalid_bound_rows = int((has_bounds & ~valid_order).sum())
    prices = prices[~(has_bounds & ~valid_order)].copy()
    if prices.empty:
        raise ValueError("No valid positive modal-price observations remain")

    # Aggregate prices to monthly values; use median of daily modal quotations.
    series_keys = [
        "state", "district", "MARKET_KEY", "market",
        "commodity", "commodity_key", "variety", "variety_key",
    ]
    monthly_prices = (
        prices.groupby(series_keys + ["period"], dropna=False)
        .agg(
            modal_price=("modal_price", "median"),
            min_price=("min_price", "median"),
            max_price=("max_price", "median"),
            reporting_days=("date", "nunique"),
            daily_price_rows=("modal_price", "size"),
        )
        .reset_index()
    )

    monthly_prices["arrival_qty"] = np.nan
    quantity_rows = 0
    if quantities_path is not None and quantities_path.exists():
        raw_quantities = pd.read_csv(quantities_path)
        if not raw_quantities.empty:
            _required(raw_quantities, {"date", "state", "district"}, "Quantities")
            quantities = _standardise_geography(raw_quantities, "Quantities")
            if "arrival_qty" not in quantities.columns:
                raise ValueError("Quantity CSV has no arrival_qty field")
            quantities["arrival_qty"] = pd.to_numeric(
                quantities["arrival_qty"], errors="coerce"
            )
            quantities = quantities[
                quantities["commodity_key"].str.contains(
                    re.escape(filter_key), na=False
                )
            ].copy()
            quantities = quantities[quantities["arrival_qty"].ge(0)].copy()
            quantity_rows = len(quantities)
            qkeys = [
                "state", "district", "MARKET_KEY", "commodity_key", "period"
            ]
            # A CEDA quantity row may already be aggregated to a period. Sum
            # distinct date/market records and avoid accidental duplicate rows.
            quantities = quantities.drop_duplicates(
                subset=qkeys + ["date", "arrival_qty"]
            )
            monthly_quantities = (
                quantities.groupby(qkeys, dropna=False)["arrival_qty"]
                .sum(min_count=1)
                .rename("arrival_qty")
                .reset_index()
            )
            monthly_prices = monthly_prices.merge(
                monthly_quantities,
                on=["state", "district", "MARKET_KEY", "commodity_key", "period"],
                how="left",
                suffixes=("", "_quantity"),
                validate="many_to_one",
            )
            if "arrival_qty_quantity" in monthly_prices.columns:
                monthly_prices["arrival_qty"] = monthly_prices.pop("arrival_qty_quantity")

    monthly_prices = monthly_prices.sort_values(
        ["state", "district", "market", "variety", "period"]
    ).reset_index(drop=True)

    # Stable readable series key. The category strings are metadata, not labels.
    monthly_prices["series_id"] = (
        monthly_prices["state"].map(normalise_text) + "|" +
        monthly_prices["district"].map(normalise_text) + "|" +
        monthly_prices["MARKET_KEY"].astype(str) + "|" +
        monthly_prices["commodity_key"].astype(str) + "|" +
        monthly_prices["variety_key"].astype(str)
    )
    if monthly_prices.duplicated(["series_id", "period"]).any():
        raise ValueError("Monthly panel contains duplicate series_id × period rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_prices.to_csv(output_path, index=False)
    metadata = {
        "project": "DeepMaize",
        "task": "maize_mandi_modal_price_forecasting",
        "source": "CEDA Agri Market Data / Agmarknet (DMI)",
        "source_price_file": str(prices_path),
        "source_quantity_file": str(quantities_path) if quantities_path else None,
        "commodity_filter": commodity_filter,
        "target": "monthly median of reported daily modal_price",
        "target_unit": "INR per quintal (verify against the downloaded provider metadata)",
        "aggregation": {
            "modal_price": "median of daily modal prices within market × variety × month",
            "min_price": "median of daily minimum prices within month",
            "max_price": "median of daily maximum prices within month",
            "arrival_qty": "sum of daily arrival quantity reports by market × commodity × month",
        },
        "quality": {
            "input_price_rows_for_commodity": int(total_price_rows),
            "invalid_or_nonpositive_modal_price_rows_removed": int(total_price_rows - len(prices) - invalid_bound_rows),
            "rows_removed_for_min_modal_max_order": invalid_bound_rows,
            "quantity_rows_used": quantity_rows,
            "monthly_panel_rows": int(len(monthly_prices)),
            "unique_market_variety_series": int(monthly_prices["series_id"].nunique()),
            "unique_markets": int(monthly_prices["MARKET_KEY"].nunique()),
            "date_min": str(monthly_prices["period"].min().date()),
            "date_max": str(monthly_prices["period"].max().date()),
            "arrival_coverage_fraction": float(monthly_prices["arrival_qty"].notna().mean()),
        },
        "note": (
            "Agmarknet prices are market quotations and do not represent farm-gate prices. "
            "Market-level arrival_qty is repeated across varieties when the source reports "
            "arrivals only at commodity level; do not sum it across varieties."
        ),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return monthly_prices, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--quantities", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/processed/maize_market_monthly.csv"))
    parser.add_argument("--commodity", default="maize")
    args = parser.parse_args()
    panel, metadata = prepare_price_panel(
        args.prices, args.output, args.quantities, commodity_filter=args.commodity
    )
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {len(panel):,} market-month price observations to {args.output}")


if __name__ == "__main__":
    main()
