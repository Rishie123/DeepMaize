# DeepMaize price pipeline data dictionary

## 1. Raw Agmarknet/CEDA downloads

The raw download directory preserves API JSON responses alongside standardised CSVs.

| Field | Meaning |
|---|---|
| source | CEDA Agri Market Data source attribution |
| state, state_id | State label and provider identifier |
| district, district_id | District label and provider identifier |
| market, market_id | Mandi label and provider identifier when returned |
| commodity, commodity_id | Commodity label and provider identifier |
| variety | Variety label when returned; missing values become UNSPECIFIED in the monthly panel |
| date | Observation date or source period |
| min_price | Reported minimum wholesale price |
| max_price | Reported maximum wholesale price |
| modal_price | Reported modal wholesale price; the forecasting target |
| arrival_qty | Reported quantity arrivals; an optional historical predictor |

The active workflow is intended to run at market granularity. It requires market_id or market so that two mandis in the same district remain separate time series.

## 2. Monthly panel

One row per series_id × period. The series identity includes state, district, market identity, commodity and variety.

| Field | Meaning |
|---|---|
| state | State name |
| district | District name |
| MARKET_KEY | id:<market_id> where available, otherwise name:<normalised_market> |
| market | Market label |
| commodity | Commodity label |
| variety | Variety label, or UNSPECIFIED |
| period | Month start date |
| series_id | Stable composite key for market × commodity × variety |
| modal_price | Median of daily modal prices reported in that month; target |
| min_price | Median of daily minimum prices reported in the month |
| max_price | Median of daily maximum prices reported in the month |
| arrival_qty | Sum of arrival reports at market × commodity × month, when available |
| reporting_days | Number of distinct daily records in the monthly price aggregation |
| daily_price_rows | Number of raw price records contributing to the monthly aggregation |

Because arrival quantity may be reported for a market × commodity rather than variety, it can be repeated across variety rows. Never aggregate this repeated value across varieties.

## 3. Forecast features

Every historical month in the 12-month lookback is transformed into nine numeric features:

| Feature | Definition |
|---|---|
| log_modal_price | log(1 + monthly median modal price) |
| log_min_price | log(1 + monthly median minimum price) |
| log_max_price | log(1 + monthly median maximum price) |
| log_arrival_qty | log(1 + monthly market arrival quantity); missing treated as zero |
| arrival_missing | 1 if no arrival value was reported, otherwise 0 |
| history_month_sin | Sine encoding of historical calendar month |
| history_month_cos | Cosine encoding of historical calendar month |
| forecast_month_sin | Sine encoding of known target calendar month |
| forecast_month_cos | Cosine encoding of known target calendar month |

The target is log(1 + modal_price) for optimisation and is inverse transformed before reporting. Continuous features are standardised with a scaler fitted on training data only.

## 4. Prepared model splits

The dataset builder writes train.npz, val.npz and test.npz. Each split contains:

| Array | Meaning |
|---|---|
| X | Standardised sequences, shape samples × lookback × 9 |
| y_log | log(1 + target modal price) |
| y_price | Target monthly modal price in source units |
| last_price | Last historical modal price for persistence |
| seasonal_naive_price | Modal price 12 months earlier (for a 12-month lookback) |
| series_index | Learned categorical embedding index for market × variety |
| period | Forecast target month |
| series_id | Composite market × commodity × variety key |
| state, district, market, variety | Output labels for reporting |

Split boundaries apply to the target period. A training observation never includes a feature from its own target month.

## 5. Units and provenance

Price and arrival units must be recorded from source metadata. Do not assume that arrival quantities use the same unit for all commodities. Preserve source query dates, API metadata, raw response files, extraction date, CEDA attribution and data licence terms with each experiment.

Agmarknet modal prices are wholesale market quotations. They are not the same as a farmer's realised sale price or retail prices.

## 6. Quality checks and exclusions

- Missing/non-positive modal prices are excluded from the monthly panel.
- When min, modal and max are all present, rows violating min ≤ modal ≤ max are excluded.
- Duplicate series × month rows are rejected by the dataset builder.
- Lookback samples are retained only if all preceding calendar months exist.
- Arrival missingness is represented explicitly; no arrival value is fabricated.
- Historical district/market names should be reviewed where provider identifiers are absent or have changed.
