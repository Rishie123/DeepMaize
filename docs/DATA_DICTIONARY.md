# DeepMaize data dictionary

This document defines the tabular interfaces between data collection, preprocessing, model training and evaluation. Retain the original provider's file, download date, units and processing metadata alongside every experiment.

## 1. Identifiers

| Field | Meaning | Rule |
|---|---|---|
| `DISTRICT_ID` | State-qualified district join key | Uppercase `STATE_DISTRICT`, e.g. `MAHARASHTRA_NAGPUR` |
| `State` | State name from district boundary layer | One of MAHARASHTRA, MADHYA PRADESH, CHHATTISGARH |
| `District` | District name from boundary layer | Uppercase, trimmed |
| `YEAR` | Calendar year of the monthly feature observation | Integer |
| `MONTH` | Calendar month of the feature observation | Integer 1–12 |
| `sowing_year` | Crop season's sowing year / year label | Integer; Kharif 2019-20 is normally mapped to 2019, subject to source semantics |

State-qualified identifiers prevent two states' identically named districts being accidentally merged. They do **not** solve historical boundary changes. A district crosswalk is required when historical yield labels use a different district vintage from the weather/soil polygons.

## 2. Monthly temporal feature table

One row per `DISTRICT_ID × YEAR × MONTH`.

| Feature family | Source field | Meaning |
|---|---|---|
| Weather | `PRECTOTCORR` | Corrected precipitation |
| Weather | `T2M_MAX` | Daily maximum 2 m air temperature |
| Weather | `T2M_MIN` | Daily minimum 2 m air temperature |
| Weather | `T2M` | Daily mean 2 m air temperature |
| Weather | `RH2M` | Relative humidity at 2 m |
| Weather | `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave radiation |
| Weather | `WS10M` | Wind speed at 10 m |
| Weather | `QV2M` | Specific humidity at 2 m |
| Weather | `PS` | Surface pressure |
| Derived weather | `max_vpd` | Maximum vapour-pressure deficit, kPa |
| Derived weather | `min_vpd` | Minimum vapour-pressure deficit, kPa |
| Vegetation | `NDVI` | Normalised Difference Vegetation Index |
| Vegetation | `EVI` | Enhanced Vegetation Index |

The intended temporal feature count is 13 when all variables are available. Weather and vegetation inputs are merged by district and date using an outer join, then aggregated separately to monthly means. The builder rejects a district-season when any configured month has a non-finite feature value.

**Units:** NASA POWER response headers/metadata and the exact API parameter documentation for the requested temporal product are authoritative. Store that metadata with the downloaded CSV. Do not infer unit conversions from column names. Confirm temperature, precipitation, radiation, humidity, wind and pressure units before interpreting model coefficients or transferring hyperparameters. NDVI/EVI quality screening, valid range, scale-factor application and fill-value handling must be completed in the MODIS/GEE export before monthly aggregation.

## 3. Soil feature matrix

One static matrix per district, represented in each district-season row as `soil_features` with shape `[8, 6]`.

| Property index | Soil property | Depth indices |
|---:|---|---|
| 0 | Soil Organic Carbon | All six depths, shallow to deep |
| 1 | Soil pH | All six depths |
| 2 | Bulk Density | All six depths |
| 3 | Sand Content | All six depths |
| 4 | Silt Content | All six depths |
| 5 | Clay Content | All six depths |
| 6 | Cation Exchange Capacity | All six depths |
| 7 | Total Nitrogen | All six depths |

Depth index order:

1. 0–5 cm
2. 5–15 cm
3. 15–30 cm
4. 30–60 cm
5. 60–100 cm
6. 100–200 cm

The matrix orientation is **property × depth**, not depth × property. SoilGrids provider units and property-specific scale factors must be preserved in provenance notes. The downloader computes district polygon means; inspect failed and partial WCS requests. The wrangler rejects missing or non-finite property/depth cells.

## 4. Yield table

The raw yield preparation input is long format with these fields (aliases can be passed on the CLI):

| Raw field | Meaning |
|---|---|
| `State` | State name |
| `District` | District name |
| `Crop` | Crop; filtered to labels containing MAIZE |
| `Season` | Season; filtered to labels containing KHARIF |
| `Year` | Season year, e.g. 2019 or 2019-20 |
| `Yield` | Source-reported yield |

The prepared yield table has one row per `DISTRICT_ID × sowing_year`:

| Field | Meaning |
|---|---|
| `DISTRICT_ID` | State-qualified district key |
| `sowing_year` | Starting year of Kharif season |
| `Yield` | Source yield, unchanged apart from optional explicit multiplier |

The target value remains in the original source unit in the generated dataset. Record the source unit and any conversion; do not mix yield in kg/ha, t/ha or other units. The preparation script stops when there are multiple records per district-year, because choosing an aggregation method is a substantive data decision.

## 5. Model dataset CSVs

The sequence builder writes `train.csv`, `val.csv`, `test.csv`. Their key columns are:

| Column | Stored value |
|---|---|
| `DISTRICT_ID` | District key |
| `sowing_year` | Kharif season year |
| `feature_sequence` | Nested list shaped `[number_of_months, number_of_temporal_features]`; temporal values are StandardScaler-transformed |
| `Yield` | Unscaled target in the source yield unit |
| `past_yield` | Previous-year district yield after StandardScaler transform |
| `soil_features` | Optional nested list shaped `[8, 6]`; property-wise standardised using training districts only |

All scalers are fit on training data only and saved alongside the split files. Validation and test values are transformed with those same saved scalers. The target `Yield` is not standardised by this pipeline.

## 6. Split metadata and data attrition

`metadata.json` records the configured crop calendar, feature order, soil dimensions, split years and `data_quality` counts:

- raw monthly rows and rows inside the configured months;
- candidate district-year groups and groups with every configured month;
- complete finite temporal sequences;
- yield rows and district-years with a matching yield label;
- district-years with the exact previous year's yield available;
- final records and unique districts per split.

Inspect these counts before training. A high attrition rate can change which districts and years are represented, and should be reported. The builder's strict complete-month rule is intended to prevent ragged sequences, not to claim that excluded records are random.

## 7. Provenance checklist

For each source, preserve the exact download/export date, URL or table identifier, data release, spatial reference system, temporal range, unit metadata, quality flags, fill-value handling, and all transformations. For district data, preserve the polygon release and historical name/boundary crosswalk. For yield data, record crop/season filters, source-reported unit, aggregation rules and any district-year exclusions.
