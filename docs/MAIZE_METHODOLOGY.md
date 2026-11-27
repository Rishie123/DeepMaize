# DeepMaize: methodology and replication notes

## Objective and scope

DeepMaize adapts the DeepWheat district-level multimodal yield-prediction pipeline to **Kharif maize** in **Maharashtra, Madhya Pradesh and Chhattisgarh**. The prediction unit is district × sowing year. The target is reported maize yield, not area or production.

This is a methodological adaptation, not a claim that the wheat model is already validated for maize. Model performance must be established with maize observations.

## Architecture retained from the reference implementation

The primary model remains the original `CNN_RNN`:

1. **Temporal branch:** a multi-layer bidirectional LSTM consumes monthly weather and vegetation-index features.
2. **Temporal self-attention:** attention learns which monthly hidden representations are most informative for the district-season yield estimate.
3. **Soil branch:** a one-dimensional CNN consumes soil properties arranged as property × depth.
4. **Lagged yield:** the previous available year's district yield is supplied directly to the regression head.
5. **Regression head:** concatenates temporal representation, lagged yield and soil representation, then predicts a scalar yield.

The architecture and forward pass are preserved; the adaptation happens in crop-season indexing, input assembly, split configuration, data validation and experiment protocol.

## Inputs and feature definitions

### Weather

The inherited weather workflow uses NASA POWER daily records. The reference paper's appendix lists precipitation, maximum/minimum/mean temperature, relative humidity, solar radiation, 10 m wind speed, specific humidity, surface pressure, and derived maximum/minimum vapour-pressure deficit. Confirm the exact feature column names in the generated daily weather CSV before building monthly features.

### Vegetation

The inherited workflow combines weather and remote-sensing tables by district and date, then aggregates daily observations to monthly means. The reference paper uses MODIS NDVI and EVI. Ensure the satellite composites are date-aligned and document missing-composite handling. An inner join can discard weather dates without satellite observations; the choice of join and valid-observation coverage should be explicitly assessed rather than silently assumed.

### Soil

The reference pipeline represents eight soil properties at six depths (0–5, 5–15, 15–30, 30–60, 60–100 and 100–200 cm), giving a 8 × 6 matrix per district. The CNN input convention is property × depth. Soil variables listed in the paper are soil organic carbon, pH, bulk density, sand, silt, clay, cation-exchange capacity and total nitrogen. Preserve units and provider-specific transformations in the metadata.

### Yield

Yield input must have `DISTRICT_ID`, `sowing_year` and `Yield`. The source should be documented with its table identifier, extraction date, units, revisions and any aggregation applied. Do not mix maize across Kharif/Rabi/summer seasons. District names alone are not safe keys: create and validate a state-qualified stable identifier, and maintain an explicit crosswalk when district boundaries or names change.

## Crop calendar and sequence construction

The default example uses June–November (months 6–11), with all six months assigned to the same calendar-year sowing year. This is an **initial configuration only**, not a universal maize calendar for all three states. Sowing and crop duration vary with district, rainfall onset, irrigation and maturity group. Prior to final training:

- obtain district/state agricultural crop-calendar references;
- verify whether the target yield series is season-labelled Kharif;
- select a biologically meaningful forecast cutoff;
- decide whether the model should predict at a fixed early-season date or after the full season;
- record the month list and sowing-year mapping in metadata.

A yield forecast made using June–November observations is only available after those observations have occurred. Do not describe it as an early-season forecast unless the input window is truncated accordingly.

The sequence builder expects one observation per district-month and retains only complete sequences for the configured month set. Partial sequences are excluded to guarantee consistent tensor dimensions. Alternative missing-month imputation should be a separately documented experiment.

## Splitting and leakage prevention

Splits are chronological by sowing year and applied consistently across districts. The code defaults to train through 2018, validation 2019–2021, and test 2022 onward, matching the reference paper's year split. Adjust these cutoffs to the maize yield data coverage before running the experiment. These cutoffs are configurable and must be reconciled with actual data coverage before use.

Temporal and lagged-yield scalers are fitted on training records only. Soil scaling is fitted on training districts only. Validation is used for early stopping and model selection. The test set must not be used to select checkpoints, hyperparameters, random seeds or feature sets.

Lagged yield is joined by the exact preceding sowing year. A missing intermediate year remains missing; the pipeline does not silently use the last available year as a one-year lag.

## Replication checklist

- [ ] Obtain district boundary polygons and record source/version/CRS.
- [ ] Build a stable state-qualified district identifier and a historical district crosswalk.
- [ ] Download yield statistics for maize and Kharif only; inspect missingness, units and revisions.
- [ ] Download daily weather and satellite data for all target districts and required years.
- [ ] Derive the paper's weather features and monthly NDVI/EVI summaries.
- [ ] Validate crop calendar and prediction cutoff with agronomic sources.
- [ ] Generate the district-month table and check completeness.
- [ ] Generate soil matrices with identical property/depth order for every district.
- [ ] Build chronological splits and inspect `metadata.json`.
- [ ] Train the proposed model and all benchmarks with validation-only model selection.
- [ ] Report MAE, RMSE, R² and a persistence (lagged-yield) baseline.
- [ ] Evaluate across multiple seeds and report mean and dispersion; do not cherry-pick a test-best run.
- [ ] Save predictions, configuration, scaler objects, package versions and data provenance.

## Important limitation

The repository does not contain the full raw yield, weather, vegetation and soil datasets. Creating the DeepMaize code path does not create maize labels or validate predictions. No empirical accuracy claim should be made until the data are assembled and the evaluation is run.
