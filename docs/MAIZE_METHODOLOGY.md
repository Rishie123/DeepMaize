# DeepMaize: maize mandi-price forecasting methodology

## Objective

Predict the next calendar month's wholesale modal price of maize for an individual mandi and variety in Maharashtra, Madhya Pradesh and Chhattisgarh.

This is a market-price time-series forecasting task. It supersedes the earlier yield-prediction adaptation as the active project objective. Annual crop yield, soil properties and the seasonal crop calendar are not targets or required inputs in the active price workflow.

## Observation and target grain

The source is Agmarknet market reporting accessed through CEDA Agri Market Data. Daily records carry market/commodity context and minimum, maximum and modal wholesale prices. The model's observation unit is:

    state × district × mandi × maize variety × month

For each calendar month, the target is the median of the reported daily modal-price observations for that market-variety. This robust aggregation is selected to dampen individual reporting spikes. It is not a transaction-weighted average and should not be interpreted as a guaranteed executable price.

## Forecast origin and horizon

The forecast origin is the end of the previous complete calendar month. For a target month t:

    Inputs: t-12, t-11, ..., t-1
    Target: monthly median modal price in t

A full 12 consecutive months of observations is required for the same market-variety series. Missing calendar months break a window; the pipeline does not silently substitute the last available quotation. The default horizon is one month and the default lookback is 12 months. Both are recorded in dataset metadata.

## Input features

Historical price features are minimum, maximum and modal price; historical market arrival quantity is an optional predictor. Each numeric market variable is log1p transformed to reduce scale skew and standardised using training records only. Missing arrivals are represented as zero after log transformation with a separate missingness flag. Seasonal month-of-year and forecast-month encodings are included.

Target-month prices and target-month arrivals are not available to the forecast and must never be joined into the corresponding sample.

## Model

A pooled two-layer LSTM encodes the sequence of prior monthly market features. Its final hidden state is concatenated with a learned embedding identifying the market × commodity × variety series. A small fully connected regression head outputs log1p of the next-month modal price. Predictions are transformed back to the source price unit for evaluation.

Pooling allows the recurrent model to learn common seasonal and price dynamics across mandis while preserving a learned market-level baseline. A market absent from training has no learned market-specific embedding and is a cold-start case; results for such markets are not expected to be reliable.

## Baselines and evaluation

The neural forecast is compared with:

1. Persistence: next month equals the most recent observed monthly modal price.
2. Seasonal naive: next month equals the modal price from the same month one year earlier.

Report MAE, RMSE, R² and sMAPE in source units / percent. Also inspect market-level and state-level errors, sample counts, reporting days and arrival coverage. A pooled metric can hide weak or sparse individual mandis.

## Chronological validation design

Default splits use the target month year:

- Training: through 2019.
- Validation: 2020–2021.
- Test: 2022 onward.

These boundaries are provisional and must be adjusted to source coverage and sample counts. The scaler is fitted only on training windows. The validation split chooses the checkpoint by RMSE in price units. The test split is not used for training, early stopping, feature selection or model selection.

## Data-quality decisions

- Keep source raw JSON and extraction metadata.
- Prefer market IDs over fuzzy market-name joins.
- Require market-level source data; district aggregates do not stand in for individual mandis.
- Exclude non-positive modal-price records.
- When all min/modal/max values are present, enforce min ≤ modal ≤ max.
- Aggregate daily modal price to a monthly median.
- Aggregate arrivals separately at market × commodity × month. If the source reports arrivals at that coarser grain, the value may repeat across variety rows and must not be summed over varieties.
- Preserve missing arrival values and report their coverage.
- Do not interpolate missing modal prices for the active evaluation.

## Data licence and intended use

CEDA's published API terms currently state that the API data may be downloaded, displayed or included in other products for non-commercial purposes at no cost, subject to CEDA citation and visual attribution requirements. The user should verify the current terms before each use. Commercial forecasting, customer-facing or business use must not rely on this API data without explicit permission or a separately suitable licence.

## Research limitations

- Agmarknet prices represent wholesale market reports rather than farm-gate prices.
- The modal price is a reported market statistic, not the complete transaction distribution.
- Reporting coverage varies over markets and time.
- Market identity, variety naming and district boundaries can change; provider IDs and historical crosswalks require review.
- Monthly median price is a different target from daily modal price and futures/forward prices.
- No empirical accuracy claim is justified until real data have been downloaded, audited, modelled and evaluated on the chronological test split.
