# DeepMaize price-forecasting architecture

This page describes the **active price forecasting pipeline** in src/prices/. The older src/rnn/ CNN-RNN/soil/yield architecture is retained as legacy code and is not used for price prediction.

## Prediction unit and target

The active modelling entity is market × maize variety. Each entity is represented as a monthly time series. The target for month t is the median of the reported daily Agmarknet modal-price quotations for that market and variety during month t.

The forecast horizon is one calendar month. Inputs contain only the previous 12 consecutive calendar months. The model estimates the next monthly modal-price statistic from information available through the preceding month.

## End-to-end flow

    Agmarknet / CEDA daily mandi prices
                 |
                 v
    market × variety × date price records
                 |
                 +----> separate daily arrival records
                 |                    |
                 v                    v
       monthly price aggregation   monthly market arrivals
                 |                    |
                 +-------- merge ----+
                           |
                           v
             monthly market-price panel
                           |
                           v
       rolling 12-month windows, chronological split
                           |
                           v
        training-only feature standardisation
                           |
                           v
             pooled two-layer PriceLSTM
                           |
                           v
        inverse log transform -> price-unit output
                           |
                           v
         persistence and seasonal-naive baselines
                           |
                           v
          MAE / RMSE / R² / sMAPE and predictions

## Data preparation

The Agmarknet downloader retrieves prices and arrivals separately. Price responses are archived as raw JSON and normalised to a common CSV schema. The API is requested at market granularity; the preprocessor refuses a file with no market ID or market name because district-wide prices cannot be treated as individual mandi observations.

The monthly panel builder:

1. filters to the requested commodity (maize by default);
2. parses dates and creates calendar-month keys;
3. checks that positive modal prices are consistent with available minimum and maximum prices;
4. calculates the median of daily modal prices within a market × variety × month;
5. aggregates arrival quantity at market × commodity × month and joins it to variety rows;
6. writes panel metadata with row counts, unique series, date coverage and arrival coverage.

The quantity feature is not a target. When arrivals are market × commodity values, it may appear on multiple variety rows and must not be summed across varieties.

## Forecast-window builder

The builder creates one supervised sample for each forecast month with a complete history window.

For target month t and lookback length L=12:

    X(t) = [features(t-12), ..., features(t-1)]
    y(t) = log(1 + modal_price(t))

Each historical time step contains log modal/minimum/maximum price, log arrival quantity, an arrival-missing flag and month-of-year sine/cosine terms. The known target month-of-year is also encoded. A stable series index identifies the market × commodity × variety sequence.

The builder does not fill a missing monthly price or jump over a missing calendar month. A missing month breaks the lookback window until 12 subsequent consecutive observations are available. It creates chronological train/validation/test datasets using the target month, not the last input month.

The continuous input scaler is fitted on training windows only and applied unchanged to validation and test windows. The target is log1p transformed but not standardised.

## PriceLSTM

The model contains a two-layer LSTM encoder and market-series embedding:

- Sequence input: batch × 12 months × 9 features.
- Recurrent encoder: two LSTM layers, hidden dimension configurable (default 64).
- Market context: learned embedding for the market × variety time-series identity (default 12 dimensions).
- Regression head: concatenate final temporal state and series embedding, then map through a ReLU/dropout hidden layer to one scalar.
- Output: predicted log1p next-month modal price.

The model is pooled across all retained market × variety series. The embedding can learn persistent differences between mandis observed in training. New series absent from training receive the unknown embedding index and should be treated as unsupported cold-start cases.

## Training and selection

Training uses AdamW and mean squared error in log-price space. Early stopping and checkpoint selection are based on validation RMSE after transforming predictions back to the original price scale. The test split is scored once after model selection.

The report compares the LSTM with two benchmarks:

- Persistence: next month's price equals the last available month's modal price.
- Seasonal naive: next month's price equals the price 12 months earlier.

Metrics are MAE, RMSE, R² and symmetric MAPE (sMAPE). Evaluate errors by state and market in addition to pooled metrics.

## Leakage controls

- Inputs stop at the month before the forecast target.
- Current/target-month arrivals are never used to predict that month's price.
- No target-month minimum, maximum or modal price enters the features.
- Windows require consecutive calendar months; missing data are not bridged.
- The feature scaler is fitted only on training windows.
- Train/validation/test splits are chronological by target month.
- Validation selects checkpoints; test metrics do not select epochs, seeds or features.

## Limitations

- The prediction is a monthly median of reported daily modal prices, not a transaction-weighted price.
- Agmarknet is an observational reporting system; coverage differs by market and time.
- The model is not a farm-gate or retail price model.
- CEDA API usage is currently subject to non-commercial terms and attribution requirements.
- No real-data training or empirical forecasting accuracy is claimed until the downloader has been run successfully and data quality has been reviewed.
