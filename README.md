# DeepMaize

**Market-level maize price forecasting** for Maharashtra, Madhya Pradesh and Chhattisgarh, India.

DeepMaize predicts the next month's wholesale **modal maize price** for an individual mandi and variety from its preceding 12 months of market observations. The active pipeline is a pooled LSTM trained on Agmarknet/CEDA price histories, with market-series identity embeddings and simple persistence benchmarks.

This is a change in research task from the earlier district-level crop-yield adaptation. The active objective is no longer maize yield prediction. Legacy yield-oriented code is retained in the repository for reference, but it is not used by the price-forecasting workflow documented below.

## Prediction definition

| Item | Current design |
|---|---|
| Crop | Maize |
| Region | Maharashtra, Madhya Pradesh, Chhattisgarh |
| Observation unit | Mandi × variety × month |
| Source observation | Daily Agmarknet minimum, maximum and modal wholesale prices |
| Monthly target | Median of daily modal prices reported within that mandi-month |
| Forecast horizon | One calendar month ahead |
| Lookback | Previous 12 complete consecutive months |
| Main model | Pooled two-layer LSTM with a learned market-series embedding |
| Baselines | Last observed month (persistence), same month last year |
| Evaluation | MAE, RMSE, R² and sMAPE in the source price unit |
| Split | Chronological by target month; train ≤ 2019, validation 2020–2021, test ≥ 2022 by default |

The target is an Agmarknet wholesale market quotation, not a farm-gate price, MSP, retail price or guaranteed transaction price. The model predicts a market-specific next-month modal-price statistic, not an individual farmer's realised sale price.

## 1. Environment

Python 3.10+ is recommended.

~~~bash
git clone https://github.com/Rishie123/DeepMaize.git
cd DeepMaize
git checkout deepmaize-kharif
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

The active price-training model uses PyTorch. The downloader and preprocessing modules use pandas, NumPy, scikit-learn, joblib and requests.

## 2. Download maize mandi prices

The downloader uses CEDA Agri Market Data, which attributes its Agmarknet observations to the Directorate of Marketing & Inspection (DMI), Ministry of Agriculture and Farmers Welfare. The CEDA API exposes commodity/geography metadata and price and quantity endpoints.

Start with a short single-state smoke test:

~~~bash
python data/market_data/download_agmarknet.py \
  --start-date 2020-01-01 \
  --end-date 2020-12-31 \
  --states Maharashtra \
  --level market \
  --calculation-type d \
  --output-dir data/market_data/agmarknet_smoke
~~~

For the three-state historical pull:

~~~bash
python data/market_data/download_agmarknet.py \
  --start-date 2007-01-01 \
  --end-date 2023-12-31 \
  --states Maharashtra "Madhya Pradesh" Chhattisgarh \
  --level market \
  --calculation-type d \
  --output-dir data/market_data/agmarknet
~~~

Use market level for this pipeline: it requests CEDA's market-level download view, rather than district/state summary data. The commodity defaults to maize and is resolved from the live commodity catalogue. Use daily data as the input to the monthly aggregation step; monthly data can be used for a shorter exploratory run, but verify what the API returns for the selected chart type.

The output folder contains:

- agmarknet_prices.csv — minimum, maximum and modal price records.
- agmarknet_quantities.csv — arrivals, kept separately.
- download_log.csv — per-geography and endpoint success/failure information.
- download_metadata.json — query filters, extraction timestamp and counts.
- raw/ — raw JSON responses and metadata catalogues.

Read docs/AGMARKNET_INGESTION.md for endpoint behaviour, troubleshooting and data provenance.

### Source terms

CEDA's published API terms currently permit downloading, displaying and including the data in other products **for non-commercial purposes**, subject to attribution. They specify the citation “CEDA Agri Market Data (CEDA-AMD), 2000-2023”, Centre for Economic Data & Analysis, Ashoka University. The terms also require CEDA credit in applications and visualisations. **Do not use this CEDA API data for a commercial forecasting service or commercial deployment without written permission or another suitable data licence.** Check the live API terms at https://ceda.ashoka.edu.in/api-terms-conditions/ before each use. The official Open Government Data catalogue also lists the daily mandi-price dataset, but confirm the applicable licence and access method for that source independently.

## 3. Build the monthly market-price panel

~~~bash
python src/prices/prepare_price_panel.py \
  --prices data/market_data/agmarknet/agmarknet_prices.csv \
  --quantities data/market_data/agmarknet/agmarknet_quantities.csv \
  --output data/processed/maize_market_monthly.csv
~~~

This produces one row per mandi × commodity × variety × calendar month.

Aggregation rules:

- Monthly target modal_price: median of reported daily modal prices in the month.
- min_price and max_price: median of the reported daily minimum and maximum quotations.
- arrival_qty: sum of reported arrivals over the month at market × commodity level.
- reporting_days: number of distinct dates with a modal-price observation.

The quantity series may be reported at market × commodity rather than market × variety. When joined to variety-level prices, the same market-level monthly arrival value is repeated across varieties. **Do not sum arrival values over varieties.** Missing arrivals remain missing in the panel and are represented with an arrival-missing indicator in the model inputs.

The panel builder requires a market identifier or market name. It does not accept district-level aggregate rows as if they were individual mandi prices. It also filters non-positive modal prices and rows that violate the source's minimum ≤ modal ≤ maximum consistency rule where all three values are present. Raw source data are retained separately.

## 4. Create forecasting windows

~~~bash
python src/prices/build_price_dataset.py \
  --panel data/processed/maize_market_monthly.csv \
  --output data/datasets/maize_price \
  --window 12 \
  --train-end 2019 \
  --val-end 2021
~~~

For a forecast of month t, the features are built only from months t-12 through t-1. The model receives:

- log-transformed historical modal, minimum and maximum prices;
- log-transformed historical arrival quantity and an arrival-missing indicator;
- monthly sine/cosine encodings for seasonality;
- the known calendar month of the forecast target;
- a categorical market × variety series index.

The target is log(1 + modal_price) during model fitting. Reported predictions and metrics are transformed back into the original source price unit.

A sample is created only when all 12 preceding calendar months are available for that same market × variety series. The builder does not forward-fill prices, bridge missing calendar months or use target-month data. Scalers are fit using training samples only.

Output files are train.npz, val.npz, test.npz, feature_scaler.joblib and metadata.json. Split cutoffs must be adjusted after inspecting the actual data coverage. A test period with no complete 12-month lookback cannot be forecast by this setup.

## 5. Train and evaluate the price model

~~~bash
python src/prices/train_price_model.py \
  --dataset data/datasets/maize_price \
  --output experiments/maize_price_lstm \
  --epochs 100 \
  --patience 12 \
  --batch-size 128
~~~

The LSTM encodes the 12-month sequence and concatenates its temporal representation with a learned embedding for the market × variety series. It predicts the next month's log modal price.

The best checkpoint is selected by validation RMSE after inverse transformation to the original price unit. The held-out test split is not used for early stopping or checkpoint selection.

The training report includes PriceLSTM metrics, persistence (last monthly modal price), seasonal naive (same calendar month one year earlier), test predictions by market and variety, training history and the best validation checkpoint.

Metrics include MAE, RMSE, R² and sMAPE. Because price levels differ across mandis, inspect both pooled metrics and error distributions by market. A good pooled score alone does not prove that each mandi is forecast reliably.

## 6. Tests

From the repository root:

~~~bash
python -m compileall -q src data tests
python -m pytest -q
~~~

Tests use synthetic price records and mocked CEDA responses. They validate the transformation and time-window logic, not live API coverage or empirical price-forecast accuracy.

## 7. Research and deployment safeguards

- The monthly modal price is a statistical aggregation of mandi daily modal quotations; it is not a weighted transaction-level average.
- Verify price and arrival units from source metadata before interpreting model outputs. Do not assume all arrival reports share one unit.
- Market names and market IDs can change. Preserve provider IDs and inspect unmatched markets before comparing historical records.
- Forecasts must be timestamped with the last complete month available to the model. Do not label delayed source data as live.
- Market closures and reporting gaps are informative. The current model requires a complete monthly lookback and excludes windows with missing months.
- A forecast one month ahead is not an intraday price forecast and is not a guarantee of the price at the next mandi auction.
- CEDA API usage is currently subject to non-commercial terms. Obtain explicit permission or use a separately licensed source before commercial use.
- The previous yield, weather and soil modules are legacy code and are not part of the active price pipeline.

## References

- CEDA Agri Market Data API documentation: https://api.ceda.ashoka.edu.in/documentation/
- CEDA Agri Market Data portal: https://agmarknet.ceda.ashoka.edu.in/
- CEDA API terms and conditions: https://ceda.ashoka.edu.in/api-terms-conditions/
- Government of India OGD daily market-price dataset: https://data.gov.in/resource/variety-wise-daily-market-prices-data-commodity
- AGMARKNET data-entry SOP: https://www.agmarknet.gov.in/DailyDataEntry/Docs/SOP_DEO.pdf
