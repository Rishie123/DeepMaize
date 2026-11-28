# DeepMaize

District-level **Kharif maize yield modelling** for Maharashtra, Madhya Pradesh and Chhattisgarh, adapted from the published DeepWheat multimodal yield-prediction implementation.

DeepMaize retains the reference model's multimodal design:

- **Temporal branch:** monthly meteorological and vegetation features enter a bidirectional LSTM.
- **Temporal self-attention:** the sequence representation is weighted over the monthly hidden states.
- **Soil branch:** an eight-property, six-depth SoilGrids matrix is processed by a 1D CNN.
- **Lagged-yield input:** the previous sowing year's district yield is passed directly to the regression head.
- **Regression head:** the branches are concatenated and mapped to one district-year yield prediction.

The source paper is *A hybrid deep learning approach for winter wheat yield prediction: evidence from leveraging multi-source data*, *Scientific Reports* (2026), [doi:10.1038/s41598-026-56576-5](https://doi.org/10.1038/s41598-026-56576-5). The architecture is reused as a research starting point; maize-specific crop calendars, features, splits and performance must be validated independently. This repository does not claim that wheat-trained weights or reported wheat performance transfer to maize.

## Current status

The repository contains a separate maize data-preparation path, region-specific weather and SoilGrids downloaders, training configurations for soil and no-soil variants, methodology documentation and automated unit tests.

**No raw maize yield, MODIS or district-boundary dataset is bundled. No maize model has yet been trained or empirically validated.** The default June-December window is a provisional Kharif example, not a claim that every district in all three states shares one crop calendar. The [Madhya Pradesh crop-season table](https://www.slbcmadhyapradesh.in/docs/184th_slbc_agenda_final_mp_compressed.pdf) lists June-July sowing and August-December harvesting; ICAR's [2025 Kharif advisory](https://icar.gov.in/sites/default/files/2025-10/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf) gives a general 15 June-15 July maize sowing window. DES says its nationwide crop calendar is based on 1998-99 and is being renewed ([DES note](https://desagri.gov.in/divisions-cell/special-data-dissemination-standards-sdds/)). Confirm district-level crop timing and prediction cutoff before making forecasts.

## 1. Environment

Use Python 3.10 or newer.

```bash
git clone https://github.com/Rishie123/DeepMaize.git
cd DeepMaize
git checkout deepmaize-kharif
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Fedora/Linux, install any required GDAL/GEOS system libraries for GeoPandas and Rasterstats before installing the Python dependencies.

For a concise definition of the CSV fields, feature order, tensor orientation, units/provenance requirements and attrition metadata, see [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md). The architecture and tensor dimensions are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 2. District boundaries

The weather and soil downloaders expect the Survey of India district boundary shapefile at:

`data/district_boundary_data/survey_of_india/DISTRICT_BOUNDARY.shp`

The expected fields are `STATE_UT` and `DISTRICT`. The code filters the uppercase state labels:

- `MAHARASHTRA`
- `MADHYA PRADESH`
- `CHHATTISGARH`

The generated join key is `STATE_DISTRICT`, with both parts uppercased, for example `MAHARASHTRA_NAGPUR`.

District names and boundaries change over time. Before joining APY yields to modern district polygons, prepare and inspect an explicit district-name / boundary-vintage crosswalk. Do not assume that string normalisation alone resolves historical district splits or renamed districts.

## 3. Weather data: NASA POWER

From the weather data directory:

```bash
cd data/weather_data
python fetch_all_districts.py
python wrangle.py --input NASA_POWER_Central_India_20000101_20241231.csv \
  --output weather_data_central_india.csv
python derive_features.py
cd ../..
```

The downloader requests the nine NASA POWER variables used in the reference feature set: `T2M_MAX`, `T2M_MIN`, `T2M`, `PRECTOTCORR`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `WS10M`, `QV2M` and `PS`. It converts the NASA `-999` missing-value sentinel to NaN.

The feature script adds `max_vpd` and `min_vpd`, giving eleven meteorological features. VPD is derived from maximum/minimum temperature and relative humidity using the saturation-vapour-pressure calculation documented in the paper.

The downloader currently requests 1 January 2000 through 31 December 2024, matching the inherited input period. Adjust this range to match actual maize yield and satellite coverage.

## 4. Soil data: SoilGrids

From the soil data directory:

```bash
cd data/soilgrids
python fetch_soil_data.py
python wrangle.py
cd ../..
```

The downloader retrieves the district mean for the eight reference-paper properties at six depths:

1. Soil organic carbon
2. Soil pH
3. Bulk density
4. Sand content
5. Silt content
6. Clay content
7. Cation-exchange capacity
8. Total nitrogen

Depth order is 0–5, 5–15, 15–30, 30–60, 60–100 and 100–200 cm. The wrangler writes a single `soil_features` matrix per district with shape `[8, 6]`, in the explicit property/depth order above. It fails on incomplete/non-finite matrices instead of silently changing the input structure.

The downloader uses the district boundary file's CRS and requests WCS layers in EPSG:4326. Check the source service's current coverage availability and units before interpreting or comparing soil values.

## 5. Vegetation features: MODIS NDVI/EVI

Prepare a CSV from the MODIS/GEE workflow with one observation per district/date and at least these columns:

```text
DISTRICT_ID,date,NDVI,EVI
```

The `DISTRICT_ID` values must use the same uppercase `STATE_DISTRICT` convention as weather and soil. Apply the appropriate MODIS quality flags and scale factors before producing the input file. This repository's monthly builder does not replace satellite quality filtering.

The monthly builder uses an **outer** date join so daily weather dates are not discarded merely because a MODIS composite is not dated on the same day:

```bash
python data/weather_data/get_maize_monthly_features.py \
  --weather data/weather_data/weather_features_central_india.csv \
  --vegetation data/remote_sensing_data/maize_ndvi_evi_central_india.csv \
  --output data/processed/maize_monthly_features.csv
```

Monthly means are calculated independently for every feature. A feature with no valid observation in a month remains missing; the dataset builder then excludes that incomplete district-season sequence. This is deliberate and should be reviewed in the data-coverage report before modelling.

The resulting default temporal input contains eleven weather features plus NDVI and EVI (thirteen features total), matching the reference paper's feature families.

## 6. Prepare Kharif maize yield labels

The yield-preparation script expects a **long-format** APY-style CSV containing:

```text
State,District,Crop,Season,Year,Yield
```

Column names can be changed using CLI options. The script filters Kharif maize for the three target states, parses the first year from values such as `2019-20`, normalises the district key, and writes:

```text
DISTRICT_ID,sowing_year,Yield
```

Example:

```bash
python data/yield_data/prepare_maize_yield.py \
  --input data/yield_data/source_apy_long.csv \
  --output data/yield_data/maize_kharif_central_india.csv
```

The script preserves the source yield unit. Use `--yield-multiplier` only when you have confirmed a unit conversion, and record the source unit and conversion in your experiment notes. The script deliberately stops when multiple Kharif maize yield rows exist for one district-year; aggregate them using a documented method rather than silently averaging.

## 6A. Agmarknet mandi prices and arrivals

Download maize mandi price and arrival series separately from the yield labels. The downloader uses the public CEDA Agri Market Data API, whose portal attributes its data to the Directorate of Marketing & Inspection (DMI):

```bash
python data/market_data/download_agmarknet.py \
  --start-date 2007-01-01 --end-date 2023-12-31 \
  --states Maharashtra "Madhya Pradesh" Chhattisgarh \
  --level district --calculation-type m \
  --output-dir data/market_data/agmarknet
```

Use `--level market` for mandi-level detail where returned by the source, or `--level state` for a quick state-level coverage check. Use `--calculation-type d` for daily, `m` for monthly and `y` for annual observations. The script resolves maize from the live commodity catalogue, queries geography metadata, rate-limits/retries requests, archives raw JSON, and writes separate price and quantity CSVs plus a request log and provenance JSON.

**Important:** Agmarknet is a source of wholesale prices and arrivals, not district crop-yield labels. The yield target still requires a suitable crop-statistics source. Keep the Agmarknet series out of the main DeepWheat-replication feature set initially; evaluate it later as a separately reported market-information extension after confirming the forecast cutoff, data coverage, district crosswalk and units. CEDA's published API terms currently restrict API use to non-commercial purposes and require attribution; check the [current terms](https://ceda.ashoka.edu.in/api-terms-conditions/) before use. Full instructions and caveats are in [docs/AGMARKNET_INGESTION.md](docs/AGMARKNET_INGESTION.md).

## 7. Build leakage-safe model datasets

Monthly input schema:

```text
DISTRICT_ID,YEAR,MONTH,<numeric feature columns>
```

Yield input schema:

```text
DISTRICT_ID,sowing_year,Yield
```

Optional soil input schema:

```text
DISTRICT_ID,soil_features
```

Build the default June-December example with the soil branch:

```bash
python src/preprocessing/build_maize_dataset.py \
  --monthly data/processed/maize_monthly_features.csv \
  --yields data/yield_data/maize_kharif_central_india.csv \
  --soil data/soilgrids/downloaded_data_wcs/district_soil_data_central_india_wrangled.csv \
  --output data/datasets/maize_kharif \
  --months 6,7,8,9,10,11,12 \
  --train-end 2018 --val-end 2021
```

For a no-soil experiment, omit the `--soil` argument.

The builder:
- requires one row per district/month and rejects duplicate keys;
- retains only district-seasons with every configured month and finite feature values;
- joins yield by exact `DISTRICT_ID` and `sowing_year`;
- creates a lag only from the immediately preceding sowing year (it does not bridge missing years);
- splits chronologically: training through 2018, validation 2019–2021, test 2022 onward by default;
- fits temporal, lagged-yield and optional soil scalers using training data only;
- writes train/validation/test CSVs, scaler objects and `metadata.json`.

Split cutoffs must be adjusted when the available maize yield years differ from the paper's wheat period. Review `metadata.json`, split sizes and district coverage before training.

## 8. Train the proposed model

From `src/rnn`:

```bash
cd src/rnn
python train_model.py --config config/maize_central_india.yaml
```

The soil-enabled config uses the CNN soil branch. For a model without soil input:

```bash
python train_model.py --config config/maize_central_india_no_soil.yaml
```

The YAML starts from the reference wheat paper's published proposed-model hyperparameters; these are **not maize-optimised**. Tune hyperparameters using the maize validation set only. The training script now selects the retained random-seed run by best validation loss, not test-set R². The test metrics are reported after validation-based checkpoint selection and are not used to choose the model.

## 9. Benchmark models

The reference study compares the proposed hybrid model with recurrent, convolutional, Transformer, parallel CNN-LSTM-attention and tree-based baselines. DeepMaize now includes maize-specific starting configurations for these benchmark families. All inherited wheat hyperparameters are starting points only; tune them on the maize validation split.

### LSTM and BiLSTM

These use the shared `CNN_RNN` trainer with temporal attention disabled. Run from `src/rnn`:

```bash
python train_model.py --config config/maize_lstm.yaml
python train_model.py --config config/maize_bilstm.yaml
```

### 1D-CNN, Transformer and parallel CNN-LSTM-attention

These baseline scripts consume the same sequence CSVs. Run from `src/baselines`:

```bash
python train_cnn1d.py --config config/cnn1d_maize_with_soil.yaml
python train_transformer.py --config config/transformer_maize_with_soil.yaml
python train_parallel.py --config config/parallel_maize_with_soil.yaml
```

Matching `*_maize_without_soil.yaml` variants are included. These baselines use their own architecture-specific soil handling; the main proposed model's soil branch is a 1D CNN over the 8 × 6 matrix.

### Random Forest, XGBoost and LightGBM

The tree trainers expect flattened numeric features. First convert the sequence splits:

```bash
python src/preprocessing/build_maize_flat_dataset.py \
  --input data/datasets/maize_kharif \
  --output data/datasets/rf_flat_maize
```

Then run HPO and the five-seed evaluation from `src/baselines`:

```bash
python train_gbm.py --model rf --soil with --config config/rf_maize.yaml
python train_gbm.py --model xgb --soil with --config config/xgb_maize.yaml
python train_gbm.py --model lgbm --soil with --config config/lgbm_maize.yaml
```

Use `--n-trials 20` for a short pipeline check before the full configured search. All three tree models tune by validation MSE; the final seed is selected by validation RMSE. Test metrics are saved for reporting only. Flattened soil columns preserve the same eight-property, six-depth ordering. The no-soil option is selected with `--soil without`.

### Baseline comparison rules

- Run every architecture on the same district-year splits and target unit.
- Keep preprocessing and feature availability consistent across models.
- Use validation data for hyperparameter search, early stopping and seed selection.
- Report test metrics only after model selection is complete.
- Include a persistence forecast `prediction = past_yield`; report it alongside the learned models because lagged yield can be a strong baseline.
- Run the stand-alone persistence report after the sequence dataset has been built:
  ```bash
  python src/analysis/maize_persistence_baseline.py
  ```
  It reverses the training-fitted lagged-yield scaler and reports test MAE, RMSE and R², including a per-sowing-year breakdown.
- Report multiple-seed mean and dispersion rather than selecting the seed with the highest test R².

## 10. Reproducibility and testing


The maize preprocessing tests can be run from the repository root:

```bash
python -m compileall -q src data tests
python -m pytest -q
```

GitHub Actions runs these checks on pushes to `deepmaize-kharif` and pull requests to the configured branches. These tests validate data-preparation mechanics using synthetic data; they do not validate actual maize yield accuracy, external data coverage or the remote download services.

For every real experiment, retain:
- the source URL/table and download date for yield, weather, vegetation and soil data;
- yield units, district crosswalk and crop-calendar source;
- the month window and forecast cutoff;
- split years, data counts and dropped district-years;
- package versions, random seeds, YAML configuration and model checkpoint;
- MAE, RMSE and R², plus a persistence baseline using lagged yield.

## 11. Important research limitations

- June-December is a starting Kharif window, not a validated universal calendar for all districts. Prediction availability must be stated relative to the last included month.
- District-level means hide within-district variation in rainfall, soils, management and crop practices.
- Historical district boundaries and yield reporting may not align with present-day polygons.
- SoilGrids is static in this pipeline; it does not represent seasonal management or changing field conditions.
- A lagged-yield feature is a strong persistence baseline. Report its performance separately so the incremental value of weather, satellite and soil information is clear.
- The reference wheat study's benchmark scripts and HPO values are not evidence of maize performance. Benchmark configurations need to be run and tuned on the same maize splits before claiming a close empirical replication.

## Reference paper

R. L., M., Punglia, S. & Joshi, S. T. (2026). A hybrid deep learning approach for winter wheat yield prediction: evidence from leveraging multi-source data. *Scientific Reports*, 16, 28538. https://doi.org/10.1038/s41598-026-56576-5

```bibtex
@article{rl2026hybrid,
  title   = {A hybrid deep learning approach for winter wheat yield prediction: evidence from leveraging multi-source data},
  author  = {R. L, Manogna and Punglia, Shaanil and Joshi, Siddhant Tushar},
  journal = {Scientific Reports},
  volume  = {16},
  number  = {1},
  pages   = {28538},
  year    = {2026},
  doi     = {10.1038/s41598-026-56576-5}
}
```

## License

MIT — see [LICENSE](LICENSE).
