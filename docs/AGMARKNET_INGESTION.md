# Agmarknet price and arrival ingestion

## What this source provides

DeepMaize can download Agmarknet-derived daily mandi wholesale price and arrival series through CEDA Agri Market Data. Its portal attributes the source data to the Directorate of Marketing & Inspection (DMI), Ministry of Agriculture and Farmers Welfare.

This is **not the maize yield-label source**. Agmarknet records market observations (minimum, maximum and modal prices, and arrival quantities). The DeepMaize target Yield remains a district × Kharif-sowing-year yield observation from the appropriate crop-statistics source. Do not relabel a price or arrival field as crop yield.

CEDA documentation: https://api.ceda.ashoka.edu.in/documentation/  
Portal: https://agmarknet.ceda.ashoka.edu.in/  
API terms: https://ceda.ashoka.edu.in/api-terms-conditions/

**Licence/terms:** CEDA's published API terms currently state that API data may be used for non-commercial purposes at no cost, subject to attribution. They specify the citation “CEDA Agri Market Data (CEDA-AMD), 2000-2023”, Centre for Economic Data & Analysis, Ashoka University. Check the live terms before use, especially before commercial or client-facing deployment.

## Install and run

The downloader uses the existing requests and pandas dependencies.

From the repository root:

    python data/market_data/download_agmarknet.py \
      --start-date 2007-01-01 \
      --end-date 2023-12-31 \
      --states Maharashtra "Madhya Pradesh" Chhattisgarh \
      --level district \
      --calculation-type m \
      --output-dir data/market_data/agmarknet

The default commodity is maize. The downloader queries CEDA's live commodity catalogue and resolves the exact commodity ID by name. It also queries state and district catalogues instead of assuming district IDs are stable across releases.

Choose the granularity deliberately:

| Option | API selection | Use |
|---|---|---|
| state | One aggregate selection per state (district_id=0) | Quick coverage check |
| district | One selection per district | District-level market series |
| market | One selection per district using chart_type=datadownload | Mandi-level detail where the API returns it |

The calculation interval is d (daily), m (monthly) or y (yearly). Monthly is the sensible first pass for district × year modelling; daily/market-level downloads can be substantially larger. The actual observations returned determine the data coverage.

A short smoke test for a single state and month:

    python data/market_data/download_agmarknet.py \
      --start-date 2020-06-01 --end-date 2020-06-30 \
      --states Maharashtra --level district --calculation-type m \
      --output-dir data/market_data/agmarknet_smoke

The downloader waits 1.5 seconds before requests and retries transient connection, rate-limit and server errors with exponential backoff. It does not try to bypass CAPTCHA or authentication gates. A partially successful run saves its outputs and a download_log.csv; the process exits with code 2 so failures are visible.

## Outputs

| File | Contents |
|---|---|
| agmarknet_prices.csv | Normalised minimum, maximum and modal wholesale prices |
| agmarknet_quantities.csv | Normalised arrival quantity observations |
| download_log.csv | One success/failure row per geographic selection and endpoint |
| download_metadata.json | Source, extraction time, filters, record counts and output paths |
| raw/*.json | Unmodified API responses plus downloaded catalogue metadata |

Both normalised CSVs retain state/district and numeric IDs, market/market ID when supplied, commodity, variety, date, and common price/arrival columns. For the price table, arrival_qty will normally be empty; for the quantity table, price columns will normally be empty. Raw JSON is retained to preserve provider fields not included in the normalised schema.

The source series is not guaranteed to be complete or current to today. Inspect the minimum and maximum date, row counts and missingness; treat the latest observation date as the source timestamp rather than assuming a live quote. Preserve download metadata with every modelling experiment.

## Methodological treatment in DeepMaize

The main paper-replication configuration continues to use the reference meteorological/vegetation sequence, static soil matrix and previous-year yield. Do not silently add market data to that baseline: it changes the feature set and can make comparison with the paper unfair.

After the download is validated, Agmarknet may be assessed as a separate market-information extension. Before joining it to DeepMaize, create and inspect a state-qualified district crosswalk between CEDA names and Survey of India boundary names. Historical district splits/renames must be resolved explicitly. Decide whether market price/arrival values are information available by the forecast cutoff; avoid using post-harvest prices or arrivals to predict the same season's yield.

Price units are expected to follow source conventions (often ₹/quintal), but inspect the specific API output and provider metadata rather than hard-coding a unit assumption. Arrival quantity units may vary by commodity/report; preserve source units and verify them before aggregation. Do not sum market-level varieties or duplicate rows without checking data grain.

## Tests

    python -m pytest -q tests/test_agmarknet_download.py

These tests use mocked API responses. They validate parsing, commodity resolution, provenance outputs and separate price/quantity files; they do not establish that the live CEDA service is currently reachable or that every target district has maize observations.
