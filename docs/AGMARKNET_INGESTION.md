# Agmarknet/CEDA ingestion for price forecasting

## Data source and API

DeepMaize retrieves Agmarknet-derived market quotations and arrival quantities through CEDA Agri Market Data. The CEDA portal attributes this information to the Directorate of Marketing & Inspection (DMI), Ministry of Agriculture and Farmers Welfare, Government of India.

Official references:

- API reference: https://api.ceda.ashoka.edu.in/documentation/
- Data portal: https://agmarknet.ceda.ashoka.edu.in/
- API terms: https://ceda.ashoka.edu.in/api-terms-conditions/
- Government of India OGD record: https://data.gov.in/resource/variety-wise-daily-market-prices-data-commodity

The active project needs market-level price records, not a state or district aggregate. The downloader uses the CEDA market-download chart selection for each district and retains returned market IDs/names.

## Terms of use

CEDA's published API terms currently permit downloading, displaying or including the data in other products for **non-commercial purposes** at no cost, subject to citation and visual attribution. Required citation:

    CEDA Agri Market Data (CEDA-AMD), 2000-2023.
    Centre for Economic Data & Analysis, Ashoka University.

The terms also require the CEDA logo and a source credit in visualisations/applications. Do not deploy these data in a commercial price product or business service without explicit permission or a separately licensed source. Verify the current terms before using the API.

## Download

A short smoke test:

~~~bash
python data/market_data/download_agmarknet.py \
  --start-date 2020-01-01 --end-date 2020-12-31 \
  --states Maharashtra --level market --calculation-type d \
  --output-dir data/market_data/agmarknet_smoke
~~~

Full Central India extraction:

~~~bash
python data/market_data/download_agmarknet.py \
  --start-date 2007-01-01 --end-date 2023-12-31 \
  --states Maharashtra "Madhya Pradesh" Chhattisgarh \
  --level market --calculation-type d \
  --output-dir data/market_data/agmarknet
~~~

The default commodity is maize. Commodity IDs are resolved from the live CEDA catalogue. State and district catalogues are queried to retrieve API identifiers.

The script supports:
- level=state: one state-level aggregate.
- level=district: district-level aggregate.
- level=market: market-level download view; required for this project.
- calculation-type d|m|y: daily, monthly or yearly calculation period.

Daily market-level data is preferred for the canonical extraction because it preserves the raw reported quotation dates. The next step aggregates it to monthly observations. A monthly CEDA pull is useful for a quick trial, but confirm how that endpoint represents markets and periods before combining it with daily files.

## Output files

- agmarknet_prices.csv: normalised price records.
- agmarknet_quantities.csv: normalised arrivals.
- download_log.csv: endpoint result and errors by state/district.
- download_metadata.json: query details, output counts and extraction time.
- raw/: unmodified JSON data and lookup catalogues.

The normalised rows include state/district, market and IDs when returned, commodity, variety, date, min/max/modal prices and arrivals. Retain the raw JSON if the provider adds fields or changes schema.

## Convert to monthly panel

~~~bash
python src/prices/prepare_price_panel.py \
  --prices data/market_data/agmarknet/agmarknet_prices.csv \
  --quantities data/market_data/agmarknet/agmarknet_quantities.csv \
  --output data/processed/maize_market_monthly.csv
~~~

The active monthly target is the median of daily modal prices within each mandi × variety × month. Minimum and maximum price series are retained as historical features. Market × commodity monthly arrivals are summed from source records and joined to variety-level price rows; they can repeat across varieties, so do not sum them across variety rows.

The preprocessor requires a market ID or a market label. If the downloaded prices contain only district-level totals, rerun at level=market; do not treat district aggregates as market prices.

## Coverage checks before modelling

Inspect:
- earliest and latest observation dates;
- number of markets and varieties by state;
- modal-price missingness and non-positive values;
- violations of min ≤ modal ≤ max where all bounds are present;
- monthly coverage and length of consecutive histories;
- quantity coverage and its reporting unit;
- market IDs and names that change over time.

The dataset builder excludes windows that do not have all 12 consecutive preceding months. It does not fill missing prices.

## Caution

CEDA states that data accuracy is not warranted. Verify unusual observations with the source and report exclusions. A month-level prediction is not an auction-day quote; it should be displayed with the latest input month and the target month, not labelled live.
