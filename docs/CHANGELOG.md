# Changelog

## Market-price forecasting refactor

- Changed the active research target from annual Kharif maize district yield to monthly mandi-level maize modal price.
- Defined the prediction unit as market × commodity × variety × month and the initial forecast horizon as one calendar month.
- Added a monthly panel builder that aggregates daily modal-price records, keeps price and arrival series distinct, and preserves market identity.
- Added leakage-safe rolling 12-month price windows with chronological train/validation/test splits and training-only feature scaling.
- Added a pooled LSTM with market-series embeddings and price-unit persistence/seasonal-naive baselines.
- Added market-price metrics and prediction output files.
- Rewrote the main project, methodology, architecture, ingestion and data dictionary documentation around price forecasting.
- Kept older yield, weather and soil modules in the repository as legacy reference code; they are not used by the active forecasting workflow.
- CEDA API terms and source credit requirements are documented; commercial use requires permission or a separately licensed data source.

## Previous yield adaptation (legacy)

The original branch work adapted DeepWheat's yield modelling pipeline to Kharif maize. Those files remain for research history but no longer define the active DeepMaize task.
