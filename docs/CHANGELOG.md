# Changelog

## DeepMaize adaptation (initial scaffold)

- Added a separate maize/Kharif dataset builder; original wheat preprocessing is unchanged.
- Added Central India configuration for Maharashtra, Madhya Pradesh and Chhattisgarh.
- Added methodology notes covering model architecture, crop-calendar assumptions, data requirements and leakage controls.
- Added unit tests for month parsing, sequence construction, incomplete months and duplicate records.
- The model architecture is reused from the existing `CNN_RNN` implementation; its hyperparameters are a starting point, not maize-optimised values.
- No raw maize dataset is included and no model training or empirical validation is claimed.

## Follow-up implementation hardening

- Corrected pandas object-column assignment for nested temporal sequences and soil matrices.
- Corrected split-specific soil scaling bookkeeping and added train/validation/test integration coverage.
- Added an architecture note documenting actual BiLSTM, self-attention, soil-CNN, lagged-yield and MLP tensor dimensions.
- Corrected the methodology document to the 2018 / 2019–2021 / 2022+ default chronological split.
- Removed a legacy manual NASA POWER network test from pytest auto-discovery; preserved it as a manual check script.
- Changed CNN1D, Transformer and parallel CNN-LSTM baseline seed selection to use validation loss rather than held-out test R².

- Added maize LSTM and BiLSTM no-attention configurations, plus neural and tree baseline configurations.
- Added a sequence-to-flat converter for Random Forest, XGBoost and LightGBM.
- Updated tree-based multi-seed selection to use validation RMSE; test results remain reporting-only.
- Added a stand-alone persistence baseline report that inverse-transforms the lagged-yield feature.
