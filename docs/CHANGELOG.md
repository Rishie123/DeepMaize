# Changelog

## DeepMaize adaptation (initial scaffold)

- Added a separate maize/Kharif dataset builder; original wheat preprocessing is unchanged.
- Added Central India configuration for Maharashtra, Madhya Pradesh and Chhattisgarh.
- Added methodology notes covering model architecture, crop-calendar assumptions, data requirements and leakage controls.
- Added unit tests for month parsing, sequence construction, incomplete months and duplicate records.
- The model architecture is reused from the existing `CNN_RNN` implementation; its hyperparameters are a starting point, not maize-optimised values.
- No raw maize dataset is included and no model training or empirical validation is claimed.
