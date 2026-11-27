"""Integration tests for the DeepMaize data-preparation path."""
from pathlib import Path
import ast

import numpy as np
import pandas as pd

from src.preprocessing.build_maize_dataset import build_dataset
from data.weather_data.get_maize_monthly_features import make_monthly_features
from data.yield_data.prepare_maize_yield import prepare_yield
from src.preprocessing.build_maize_flat_dataset import build_flat_datasets


def test_build_dataset_creates_chronological_splits_and_scalers(tmp_path: Path):
    monthly_rows = []
    yield_rows = []
    districts = ["MAHARASHTRA_NAGPUR", "MADHYA PRADESH_CHHINDWARA"]
    for district_index, district in enumerate(districts):
        for year in range(2015, 2023):
            yield_rows.append({
                "DISTRICT_ID": district, "sowing_year": year,
                "Yield": 4000 + 100 * district_index + (year - 2015) * 20,
            })
            for month in [6, 7, 8]:
                monthly_rows.append({
                    "DISTRICT_ID": district, "YEAR": year, "MONTH": month,
                    "rain": float(year + month),
                    "ndvi": float(0.2 + 0.01 * (month - 6) + 0.001 * district_index),
                })

    monthly_path = tmp_path / "monthly.csv"
    yields_path = tmp_path / "yields.csv"
    out_dir = tmp_path / "dataset"
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    pd.DataFrame(yield_rows).to_csv(yields_path, index=False)

    metadata = build_dataset(
        monthly_path, yields_path, out_dir, [6, 7, 8],
        train_end=2017, val_end=2019
    )

    assert metadata["split_years"]["train"] == [2016, 2017]
    assert metadata["split_years"]["validation"] == [2018, 2019]
    assert metadata["split_years"]["test"] == [2020, 2021, 2022]
    assert (out_dir / "temporal_scaler.joblib").exists()
    assert (out_dir / "past_yield_scaler.joblib").exists()
    train = pd.read_csv(out_dir / "train.csv")
    test = pd.read_csv(out_dir / "test.csv")
    assert train["feature_sequence"].map(lambda x: len(ast.literal_eval(x))).eq(3).all()
    assert set(test["sowing_year"]) == {2020, 2021, 2022}


def test_soil_matrices_are_written_in_model_input_shape(tmp_path: Path):
    monthly_rows, yield_rows, soil_rows = [], [], []
    districts = ["MAHARASHTRA_NAGPUR", "MADHYA PRADESH_CHHINDWARA"]
    for district_index, district in enumerate(districts):
        matrix = [
            [float((property_index + 1) * (depth_index + 1) + district_index)
             for depth_index in range(6)]
            for property_index in range(8)
        ]
        soil_rows.append({"DISTRICT_ID": district, "soil_features": str(matrix)})
        for year in range(2015, 2022):
            yield_rows.append({
                "DISTRICT_ID": district, "sowing_year": year,
                "Yield": 4200 + district_index * 100 + (year - 2015) * 10,
            })
            for month in [6, 7, 8]:
                monthly_rows.append({
                    "DISTRICT_ID": district, "YEAR": year, "MONTH": month,
                    "rain": float(year + month),
                })

    monthly_path, yield_path, soil_path = (
        tmp_path / "monthly.csv", tmp_path / "yields.csv", tmp_path / "soil.csv"
    )
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    pd.DataFrame(yield_rows).to_csv(yield_path, index=False)
    pd.DataFrame(soil_rows).to_csv(soil_path, index=False)

    out_dir = tmp_path / "with_soil"
    metadata = build_dataset(
        monthly_path, yield_path, out_dir, [6, 7, 8],
        train_end=2017, val_end=2019, soil_path=soil_path
    )

    train = pd.read_csv(out_dir / "train.csv")
    matrix = np.asarray(ast.literal_eval(train.iloc[0]["soil_features"]), dtype=float)
    assert matrix.shape == (8, 6)
    assert metadata["soil_matrix_shape"] == [8, 6]
    assert (out_dir / "soil_scaler.joblib").exists()


def test_monthly_weather_vegetation_outer_join_preserves_unsynchronised_dates(tmp_path: Path):
    weather_path = tmp_path / "weather.csv"
    vegetation_path = tmp_path / "vegetation.csv"
    output_path = tmp_path / "monthly.csv"
    pd.DataFrame([
        {"DISTRICT_ID": "MAHARASHTRA_NAGPUR", "DATE": "2020-06-01", "rain": 10},
        {"DISTRICT_ID": "MAHARASHTRA_NAGPUR", "DATE": "2020-06-02", "rain": 20},
    ]).to_csv(weather_path, index=False)
    pd.DataFrame([
        {"DISTRICT_ID": "MAHARASHTRA_NAGPUR", "date": "2020-06-01", "NDVI": 0.5, "EVI": 0.3},
    ]).to_csv(vegetation_path, index=False)

    monthly = make_monthly_features(weather_path, vegetation_path, output_path)
    assert len(monthly) == 1
    assert monthly.loc[0, "rain"] == 15
    assert monthly.loc[0, "NDVI"] == 0.5


def test_yield_preparation_filters_kharif_maize_and_normalises_ids(tmp_path: Path):
    source = tmp_path / "source_yields.csv"
    output = tmp_path / "maize_yields.csv"
    pd.DataFrame([
        {"State": "Maharashtra", "District": "Nagpur", "Crop": "Maize",
         "Season": "Kharif", "Year": "2018-19", "Yield": 5000},
        {"State": "Madhya Pradesh", "District": "Chhindwara", "Crop": "Maize",
         "Season": "Kharif", "Year": "2019-20", "Yield": 5100},
        {"State": "Chhattisgarh", "District": "Bilaspur", "Crop": "Maize",
         "Season": "Rabi", "Year": "2018-19", "Yield": 5200},
        {"State": "Punjab", "District": "Ludhiana", "Crop": "Maize",
         "Season": "Kharif", "Year": "2018-19", "Yield": 5300},
    ]).to_csv(source, index=False)

    result = prepare_yield(source, output)
    assert result["DISTRICT_ID"].tolist() == [
        "MADHYA PRADESH_CHHINDWARA", "MAHARASHTRA_NAGPUR"
    ]
    assert result["sowing_year"].tolist() == [2019, 2018]


def test_flattened_tree_dataset_preserves_splits_and_feature_values(tmp_path: Path):
    sequence_dir = tmp_path / "sequence"
    sequence_dir.mkdir()
    metadata = {"feature_columns": ["rain", "NDVI"]}
    (sequence_dir / "metadata.json").write_text(__import__("json").dumps(metadata))
    for split, year in [("train", 2018), ("val", 2019), ("test", 2020)]:
        pd.DataFrame([{
            "DISTRICT_ID": "MAHARASHTRA_NAGPUR",
            "sowing_year": year,
            "feature_sequence": str([[1.0, 0.2], [2.0, 0.3]]),
            "past_yield": 0.5,
            "Yield": 5000.0,
        }]).to_csv(sequence_dir / f"{split}.csv", index=False)

    output_dir = tmp_path / "flat"
    summary = build_flat_datasets(sequence_dir, output_dir)
    train = pd.read_csv(output_dir / "train_rf_flat.csv")
    assert summary["split_rows"] == {"train": 1, "val": 1, "test": 1}
    assert train.loc[0, "temporal_m01_rain"] == 1.0
    assert train.loc[0, "temporal_m02_NDVI"] == 0.3
    assert train.loc[0, "past_yield"] == 0.5
    assert train.loc[0, "yield_value"] == 5000.0
    assert train.loc[0, "district"] == "MAHARASHTRA_NAGPUR"
