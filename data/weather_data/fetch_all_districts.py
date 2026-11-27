import geopandas as gpd
import requests
import pandas as pd
from shapely.geometry import Point
from io import StringIO
import time
import os
from datetime import datetime
from tqdm import tqdm


def read_shapefile(filepath):
    """Read district polygons and report the available schema."""
    try:
        gdf = gpd.read_file(filepath)
        print(f"Successfully loaded shapefile with {len(gdf)} records")
        print(f"Columns: {list(gdf.columns)}")
        return gdf
    except Exception as exc:
        print(f"Error reading shapefile: {exc}")
        return None


def filter_districts_by_states(gdf, target_states):
    """Filter the boundary layer using uppercase STATE_UT values."""
    if "STATE_UT" not in gdf.columns:
        raise ValueError("District boundary file must contain STATE_UT")
    states = {state.strip().upper() for state in target_states}
    filtered = gdf[gdf["STATE_UT"].astype(str).str.upper().isin(states)].copy()
    if filtered.empty:
        available = sorted(gdf["STATE_UT"].dropna().astype(str).unique())
        raise ValueError(
            f"No districts found for {sorted(states)}. Available state values: {available}"
        )
    return filtered


def get_centroid_coordinates(gdf_row, crs):
    """Return a district centroid in WGS84; project before centroid calculation."""
    geometry = gdf_row.geometry
    if crs is None:
        raise ValueError("District boundary file has no CRS")
    projected_crs = "EPSG:6933"
    centroid = gpd.GeoSeries([geometry], crs=crs).to_crs(projected_crs).centroid
    centroid_wgs84 = centroid.to_crs("EPSG:4326").iloc[0]
    return float(centroid_wgs84.x), float(centroid_wgs84.y)


def fetch_nasa_power_data(
    longitude, latitude, start_date, end_date, parameters,
    max_retries=3, retry_delay=5
):
    """Fetch NASA POWER daily data with bounded retries and request timeouts."""
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "CSV",
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                print(f"NASA POWER request failed after {max_retries} attempts: {exc}")
                return None
            wait = retry_delay * (2 ** attempt)
            print(f"NASA POWER request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    return None


def process_nasa_csv_data(csv_data):
    """Remove NASA POWER's metadata header and parse YYYY/DOY into a date."""
    lines = csv_data.strip().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "-END HEADER-") + 1
    except StopIteration:
        raise ValueError("NASA POWER response has no -END HEADER- marker")
    payload = "\n".join(line for line in lines[start:] if line.strip())
    frame = pd.read_csv(StringIO(payload))
    # NASA POWER uses -999 as a missing-value sentinel; never treat it as weather.
    frame = frame.replace(-999, float("nan"))
    if not {"YEAR", "DOY"}.issubset(frame.columns):
        raise ValueError("NASA POWER CSV must contain YEAR and DOY")
    frame["Date"] = pd.to_datetime(
        frame["YEAR"].astype(str) + "-" + frame["DOY"].astype(str),
        format="%Y-%j", errors="raise"
    )
    return frame.drop(columns=["YEAR", "DOY"])


def fetch_weather_for_district(
    gdf_row, gdf_crs, start_date, end_date, parameters,
    max_retries=3, retry_delay=5
):
    """Fetch daily weather for one district centroid and attach stable labels."""
    state_name = str(gdf_row["STATE_UT"]).strip().upper()
    district_name = str(gdf_row["DISTRICT"]).strip().upper()
    try:
        longitude, latitude = get_centroid_coordinates(gdf_row, gdf_crs)
        response = fetch_nasa_power_data(
            longitude, latitude, start_date, end_date, parameters,
            max_retries, retry_delay
        )
        if response is None:
            return None
        frame = process_nasa_csv_data(response)
        frame["State"] = state_name
        frame["District"] = district_name
        frame["DISTRICT_ID"] = state_name + "_" + district_name
        cols = ["DISTRICT_ID", "State", "District", "Date"] + [
            col for col in frame.columns
            if col not in {"DISTRICT_ID", "State", "District", "Date"}
        ]
        return frame[cols]
    except Exception as exc:
        print(f"Failed {state_name}/{district_name}: {exc}")
        return None


def main():
    """Download NASA POWER weather inputs for the three Central Indian states."""
    shapefile_path = "../district_boundary_data/survey_of_india/DISTRICT_BOUNDARY.shp"
    target_states = ["MAHARASHTRA", "MADHYA PRADESH", "CHHATTISGARH"]
    start_date, end_date = "20000101", "20241231"
    parameters = [
        "T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "RH2M",
        "ALLSKY_SFC_SW_DWN", "WS10M", "QV2M", "PS",
    ]
    request_delay, max_retries, retry_delay = 2, 3, 5

    print("DeepMaize NASA POWER weather downloader")
    print(f"States: {', '.join(target_states)}")
    print(f"Dates: {start_date} to {end_date}")

    gdf = read_shapefile(shapefile_path)
    if gdf is None:
        raise RuntimeError(f"Could not read district boundary file: {shapefile_path}")
    filtered = filter_districts_by_states(gdf, target_states)

    # The original shapefile's geometry is passed to centroid calculation;
    # no district polygons are modified or written back.
    all_data = []
    failures = []
    for _, district_row in tqdm(filtered.iterrows(), total=len(filtered), desc="Districts"):
        frame = fetch_weather_for_district(
            district_row, filtered.crs, start_date, end_date, parameters,
            max_retries, retry_delay
        )
        if frame is None:
            failures.append(str(district_row.get("DISTRICT", "unknown")))
        else:
            all_data.append(frame)
        time.sleep(request_delay)

    if not all_data:
        raise RuntimeError("No district weather data was retrieved")
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["DISTRICT_ID", "Date"]).reset_index(drop=True)
    output = "NASA_POWER_Central_India_20000101_20241231.csv"
    combined.to_csv(output, index=False)
    print(f"Saved {len(combined):,} daily records to {output}")
    print(f"Districts retrieved: {len(all_data)} / {len(filtered)}")
    if failures:
        print(f"Failed districts ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
