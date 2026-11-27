import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import os
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

SHAPEFILE_PATH = "../district_boundary_data/survey_of_india/DISTRICT_BOUNDARY.shp"
STATE_COLUMN = "STATE_UT"
DISTRICT_COLUMN = "DISTRICT"

# Target region for the DeepMaize Kharif maize experiment.
TARGET_STATES = ["MAHARASHTRA", "MADHYA PRADESH", "CHHATTISGARH"]
OUTPUT_DIR = "downloaded_data_wcs"
OUTPUT_FILENAME = "district_soil_data_central_india.csv"

# Keep this property order aligned with the reference paper's Appendix A.
SOIL_FEATURES = {
    "Soil Organic Carbon": "soc",
    "Soil pH": "phh2o",
    "Bulk Density": "bdod",
    "Sand Content": "sand",
    "Silt Content": "silt",
    "Clay Content": "clay",
    "Cation Exchange Capacity": "cec",
    "Total Nitrogen": "nitrogen",
}
DEPTH_INTERVALS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

MAX_WORKERS_FEATURES = 2
REQUEST_DELAY = 0.5
MAX_RETRIES = 3
CONNECTION_TIMEOUT = 60


def create_robust_session():
    """Create a requests session with bounded retries for transient HTTP errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_soil_feature(args):
    """Download one SoilGrids property/depth layer and compute its district mean."""
    (district_name, state_name, district_geom, feature_name, service_id,
     depth, minx, miny, maxx, maxy, output_dir) = args
    time.sleep(REQUEST_DELAY)
    session = create_robust_session()
    safe_state = state_name.replace(" ", "_")
    safe_district = district_name.replace(" ", "_")
    temp_path = os.path.join(
        output_dir, f"{safe_state}_{safe_district}_{service_id}_{depth}.tif"
    )
    base_url = "https://maps.isric.org/mapserv"
    coverage_id = f"{service_id}_{depth}_mean"
    params = {
        "map": f"/map/{service_id}.map",
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": coverage_id,
        "crs": "EPSG:4326",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": max(int(round((maxx - minx) / 0.002083333333333)), 2),
        "height": max(int(round((maxy - miny) / 0.002083333333333)), 2),
        "format": "GEOTIFF_INT16",
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(base_url, params=params, timeout=CONNECTION_TIMEOUT)
            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                handle.write(response.content)
            stats = zonal_stats(district_geom, temp_path, stats="mean", geojson_out=False)
            value = stats[0].get("mean") if stats and stats[0] else None
            return {
                "state": state_name,
                "district": district_name,
                "feature_name": feature_name,
                "depth": depth,
                "value": value if value is not None else float("nan"),
                "success": value is not None,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            if attempt == MAX_RETRIES:
                return {
                    "state": state_name, "district": district_name,
                    "feature_name": feature_name, "depth": depth,
                    "value": float("nan"), "success": False,
                    "error": str(exc), "attempts": attempt + 1,
                }
            time.sleep((2 ** attempt) * REQUEST_DELAY)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def main():
    """Download SoilGrids 8 × 6 district matrices for the target states."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    india_gdf = gpd.read_file(SHAPEFILE_PATH)
    if india_gdf.crs is None:
        raise ValueError("District shapefile has no CRS")
    if india_gdf.crs.to_epsg() != 4326:
        india_gdf = india_gdf.to_crs(epsg=4326)

    roi = india_gdf[
        india_gdf[STATE_COLUMN].astype(str).str.upper().isin(TARGET_STATES)
    ].copy()
    if roi.empty:
        raise ValueError(
            f"No districts found for {TARGET_STATES}; check {STATE_COLUMN} in the shapefile"
        )

    tasks = []
    for _, row in roi.iterrows():
        district, state, geometry = row[DISTRICT_COLUMN], row[STATE_COLUMN], row.geometry
        minx, miny, maxx, maxy = geometry.bounds
        for feature_name, service_id in SOIL_FEATURES.items():
            for depth in DEPTH_INTERVALS:
                tasks.append((
                    district, state, geometry, feature_name, service_id,
                    depth, minx, miny, maxx, maxy, OUTPUT_DIR
                ))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FEATURES) as executor:
        futures = [executor.submit(download_soil_feature, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="SoilGrids"):
            results.append(future.result())

    result_frame = pd.DataFrame(results)
    result_frame["district_id"] = (
        result_frame["state"].astype(str).str.upper()
        + "_" + result_frame["district"].astype(str).str.strip()
    )
    result_frame = result_frame.rename(columns={"district_id": "DISTRICT_ID"})
    columns = ["DISTRICT_ID", "feature_name", "depth", "value", "success"]
    result_frame[columns].to_csv(
        os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), index=False
    )
    failed = int((~result_frame["success"]).sum())
    print(f"Saved {len(result_frame)} property-depth records; failed: {failed}")


if __name__ == "__main__":
    main()
