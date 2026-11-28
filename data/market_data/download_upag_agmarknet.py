#!/usr/bin/env python3
"""Download Agmarknet price/arrival rows from the official UPAg Open API.

UPAg API access requires a DA&FW-issued username/password. Credentials are
read from UPAG_USERNAME and UPAG_PASSWORD environment variables.

The OpenAPI contract documents a year filter plus offset pagination (up to
10,000 rows/request). Geographic and commodity filters are applied locally
because they are not part of the documented AGMARKNETSource request schema.

Run:
    export UPAG_USERNAME='...'
    export UPAG_PASSWORD='...'
    python data/market_data/download_upag_agmarknet.py \
      --start-year 2014 --end-year 2026 \
      --states Maharashtra "Madhya Pradesh" Chhattisgarh \
      --commodity maize \
      --output-dir data/market_data/upag_agmarknet

API documentation: https://data.upag.gov.in/docs
OpenAPI schema: https://data.upag.gov.in/api/v1/openapi.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_BASE_URL = "https://data.upag.gov.in"
LOGIN_PATH = "/v1/upag/api-data-share/login"
ALLOWED_SOURCES_PATH = "/v1/upag/api-data-share/sources/user-allowed-sources"
AGMARKNET_PATH = "/v1/upag/api-data-share/sources/agmarknet"
SOURCE_NAME = "UPAg official Open API — Agmarknet"
DEFAULT_STATES = ("Maharashtra", "Madhya Pradesh", "Chhattisgarh")


def norm(value: object) -> str:
    """Comparison normalisation for state/crop labels."""
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def records_from_response(payload: Any) -> list[dict[str, Any]]:
    """Return records from the documented success response."""
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object response, got {type(payload).__name__}")
    if str(payload.get("status", "")).casefold() != "success":
        raise RuntimeError(
            f"UPAg returned status={payload.get('status')!r}: "
            f"{payload.get('message', payload.get('error', 'unknown API error'))}"
        )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("UPAg success response has no data list")
    return [row for row in rows if isinstance(row, dict)]


class UPAgClient:
    """Authenticated client for the documented UPAg Data Share API."""

    def __init__(
        self,
        username: str,
        password: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
        retries: int = 4,
        request_delay: float = 0.5,
        session: requests.Session | None = None,
    ) -> None:
        if not username or not password:
            raise ValueError(
                "UPAg credentials missing. Set UPAG_USERNAME and UPAG_PASSWORD."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "DeepMaize/1.0 (research data ingestion)",
            "Accept": "application/json",
        })
        self.username = username
        self.password = password
        self.access_token: str | None = None
        self.token_type = "Bearer"

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        """Perform an authenticated JSON request with transient-error retries."""
        url = self.base_url + path
        last_error: Exception | None = None
        for attempt in range(self.retries):
            if self.request_delay:
                time.sleep(self.request_delay)
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"Retryable UPAg HTTP {response.status_code}: "
                        f"{response.text[:500]}",
                        response=response,
                    )
                if response.status_code in (401, 403):
                    raise PermissionError(
                        f"UPAg rejected the credentials/token (HTTP {response.status_code}). "
                        "Confirm that the account is authorised for the Agmarknet data source."
                    )
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"UPAg endpoint {path} returned a non-JSON response"
                    ) from exc
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in (429, 500, 502, 503, 504):
                    raise
                if attempt + 1 < self.retries:
                    delay = min(60.0, 2.0 ** attempt * 2.0)
                    logging.warning("%s failed (%s); retrying in %.1fs", path, exc, delay)
                    time.sleep(delay)
        raise RuntimeError(f"UPAg request failed after {self.retries} attempts: {last_error}")

    def login(self) -> dict[str, Any]:
        """Obtain the OAuth2 password-grant bearer token."""
        payload = self._call(
            "POST",
            LOGIN_PATH,
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"UPAg login response has no access_token: {payload}")
        self.access_token = str(token)
        self.token_type = str(payload.get("token_type", "Bearer"))
        self.session.headers["Authorization"] = f"{self.token_type} {self.access_token}"
        return {"token_type": self.token_type}

    def allowed_sources(self) -> Any:
        """Return the data-source permissions granted to the logged-in account."""
        if not self.access_token:
            raise RuntimeError("Call login() before allowed_sources()")
        return self._call("GET", ALLOWED_SOURCES_PATH)

    def fetch_agmarknet_page(
        self, year: int, *, limit: int = 10000, offset: int = 0
    ) -> dict[str, Any]:
        """Fetch one page of the official Agmarknet source for a calendar year."""
        if not self.access_token:
            raise RuntimeError("Call login() before requesting source data")
        if not 1 <= limit <= 10000:
            raise ValueError("UPAg OpenAPI requires 1 <= limit <= 10000")
        payload = {
            "source_input_object": {
                "source_name": "agmarknet",
                "year": [str(year)],
                "limit": limit,
                "offset": offset,
            }
        }
        return self._call("POST", AGMARKNET_PATH, json=payload)


def _source_allowed(payload: Any, source_name: str = "agmarknet") -> bool | None:
    """Inspect a heterogeneous allowed-sources response without assuming its shape."""
    target = norm(source_name)
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            found.append(norm(value))
        elif isinstance(value, dict):
            for key, item in value.items():
                if norm(key) in {"sourcename", "source_name", "name", "value"} and isinstance(item, str):
                    found.append(norm(item))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    if target in found:
        return True
    if found:
        return False
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    """Case/punctuation-insensitive alias extraction from provider row."""
    index = {
        re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
        for key, value in row.items()
    }
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.casefold())
        value = index.get(key)
        if value not in (None, ""):
            return value
    return None


def canonicalise_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Preserve all UPAg provider columns and append stable canonical fields."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    alias_map = {
        "state": ("State", "StateName"),
        "state_code": ("StateCode", "StateID", "StateId"),
        "district": ("District", "DistrictName"),
        "district_id": ("DistrictCode", "DistrictID", "DistrictId", "LGDDistrictKey"),
        "market": ("Market", "MarketName", "Mandi", "MandiName", "Market_Name", "APMC"),
        "market_id": ("MarketID", "MarketId", "MandiID", "MandiId", "MarketCode"),
        "commodity": ("Crop", "crop", "Commodity", "CommodityName"),
        "commodity_id": ("CropCode", "CropKey", "CommodityCode", "CommodityID"),
        "variety": ("Variety", "VarietyName", "CommodityVariety"),
        "grade": ("Grade", "GradeName"),
        "date": ("CalendarDay", "Date", "ArrivalDate", "TradeDate"),
        "min_price": ("MandiMinimumPrice", "MinimumPrice", "MinPrice"),
        "max_price": ("MandiMaximumPrice", "MaximumPrice", "MaxPrice"),
        "modal_price": ("MandiModalPrice", "ModalPrice", "Modal"),
        "arrival_qty": ("MandiArrivalQuantity", "ArrivalQuantity", "Arrivals", "Quantity"),
        "price_uom": ("PriceUOM", "PriceUnit", "PriceUoM"),
        "quantity_uom": ("QuantityUOM", "QuantityUnit", "QuantityUoM"),
    }
    for canonical, aliases in alias_map.items():
        if canonical not in frame.columns:
            source = next(
                (column for column in frame.columns if norm(column).replace(" ", "") in
                 {norm(alias).replace(" ", "") for alias in aliases}),
                None,
            )
            if source is not None:
                frame[canonical] = frame[source]
    if "date" not in frame.columns or "modal_price" not in frame.columns:
        raise ValueError(
            "UPAg Agmarknet response does not contain the required date/modal-price fields. "
            f"Observed columns: {list(frame.columns)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("min_price", "max_price", "modal_price", "arrival_qty"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _filter_selected(
    frame: pd.DataFrame,
    states: list[str] | None,
    commodity_pattern: str | None,
) -> pd.DataFrame:
    """Apply state/crop selection locally; the documented API only accepts year."""
    result = frame.copy()
    if states and "state" in result.columns:
        allowed = {norm(state) for state in states}
        result = result[result["state"].map(norm).isin(allowed)]
    if commodity_pattern:
        if "commodity" not in result.columns:
            raise ValueError(
                "UPAg response has no Crop/Commodity field; cannot locally filter maize."
            )
        result = result[result["commodity"].map(norm).str.contains(
            re.escape(norm(commodity_pattern)), na=False
        )]
    return result.reset_index(drop=True)


def download(
    client: UPAgClient,
    *,
    start_year: int,
    end_year: int,
    output_dir: Path,
    states: list[str] | None = None,
    commodity: str = "maize",
    limit: int = 10000,
    page_delay: float = 0.5,
) -> dict[str, Any]:
    """Download all pages/year-by-year, then create a local selected-commodity CSV."""
    if start_year > end_year:
        raise ValueError("start-year must be <= end-year")
    if limit > 10000 or limit < 1:
        raise ValueError("limit must be between 1 and 10,000")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    login_meta = client.login()
    allowed = client.allowed_sources()
    _write_json(raw_dir / "allowed_sources.json", allowed)
    permission = _source_allowed(allowed)
    if permission is False:
        raise PermissionError(
            "The authenticated UPAg account does not list 'agmarknet' among its allowed sources."
        )

    all_rows: list[dict[str, Any]] = []
    page_log: list[dict[str, Any]] = []
    yearly_counts: dict[str, int] = {}
    for year in range(start_year, end_year + 1):
        offset = 0
        year_count = 0
        while True:
            response = client.fetch_agmarknet_page(year, limit=limit, offset=offset)
            _write_json(raw_dir / f"agmarknet_{year}_offset_{offset}.json", response)
            rows = records_from_response(response)
            total = int(response.get("totalRecords", len(rows)))
            all_rows.extend(rows)
            year_count += len(rows)
            page_log.append({
                "year": year, "offset": offset, "limit": limit,
                "total_records_reported": total,
                "records_received": len(rows),
                "status": "ok",
            })
            logging.info("UPAg Agmarknet %s offset %s: %s / %s records",
                         year, offset, len(rows), total)
            if not rows or offset + len(rows) >= total:
                break
            offset += len(rows)
            if page_delay:
                time.sleep(page_delay)
        yearly_counts[str(year)] = year_count

    all_frame = canonicalise_rows(all_rows)
    # Save an unfiltered, canonical copy as well as the selected modelling file.
    all_path = output_dir / "upag_agmarknet_all_canonical.csv"
    all_frame.to_csv(all_path, index=False)
    selected = _filter_selected(all_frame, states, commodity)
    selected_path = output_dir / "upag_agmarknet_selected.csv"
    selected.to_csv(selected_path, index=False)
    pd.DataFrame(page_log).to_csv(output_dir / "download_log.csv", index=False)

    date_min = None
    date_max = None
    if "date" in selected.columns and selected["date"].notna().any():
        date_min = str(selected["date"].min())
        date_max = str(selected["date"].max())
    metadata = {
        "project": "DeepMaize",
        "source": SOURCE_NAME,
        "api_base_url": client.base_url,
        "source_name": "agmarknet",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authentication": "UPAg OAuth2 password grant; credentials not saved",
        "allowed_source_permission": permission,
        "token_type": login_meta.get("token_type"),
        "requested_calendar_years": list(range(start_year, end_year + 1)),
        "states_selected_locally": states,
        "commodity_filter_selected_locally": commodity,
        "page_limit": limit,
        "rows_by_year": yearly_counts,
        "all_rows": int(len(all_frame)),
        "selected_rows": int(len(selected)),
        "selected_date_min": date_min,
        "selected_date_max": date_max,
        "selected_fields": list(selected.columns),
        "geographic_fields_present": {
            "state": "state" in selected.columns,
            "district": "district" in selected.columns,
            "market": "market" in selected.columns,
            "market_id": "market_id" in selected.columns,
        },
        "files": {
            "all_canonical": str(all_path),
            "selected": str(selected_path),
            "log": str(output_dir / "download_log.csv"),
            "raw_responses": str(raw_dir),
        },
        "important": (
            "The official source request schema documents year, limit and offset; "
            "state and commodity filters are applied locally. Check the returned fields "
            "to establish whether records are market-, district- or state-level."
        ),
    }
    _write_json(output_dir / "download_metadata.json", metadata)
    if selected.empty:
        raise RuntimeError(
            "The UPAg download returned no rows after local state/commodity filtering. "
            "Inspect upag_agmarknet_all_canonical.csv and raw/ to confirm crop labels and coverage."
        )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--commodity", default="maize")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--page-delay", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/market_data/upag_agmarknet"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    username = os.environ.get("UPAG_USERNAME")
    password = os.environ.get("UPAG_PASSWORD")
    try:
        client = UPAgClient(
            username or "", password or "", base_url=args.base_url,
            timeout=args.timeout, retries=args.retries,
        )
        summary = download(
            client,
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=args.output_dir,
            states=args.states,
            commodity=args.commodity,
            limit=args.limit,
            page_delay=args.page_delay,
        )
    except Exception as exc:
        logging.error("UPAg Agmarknet download failed: %s", exc)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
