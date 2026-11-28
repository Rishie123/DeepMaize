#!/usr/bin/env python3
"""Download Agmarknet-derived mandi prices and arrivals from CEDA.

The public CEDA Agri Market Data API republishes data sourced from the
Directorate of Marketing & Inspection (DMI), Ministry of Agriculture and
Farmers Welfare. This downloader keeps price and arrival responses separate,
archives unmodified JSON responses, and writes normalised CSV files.

Important: Agmarknet provides wholesale market prices and arrivals. It does
not provide the district crop-yield labels required by DeepMaize.
CEDA's published API terms currently restrict use to non-commercial purposes
and require attribution. Check the current terms before each new use.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

BASE_URL = "https://agmarknet.ceda.ashoka.edu.in/api"
SOURCE_NAME = "CEDA Agri Market Data (Agmarknet; source: DMI, Government of India)"
TARGET_STATES = ("Chhattisgarh", "Madhya Pradesh", "Maharashtra")
STATE_ID_FALLBACK = {
    "Chhattisgarh": 22,
    "Madhya Pradesh": 23,
    "Maharashtra": 27,
}
COMMODITY_ALIASES = {
    "maize": {"maize", "corn", "maize (corn)", "maize grain"},
}


def _normalise_text(value: object) -> str:
    """Return a case-insensitive comparison key for provider labels."""
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _slug(value: object) -> str:
    """Make a safe, readable filename fragment."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_").lower()


def _records(payload: Any) -> list[dict[str, Any]]:
    """Extract record lists from common JSON response envelopes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "results", "items", "states",
                    "districts", "commodities", "markets"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    return []


def _first_value(row: dict[str, Any], names: Iterable[str]) -> Any:
    """Read the first non-empty matching field, ignoring case/punctuation."""
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


def _api_timestamp(value: str, *, end_of_day: bool = False) -> str:
    """Convert YYYY-MM-DD into the UTC ISO string expected by the CEDA API."""
    parsed = datetime.strptime(value, "%Y-%m-%d")
    suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
    return parsed.strftime("%Y-%m-%d") + suffix


def _normalise_response_records(
    payload: Any,
    *,
    record_type: str,
    state: str,
    state_id: int,
    district: str,
    district_id: int,
    commodity: str,
    commodity_id: int,
) -> list[dict[str, Any]]:
    """Map CEDA fields into a stable schema while retaining raw JSON elsewhere."""
    output: list[dict[str, Any]] = []
    for row in _records(payload):
        output.append({
            "source": SOURCE_NAME,
            "record_type": record_type,
            "state": _first_value(row, ("state", "state_name")) or state,
            "state_id": state_id,
            "district": _first_value(row, ("district", "district_name", "dist")) or district,
            "district_id": district_id,
            "market": _first_value(row, ("market", "market_name", "mkt", "marketname", "market_nm")),
            "market_id": _first_value(row, ("market_id", "marketid", "mkt_id")),
            "commodity": _first_value(row, ("cmdty", "commodity", "commodity_name")) or commodity,
            "commodity_id": commodity_id,
            "variety": _first_value(row, ("variety", "variety_name")),
            "date": _first_value(row, ("t", "date", "arrival_date", "period")),
            "min_price": _first_value(row, ("p_min", "min_price", "minimum_price")),
            "max_price": _first_value(row, ("p_max", "max_price", "maximum_price")),
            "modal_price": _first_value(row, ("p_modal", "modal_price", "modal")),
            "arrival_qty": _first_value(row, ("arrival_qty", "arrivals", "quantity", "qty", "q")),
        })
    return output


class CEDAAgmarknetClient:
    """Throttled HTTP client for the public CEDA Agri Market Data API."""

    def __init__(
        self,
        *,
        timeout: int = 60,
        retries: int = 4,
        request_delay: float = 1.5,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "DeepMaize/1.0 (research data ingestion)",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://agmarknet.ceda.ashoka.edu.in",
            "Referer": "https://agmarknet.ceda.ashoka.edu.in/",
        })

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        """Request JSON with retry/backoff; never silently bypass CAPTCHA."""
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            if self.request_delay:
                time.sleep(self.request_delay)
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Retryable HTTP {response.status_code}: {response.text[:300]}",
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in (429, 500, 502, 503, 504):
                    raise
                if attempt + 1 < self.retries:
                    wait = min(60.0, 2.0 ** attempt * 2.0)
                    logging.warning("%s failed (%s); retrying in %.1fs", endpoint, exc, wait)
                    time.sleep(wait)
            except ValueError as exc:
                raise RuntimeError(
                    f"{url} returned non-JSON data. The API may have changed."
                ) from exc
        raise RuntimeError(f"Request failed after {self.retries} attempts: {last_error}")

    def get_metadata(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch a CEDA metadata endpoint and extract its record list."""
        payload = self._request("GET", endpoint, params=params or {})
        rows = _records(payload)
        if not rows:
            raise RuntimeError(
                f"No metadata records found at {endpoint}; response keys="
                f"{list(payload) if isinstance(payload, dict) else type(payload).__name__}"
            )
        return rows

    def commodities(self) -> list[dict[str, Any]]:
        return self.get_metadata("commodities")

    def states(self) -> list[dict[str, Any]]:
        return self.get_metadata("states")

    def districts(self, state_id: int) -> list[dict[str, Any]]:
        return self.get_metadata("districts", params={"state_id": state_id})

    def fetch_series(
        self, *, kind: str, state_id: int, district_id: int,
        commodity_id: int, start_date: str, end_date: str,
        calculation_type: str, chart_type: str,
    ) -> Any:
        """Fetch one price or arrival series for one geographic selection."""
        if kind not in {"prices", "quantities"}:
            raise ValueError("kind must be 'prices' or 'quantities'")
        payload = {
            "state_id": state_id,
            "commodity_id": commodity_id,
            "district_id": district_id,
            "calculation_type": calculation_type,
            "chart_type": chart_type,
            "start_date": _api_timestamp(start_date),
            "end_date": _api_timestamp(end_date, end_of_day=True),
        }
        return self._request("POST", kind, json=payload)


def _metadata_name_id(
    rows: list[dict[str, Any]], *, wanted: str,
    name_fields: tuple[str, ...], id_fields: tuple[str, ...],
) -> tuple[str, int] | None:
    """Resolve a provider name to a numeric API identifier."""
    target = _normalise_text(wanted)
    for row in rows:
        name = _first_value(row, name_fields)
        identifier = _first_value(row, id_fields)
        if name is not None and identifier is not None and _normalise_text(name) == target:
            return str(name), int(identifier)
    return None


def resolve_commodity(
    rows: list[dict[str, Any]], requested: str = "maize",
    commodity_id_override: int | None = None,
) -> tuple[str, int]:
    """Resolve a crop from the live commodity catalogue; reject ambiguity."""
    name_fields = ("commodity", "commodity_name", "cmdty", "name", "title", "label")
    id_fields = ("commodity_id", "commodityid", "id", "value")
    if commodity_id_override is not None:
        match = next((
            row for row in rows
            if _first_value(row, id_fields) is not None
            and int(_first_value(row, id_fields)) == commodity_id_override
        ), None)
        if match is None:
            raise ValueError(f"Commodity ID {commodity_id_override} was not found in the live catalogue")
        return str(_first_value(match, name_fields) or requested), commodity_id_override

    aliases = COMMODITY_ALIASES.get(_normalise_text(requested), {_normalise_text(requested)})
    matches = []
    for row in rows:
        name = _first_value(row, name_fields)
        identifier = _first_value(row, id_fields)
        if name is not None and identifier is not None and _normalise_text(name) in aliases:
            matches.append((str(name), int(identifier)))
    unique = sorted(set(matches), key=lambda item: (item[0].casefold(), item[1]))
    if len(unique) != 1:
        candidates = [
            (str(_first_value(row, name_fields)), _first_value(row, id_fields))
            for row in rows
            if any(alias in _normalise_text(_first_value(row, name_fields)) for alias in aliases)
        ]
        raise ValueError(
            f"Expected exactly one catalogue match for {requested!r}; "
            f"found {unique or candidates}. Use --commodity-id after checking the CEDA catalogue."
        )
    return unique[0]


def resolve_state(rows: list[dict[str, Any]], state: str) -> tuple[str, int]:
    """Resolve state metadata, using Census 2011 fallback IDs if needed."""
    result = _metadata_name_id(
        rows, wanted=state,
        name_fields=("state", "state_name", "name", "state_ut"),
        id_fields=("state_id", "stateid", "id", "census_id", "value"),
    )
    if result:
        return result
    if state in STATE_ID_FALLBACK:
        return state, STATE_ID_FALLBACK[state]
    raise ValueError(f"State {state!r} was not found in CEDA metadata")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download(
    *,
    client: CEDAAgmarknetClient,
    output_dir: Path,
    start_date: str,
    end_date: str,
    states: Iterable[str],
    level: str = "district",
    calculation_type: str = "m",
    commodity: str = "maize",
    commodity_id: int | None = None,
) -> dict[str, Any]:
    """Download price and arrival series for the requested states/geographies.

    level=state uses district_id=0; level=district requests each district;
    level=market requests each district using the datadownload chart view.
    """
    if date.fromisoformat(start_date) > date.fromisoformat(end_date):
        raise ValueError("start-date must be on or before end-date")
    if level not in {"state", "district", "market"}:
        raise ValueError("level must be state, district, or market")
    if calculation_type not in {"d", "m", "y"}:
        raise ValueError("calculation-type must be d (daily), m (monthly), or y (yearly)")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    state_list = list(states)

    commodity_rows = client.commodities()
    _save_json(raw_dir / "metadata_commodities.json", commodity_rows)
    commodity_name, commodity_identifier = resolve_commodity(
        commodity_rows, commodity, commodity_id
    )
    state_rows = client.states()
    _save_json(raw_dir / "metadata_states.json", state_rows)

    prices: list[dict[str, Any]] = []
    quantities: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    chart_type = "datadownload" if level == "market" else "timeseries"

    for requested_state in state_list:
        state_name, state_identifier = resolve_state(state_rows, requested_state)
        if level == "state":
            geography = [("ALL_DISTRICTS", 0)]
        else:
            district_rows = client.districts(state_identifier)
            _save_json(raw_dir / f"metadata_districts_{_slug(state_name)}.json", district_rows)
            geography = []
            for row in district_rows:
                district_name = _first_value(
                    row, ("district", "district_name", "name", "districtname", "label")
                )
                district_identifier = _first_value(
                    row, ("district_id", "districtid", "id", "census_id", "value")
                )
                if district_name is not None and district_identifier is not None:
                    geography.append((str(district_name), int(district_identifier)))
            if not geography:
                raise RuntimeError(f"CEDA returned no usable district IDs for {state_name}")

        for district_name, district_identifier in geography:
            for kind in ("prices", "quantities"):
                request_key = (
                    f"{_slug(commodity_name)}_{_slug(state_name)}_"
                    f"{_slug(district_name)}_{kind}"
                )
                try:
                    payload = client.fetch_series(
                        kind=kind, state_id=state_identifier,
                        district_id=district_identifier,
                        commodity_id=commodity_identifier,
                        start_date=start_date, end_date=end_date,
                        calculation_type=calculation_type, chart_type=chart_type,
                    )
                    _save_json(raw_dir / f"{request_key}.json", payload)
                    normalised = _normalise_response_records(
                        payload, record_type=kind, state=state_name,
                        state_id=state_identifier, district=district_name,
                        district_id=district_identifier, commodity=commodity_name,
                        commodity_id=commodity_identifier,
                    )
                    (prices if kind == "prices" else quantities).extend(normalised)
                    log_rows.append({
                        "state": state_name, "district": district_name,
                        "record_type": kind, "status": "ok",
                        "records": len(normalised), "error": "",
                    })
                    logging.info("%s / %s / %s: %s records",
                                 state_name, district_name, kind, len(normalised))
                except Exception as exc:
                    log_rows.append({
                        "state": state_name, "district": district_name,
                        "record_type": kind, "status": "error",
                        "records": 0, "error": str(exc),
                    })
                    logging.error("%s / %s / %s failed: %s",
                                  state_name, district_name, kind, exc)

    columns = [
        "source", "record_type", "state", "state_id", "district",
        "district_id", "market", "market_id", "commodity", "commodity_id",
        "variety", "date", "min_price", "max_price", "modal_price", "arrival_qty",
    ]
    price_frame = pd.DataFrame(prices, columns=columns)
    quantity_frame = pd.DataFrame(quantities, columns=columns)
    for frame in (price_frame, quantity_frame):
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            for col in ("min_price", "max_price", "modal_price", "arrival_qty"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    price_path = output_dir / "agmarknet_prices.csv"
    quantity_path = output_dir / "agmarknet_quantities.csv"
    log_path = output_dir / "download_log.csv"
    price_frame.to_csv(price_path, index=False)
    quantity_frame.to_csv(quantity_path, index=False)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    failed = sum(row["status"] == "error" for row in log_rows)
    summary = {
        "source": SOURCE_NAME,
        "source_url": "https://agmarknet.ceda.ashoka.edu.in/",
        "downloaded_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "commodity": commodity_name,
        "commodity_id": commodity_identifier,
        "start_date": start_date,
        "end_date": end_date,
        "level": level,
        "calculation_type": calculation_type,
        "chart_type": chart_type,
        "states": state_list,
        "price_records": len(price_frame),
        "quantity_records": len(quantity_frame),
        "failed_requests": failed,
        "files": {
            "prices": str(price_path),
            "quantities": str(quantity_path),
            "request_log": str(log_path),
            "raw_json_directory": str(raw_dir),
        },
        "important": (
            "Agmarknet prices/arrivals are market indicators, not district crop-yield labels. "
            "CEDA API terms currently restrict use to non-commercial purposes; verify current terms."
        ),
    }
    (output_dir / "download_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not prices and not quantities:
        raise RuntimeError(
            "No price or arrival records were returned. Check dates, commodity ID, "
            "geographies and current CEDA API availability."
        )
    if failed:
        logging.warning("%s requests failed. Partial outputs saved; inspect %s", failed, log_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2007-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2023-12-31", help="YYYY-MM-DD")
    parser.add_argument("--states", nargs="+", default=list(TARGET_STATES),
                        choices=list(TARGET_STATES))
    parser.add_argument("--level", choices=("state", "district", "market"), default="district")
    parser.add_argument("--calculation-type", choices=("d", "m", "y"), default="m",
                        help="d=daily, m=monthly, y=yearly")
    parser.add_argument("--commodity", default="maize")
    parser.add_argument("--commodity-id", type=int, default=None,
                        help="Override only after checking the live CEDA commodity catalogue")
    parser.add_argument("--output-dir", type=Path, default=Path("data/market_data/agmarknet"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-delay", type=float, default=1.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    client = CEDAAgmarknetClient(
        timeout=args.timeout, retries=args.retries, request_delay=args.request_delay
    )
    try:
        summary = download(
            client=client, output_dir=args.output_dir,
            start_date=args.start_date, end_date=args.end_date,
            states=args.states, level=args.level,
            calculation_type=args.calculation_type,
            commodity=args.commodity, commodity_id=args.commodity_id,
        )
    except Exception as exc:
        logging.error("Agmarknet download stopped: %s", exc)
        return 1
    print(json.dumps(summary, indent=2))
    return 2 if summary["failed_requests"] else 0


if __name__ == "__main__":
    sys.exit(main())
