"""Unit tests for the Agmarknet/CEDA downloader without live network access."""
from pathlib import Path

from data.market_data.download_agmarknet import (
    CEDAAgmarknetClient,
    _normalise_response_records,
    resolve_commodity,
    resolve_state,
    download,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/commodities"):
            return FakeResponse({"data": [
                {"commodity_id": 44, "commodity_name": "Maize"},
                {"commodity_id": 78, "commodity_name": "Tomato"},
            ]})
        if url.endswith("/states"):
            return FakeResponse({"data": [
                {"state_id": 27, "state_name": "Maharashtra"},
            ]})
        if url.endswith("/districts"):
            return FakeResponse({"data": [
                {"district_id": 501, "district_name": "Nagpur"},
            ]})
        if url.endswith("/prices"):
            return FakeResponse({"data": [{
                "cmdty": "Maize", "t": "2020-06-01",
                "p_min": 1500, "p_max": 1800, "p_modal": 1650,
                "market": "Nagpur APMC",
            }]})
        if url.endswith("/quantities"):
            return FakeResponse({"data": [{
                "cmdty": "Maize", "t": "2020-06-01",
                "q": 120, "market": "Nagpur APMC",
            }]})
        raise AssertionError(f"Unexpected URL {url}")


def test_commodity_and_state_resolution_are_exact():
    commodities = [
        {"commodity_id": 44, "commodity_name": "Maize"},
        {"commodity_id": 45, "commodity_name": "Maize Seed"},
    ]
    states = [{"state_id": 27, "state_name": "Maharashtra"}]
    assert resolve_commodity(commodities, "maize") == ("Maize", 44)
    assert resolve_state(states, "Maharashtra") == ("Maharashtra", 27)


def test_ambiguous_commodity_resolution_fails_loudly():
    commodities = [
        {"commodity_id": 44, "commodity_name": "Maize"},
        {"commodity_id": 45, "commodity_name": "Corn"},
    ]
    try:
        resolve_commodity(commodities, "maize")
    except ValueError as exc:
        assert "Expected exactly one" in str(exc)
    else:
        raise AssertionError("Expected ambiguous aliases to be rejected")


def test_price_and_quantity_field_normalisation():
    payload = {"data": [{
        "cmdty": "Maize", "t": "2020-06-01",
        "p_min": "1500", "p_max": "1800", "p_modal": "1650",
        "market": "Nagpur APMC",
    }]}
    rows = _normalise_response_records(
        payload, record_type="prices", state="Maharashtra", state_id=27,
        district="Nagpur", district_id=501, commodity="Maize", commodity_id=44,
    )
    assert rows[0]["date"] == "2020-06-01"
    assert rows[0]["modal_price"] == "1650"
    assert rows[0]["market"] == "Nagpur APMC"
    assert rows[0]["district_id"] == 501


def test_download_writes_separate_price_quantity_and_provenance_files(tmp_path: Path):
    session = FakeSession()
    client = CEDAAgmarknetClient(session=session, request_delay=0, retries=1)
    summary = download(
        client=client, output_dir=tmp_path,
        start_date="2020-06-01", end_date="2020-06-30",
        states=["Maharashtra"], level="district", calculation_type="m",
    )
    assert summary["price_records"] == 1
    assert summary["quantity_records"] == 1
    assert summary["failed_requests"] == 0
    assert (tmp_path / "agmarknet_prices.csv").exists()
    assert (tmp_path / "agmarknet_quantities.csv").exists()
    assert (tmp_path / "download_log.csv").exists()
    assert (tmp_path / "download_metadata.json").exists()
    raw_payloads = list((tmp_path / "raw").glob("*.json"))
    assert len(raw_payloads) == 5
    price_call = next(call for call in session.calls if call[1].endswith("/prices"))
    assert price_call[2]["json"]["district_id"] == 501
    assert price_call[2]["json"]["calculation_type"] == "m"
    assert price_call[2]["json"]["chart_type"] == "timeseries"
