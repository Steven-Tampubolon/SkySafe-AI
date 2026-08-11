"""
tests/test_ingestion.py — Unit test untuk ingestion.py, semua pakai mock response.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from ingestion.ingestion import (
    fetch_kp_index,
    fetch_flare_data,
    get_current_conditions,
    IngestionError,
    _parse_kp_forecast_text,
)

MOCK_KP_REALTIME_RESPONSE = [
    {"time_tag": "2026-08-10T09:00:00", "Kp": 3.0, "a_running": 10, "station_count": 8},
    {"time_tag": "2026-08-10T12:00:00", "Kp": 6.0, "a_running": 20, "station_count": 8},
]

MOCK_KP_FORECAST_TEXT = """\
:Product: 3-Day Forecast
:Issued: 2026 Aug 10 2200 UTC
# Some header text

NOAA Kp index breakdown Aug 11-Aug 13 2026

             Aug 11       Aug 12       Aug 13
00-03UT       2.33         2.00         3.33     
06-09UT       2.33         5.00 (G1)    2.67     
"""

MOCK_DONKI_RESPONSE = [
    {
        "flrID": "2026-08-09T10:00:00-FLR-001",
        "classType": "M4.1",
        "beginTime": "2026-08-09T09:50:00Z",
        "peakTime": "2026-08-09T10:00:00Z",
    },
    {
        "flrID": "2026-08-08T05:00:00-FLR-001",
        "classType": "C2.0",
        "beginTime": "2026-08-08T04:50:00Z",
        "peakTime": "2026-08-08T05:00:00Z",
    },
]


class TestFetchKpIndex:
    @patch("ingestion.ingestion.requests.get")
    def test_fetch_kp_realtime_success(self, mock_get):
        mock_realtime = MagicMock()
        mock_realtime.json.return_value = MOCK_KP_REALTIME_RESPONSE
        mock_realtime.raise_for_status.return_value = None

        mock_forecast = MagicMock()
        mock_forecast.text = MOCK_KP_FORECAST_TEXT
        mock_forecast.raise_for_status.return_value = None

        # urutan pemanggilan: realtime dulu, lalu forecast
        mock_get.side_effect = [mock_realtime, mock_forecast]

        result = fetch_kp_index()

        assert result["kp_index"] == 6  # ambil baris TERAKHIR
        assert isinstance(result["kp_forecast"], list)
        assert len(result["kp_forecast"]) == 6  # 2 baris x 3 kolom tanggal

    @patch("ingestion.ingestion.requests.get")
    def test_fetch_kp_realtime_api_down_raises(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("timeout")

        with pytest.raises(IngestionError):
            fetch_kp_index()

    @patch("ingestion.ingestion.requests.get")
    def test_forecast_parse_failure_returns_empty_list_not_crash(self, mock_get):
        mock_realtime = MagicMock()
        mock_realtime.json.return_value = MOCK_KP_REALTIME_RESPONSE
        mock_realtime.raise_for_status.return_value = None

        mock_forecast = MagicMock()
        mock_forecast.text = "format tidak dikenal sama sekali!!"
        mock_forecast.raise_for_status.return_value = None

        mock_get.side_effect = [mock_realtime, mock_forecast]

        result = fetch_kp_index()
        assert result["kp_index"] == 6
        assert result["kp_forecast"] == []  # gagal parse -> kosong, bukan crash


class TestParseForecastText:
    def test_parse_valid_text(self):
        forecast = _parse_kp_forecast_text(MOCK_KP_FORECAST_TEXT)
        assert len(forecast) == 6
        assert forecast[0]["kp"] == 2.33
        assert "T00:00" in forecast[0]["time"]

    def test_parse_garbage_text_returns_empty(self):
        forecast = _parse_kp_forecast_text("tidak ada data valid di sini")
        assert forecast == []


class TestFetchFlareData:
    @patch("ingestion.ingestion.requests.get")
    @patch("ingestion.ingestion.os.getenv", return_value="fake_key")
    def test_fetch_flare_success_returns_latest(self, mock_getenv, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_DONKI_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = fetch_flare_data()
        assert result["flare_class"] == "M4.1"
        assert result["flare_time"] == "2026-08-09T10:00:00Z"

    @patch("ingestion.ingestion.requests.get")
    @patch("ingestion.ingestion.os.getenv", return_value="fake_key")
    def test_fetch_flare_empty_response(self, mock_getenv, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = fetch_flare_data()
        assert result["flare_class"] is None
        assert result["flare_time"] is None

    @patch("ingestion.ingestion.os.getenv", return_value=None)
    def test_fetch_flare_no_api_key_raises(self, mock_getenv):
        with pytest.raises(IngestionError):
            fetch_flare_data()


class TestGetCurrentConditions:
    @patch("ingestion.ingestion.fetch_flare_data")
    @patch("ingestion.ingestion.fetch_kp_index")
    def test_combines_schema_correctly(self, mock_kp, mock_flare, tmp_path, monkeypatch):
        import ingestion.ingestion as ing
        monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ing, "CACHE_FILE", tmp_path / "last_conditions.json")

        mock_kp.return_value = {
            "kp_index": 6,
            "kp_forecast": [{"time": "2026-08-10T00:00:00+00:00", "kp": 6.0}],
        }
        mock_flare.return_value = {"flare_class": "M4.1", "flare_time": "2026-08-09T10:00:00Z"}

        result = get_current_conditions()

        assert result["kp_index"] == 6
        assert result["flare_class"] == "M4.1"
        assert "fetched_at" in result
        assert set(result.keys()) >= {
            "kp_index", "kp_forecast", "flare_class", "flare_time", "fetched_at"
        }

    @patch("ingestion.ingestion.fetch_flare_data")
    @patch("ingestion.ingestion.fetch_kp_index")
    def test_no_flare_becomes_tidak_ada(self, mock_kp, mock_flare, tmp_path, monkeypatch):
        import ingestion.ingestion as ing
        monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ing, "CACHE_FILE", tmp_path / "last_conditions.json")

        mock_kp.return_value = {"kp_index": 2, "kp_forecast": []}
        mock_flare.return_value = {"flare_class": None, "flare_time": None}

        result = get_current_conditions()
        assert result["flare_class"] == "Tidak ada"

    @patch("ingestion.ingestion.fetch_flare_data")
    @patch("ingestion.ingestion.fetch_kp_index")
    def test_falls_back_to_cache_when_all_fail(self, mock_kp, mock_flare, tmp_path, monkeypatch):
        import ingestion.ingestion as ing
        monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
        cache_file = tmp_path / "last_conditions.json"
        monkeypatch.setattr(ing, "CACHE_FILE", cache_file)

        cached_data = {
            "kp_index": 4, "kp_forecast": [], "flare_class": "Tidak ada",
            "flare_time": None, "fetched_at": "2026-08-09T00:00:00+00:00",
        }
        cache_file.write_text(json.dumps(cached_data))

        mock_kp.side_effect = IngestionError("API down")

        result = get_current_conditions()
        assert result["kp_index"] == 4
        assert result["_from_cache"] is True

    @patch("ingestion.ingestion.fetch_flare_data")
    @patch("ingestion.ingestion.fetch_kp_index")
    def test_raises_when_no_cache_and_api_down(self, mock_kp, mock_flare, tmp_path, monkeypatch):
        import ingestion.ingestion as ing
        monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(ing, "CACHE_FILE", tmp_path / "nonexistent.json")

        mock_kp.side_effect = IngestionError("API down")

        with pytest.raises(IngestionError):
            get_current_conditions()