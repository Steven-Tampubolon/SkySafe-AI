"""
tests/test_geocoding.py — Unit tests for geocoding.py, all mocked.
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from ingestion.geocoding import geocode_location, GeocodingError

MOCK_RESPONSE = {
    "results": [
        {"name": "Nairobi", "latitude": -1.28333, "longitude": 36.81667,
         "admin1": "Nairobi County", "country": "Kenya"}
    ]
}


class TestGeocodeLocation:
    @patch("ingestion.geocoding.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = geocode_location("Nairobi, Kenya")

        assert result["latitude"] == pytest.approx(-1.28333)
        assert result["longitude"] == pytest.approx(36.81667)
        assert "Nairobi" in result["resolved_name"]
        assert "Kenya" in result["resolved_name"]

    @patch("ingestion.geocoding.requests.get")
    def test_not_found_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with pytest.raises(GeocodingError):
            geocode_location("Xyzzyplonkistan")

    @patch("ingestion.geocoding.requests.get")
    def test_api_down_raises(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        with pytest.raises(GeocodingError):
            geocode_location("Jakarta")

    def test_empty_location_raises(self):
        with pytest.raises(GeocodingError):
            geocode_location("   ")