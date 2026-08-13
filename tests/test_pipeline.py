"""
tests/test_pipeline.py — Unit tests for pipeline.build_trust_panel_input().
Pure composition over already-tested scoring functions, no mocking needed.
"""

from pipeline import build_trust_panel_input

SAMPLE_RESOLVED_LOCATION = {
    "latitude": 64.84,
    "longitude": -147.72,
    "resolved_name": "Fairbanks, Alaska, United States",
}

SAMPLE_CONDITIONS = {
    "kp_index": 7,
    "kp_forecast": [],
    "flare_class": "M4.1",
    "flare_time": "2026-08-17T10:00:00Z",
    "fetched_at": "2026-08-17T12:00:00+00:00",
}


class TestBuildTrustPanelInput:
    def test_high_latitude_produces_high_band(self):
        result = build_trust_panel_input("farmer", SAMPLE_RESOLVED_LOCATION, SAMPLE_CONDITIONS)
        assert result["geomagnetic_latitude_band"] == "High-Latitude/Auroral"
        assert result["role"] == "farmer"
        assert result["kp_index"] == 7

    def test_real_time_data_gives_high_confidence(self):
        result = build_trust_panel_input("surveyor", SAMPLE_RESOLVED_LOCATION, SAMPLE_CONDITIONS)
        assert result["confidence_level"] == "High"

    def test_cached_data_treated_as_forecast(self):
        cached_conditions = dict(SAMPLE_CONDITIONS, _from_cache=True)
        result = build_trust_panel_input("farmer", SAMPLE_RESOLVED_LOCATION, cached_conditions)
        assert result["data_type"] == "forecast"

    def test_equatorial_location_scores_lower_than_high_latitude(self):
        equatorial_location = {"latitude": -1.28, "longitude": 36.82, "resolved_name": "Nairobi, Kenya"}
        high_lat = build_trust_panel_input("farmer", SAMPLE_RESOLVED_LOCATION, SAMPLE_CONDITIONS)
        equatorial = build_trust_panel_input("farmer", equatorial_location, SAMPLE_CONDITIONS)
        assert high_lat["gps_impact_score"] > equatorial["gps_impact_score"]