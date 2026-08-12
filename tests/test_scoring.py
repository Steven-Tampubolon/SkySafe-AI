"""
tests/test_scoring.py — Unit tests for scoring.py: G-scale/R-scale mapping,
geomagnetic latitude adjustment, and confidence logic. All pure functions,
no network calls.
"""

import pytest

from scoring.scoring import (
    score_gps_impact,
    get_g_scale,
    score_hf_risk,
    get_r_scale,
    compute_confidence,
    geomagnetic_latitude,
    classify_latitude_band,
    ScoringError,
)


class TestScoreGpsImpactHighLatitude:
    """High-Latitude/Auroral = original NOAA table calibration, factor 1.0."""

    @pytest.mark.parametrize("kp,expected_label", [
        (0, "Low"), (4.99, "Low"),
        (5.0, "Moderate"), (6.0, "Moderate"), (6.99, "Moderate"),
        (7.0, "High"), (8.99, "High"),
        (9.0, "Critical"),
    ])
    def test_labels(self, kp, expected_label):
        _, label = score_gps_impact(kp, "High-Latitude/Auroral")
        assert label == expected_label

    def test_kp9_scores_100(self):
        score, label = score_gps_impact(9.0, "High-Latitude/Auroral")
        assert score == 100.0
        assert label == "Critical"

    def test_kp0_scores_0(self):
        score, _ = score_gps_impact(0, "High-Latitude/Auroral")
        assert score == 0.0


class TestScoreGpsImpactLatitudeAdjustment:
    """The same Kp should score LOWER at lower geomagnetic latitude — the
    core fix for the global-audience revision."""

    def test_same_kp_lower_score_at_lower_latitude(self):
        high, _ = score_gps_impact(7.0, "High-Latitude/Auroral")
        mid, _ = score_gps_impact(7.0, "Mid-Latitude")
        low, _ = score_gps_impact(7.0, "Equatorial/Low")
        assert high > mid > low

    def test_extreme_storm_not_critical_at_equator(self):
        """Kp=9 (G5 Extreme) is 'Critical' at high latitude but must NOT be
        reported as Critical for a farmer near the equator — this is
        exactly the over-estimation bug this revision fixes."""
        score, label = score_gps_impact(9.0, "Equatorial/Low")
        assert label != "Critical"
        assert score == pytest.approx(40.0, abs=0.1)

    def test_mid_latitude_factor(self):
        score, _ = score_gps_impact(9.0, "Mid-Latitude")
        assert score == pytest.approx(70.0, abs=0.1)

    def test_unknown_band_raises(self):
        with pytest.raises(ScoringError):
            score_gps_impact(6.0, "Somewhere Unknown")

    def test_out_of_range_kp_raises(self):
        with pytest.raises(ScoringError):
            score_gps_impact(-1, "High-Latitude/Auroral")
        with pytest.raises(ScoringError):
            score_gps_impact(9.5, "High-Latitude/Auroral")


class TestGeomagneticLatitude:
    """Sanity checks against known reference points. Dipole approximation —
    tolerance is generous on purpose."""

    def test_fargo_is_high_latitude(self):
        # Matches the worked example in the v2 prompt template doc.
        assert classify_latitude_band(46.9, -96.8) == "High-Latitude/Auroral"

    def test_jakarta_is_equatorial(self):
        assert classify_latitude_band(-6.2, 106.8) == "Equatorial/Low"

    def test_nairobi_is_equatorial(self):
        assert classify_latitude_band(-1.3, 36.8) == "Equatorial/Low"

    def test_geomagnetic_latitude_differs_from_geographic(self):
        # Same geographic latitude, different longitude -> different
        # geomagnetic latitude, because the magnetic pole is offset.
        gmlat_a = geomagnetic_latitude(50.0, -100.0)
        gmlat_b = geomagnetic_latitude(50.0, 100.0)
        assert abs(gmlat_a - gmlat_b) > 10


class TestScoreHfRisk:
    """R-scale is NOT latitude-adjusted (see module docstring)."""

    @pytest.mark.parametrize("flare_class,expected_label", [
        (None, "Low"), ("None", "Low"), ("C9.9", "Low"),
        ("M1.0", "Moderate"), ("M4.9", "Moderate"),
        ("M5.0", "Moderate"), ("X0.9", "Moderate"),
        ("X1.0", "High"), ("X19.9", "High"),
        ("X20.0", "Critical"),
    ])
    def test_labels(self, flare_class, expected_label):
        assert score_hf_risk(flare_class) == expected_label

    def test_invalid_format_raises(self):
        with pytest.raises(ScoringError):
            score_hf_risk("Z9.9")


class TestComputeConfidence:
    def test_realtime_is_high(self):
        level, reason = compute_confidence("real-time")
        assert level == "High"
        assert reason

    def test_forecast_under_24h_is_medium(self):
        level, _ = compute_confidence("forecast", forecast_horizon_hours=12)
        assert level == "Medium"

    def test_forecast_24h_plus_is_low(self):
        level, _ = compute_confidence("forecast", forecast_horizon_hours=24)
        assert level == "Low"

    def test_invalid_data_type_raises(self):
        with pytest.raises(ScoringError):
            compute_confidence("invalid_type")


class TestValidatesAgainstMay2024Storm:
    """
    Validation against the May 10-11, 2024 geomagnetic storm ("Gannon
    storm"), Kp=9 (G5 Extreme) — strongest in ~20 years, documented to have
    disrupted GPS-guided precision agriculture across North America.

    This is exactly where the latitude-adjustment revision matters: the
    storm WAS genuinely critical for high-latitude farmers, but would be
    over-stated as equally critical for farmers near the equator without
    this fix.

    Source: https://www.swpc.noaa.gov/news/g5-extreme-geomagnetic-storm-conditions-observed
    """

    def test_critical_at_high_latitude(self):
        score, label = score_gps_impact(9, "High-Latitude/Auroral")
        assert label == "Critical"
        assert score == 100.0

    def test_not_critical_at_equator(self):
        score, label = score_gps_impact(9, "Equatorial/Low")
        assert label != "Critical"
        assert score < 50