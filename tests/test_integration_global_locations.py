"""
tests/test_integration_global_locations.py — Full-pipeline integration test
across 3 locations representing 3 distinct geomagnetic latitude bands, per
Week 2 sprint requirement (Reykjavik, Madrid, Nairobi).

kp_index and flare_class are fixed identically across all 3 locations so
any difference in gps_impact_score/label is attributable ONLY to the
geomagnetic latitude adjustment — this is the core proof-of-concept for
the "global-aware" claim in the pitch.

Coordinates are hardcoded (not geocoded live) so this test is deterministic
and network-free, safe to run repeatedly without hitting rate limits.
"""

from pipeline import build_trust_panel_input

LOCATIONS = {
    "Reykjavik, Iceland": {
        "latitude": 64.1466, "longitude": -21.9426,
        "resolved_name": "Reykjavik, Iceland",
        "expected_band": "High-Latitude/Auroral",
    },
    "Madrid, Spain": {
        "latitude": 40.4168, "longitude": -3.7038,
        "resolved_name": "Madrid, Spain",
        "expected_band": "Mid-Latitude",
    },
    "Nairobi, Kenya": {
        "latitude": -1.2921, "longitude": 36.8219,
        "resolved_name": "Nairobi, Kenya",
        "expected_band": "Equatorial/Low",
    },
}

# Identical simulated storm conditions for all 3 locations.
FIXED_CONDITIONS = {
    "kp_index": 7,
    "kp_forecast": [],
    "flare_class": "M4.1",
    "flare_time": "2026-08-22T10:00:00Z",
    "fetched_at": "2026-08-22T12:00:00+00:00",
}


class TestThreeLocationsSameStorm:
    def test_each_location_classified_into_expected_band(self):
        for name, loc in LOCATIONS.items():
            result = build_trust_panel_input("farmer", loc, FIXED_CONDITIONS)
            assert result["geomagnetic_latitude_band"] == loc["expected_band"], (
                f"{name} expected {loc['expected_band']}, "
                f"got {result['geomagnetic_latitude_band']}"
            )

    def test_gps_impact_score_decreases_toward_equator(self):
        reykjavik = build_trust_panel_input("farmer", LOCATIONS["Reykjavik, Iceland"], FIXED_CONDITIONS)
        madrid = build_trust_panel_input("farmer", LOCATIONS["Madrid, Spain"], FIXED_CONDITIONS)
        nairobi = build_trust_panel_input("farmer", LOCATIONS["Nairobi, Kenya"], FIXED_CONDITIONS)

        assert reykjavik["gps_impact_score"] > madrid["gps_impact_score"] > nairobi["gps_impact_score"]

    def test_hf_blackout_risk_identical_across_all_locations(self):
        """HF risk is intentionally NOT latitude-adjusted (flares ionize the
        sunlit hemisphere roughly uniformly) — this must stay constant
        across all 3 locations given the same flare_class."""
        results = [
            build_trust_panel_input("farmer", loc, FIXED_CONDITIONS)["hf_blackout_risk_label"]
            for loc in LOCATIONS.values()
        ]
        assert len(set(results)) == 1

    def test_kp_index_and_flare_class_unchanged_across_locations(self):
        """Sanity check: the RAW inputs are genuinely identical — proves any
        score difference comes from location, not from different storm
        data being fed in."""
        results = [
            build_trust_panel_input("farmer", loc, FIXED_CONDITIONS)
            for loc in LOCATIONS.values()
        ]
        kp_values = {r["kp_index"] for r in results}
        flare_values = {r["solar_flare_class"] for r in results}
        assert kp_values == {7}
        assert flare_values == {"M4.1"}

    def test_all_four_roles_work_for_each_location(self):
        """Regression guard: every role x location combination must at
        least build a valid trust panel input without raising."""
        for role_key in ["farmer", "surveyor", "ham_radio_operator", "general_public"]:
            for loc in LOCATIONS.values():
                result = build_trust_panel_input(role_key, loc, FIXED_CONDITIONS)
                assert result["role"] == role_key