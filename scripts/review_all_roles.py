"""
scripts/review_all_roles.py — Manual QA aid for reviewing AI output tone
across all 4 roles x 2 scenarios (Low, High), using REAL Groq calls. Read
the printed output and judge whether any phrasing sounds stiff/robotic —
report back so the role templates can be refined if needed.

Requires GROQ_API_KEY in .env. Run: python3 -m scripts.review_all_roles
"""

import json

from ai_layer.translator import call_translation_layer

ROLES = ["farmer", "surveyor", "ham_radio_operator", "general_public"]

SCENARIOS = {
    "Low impact": {
        "kp_index": 2,
        "solar_flare_class": "C2.4",
        "geomagnetic_latitude_band": "Equatorial/Low",
        "gps_impact_score": 3.0,
        "gps_impact_label": "Low",
        "hf_blackout_risk_label": "Low",
        "data_type": "real-time",
        "confidence_level": "High",
        "confidence_reason": "Data measured directly from NOAA/NASA instruments, not a prediction.",
    },
    "High impact": {
        "kp_index": 8,
        "solar_flare_class": "X2.0",
        "geomagnetic_latitude_band": "High-Latitude/Auroral",
        "gps_impact_score": 77.0,
        "gps_impact_label": "High",
        "hf_blackout_risk_label": "High",
        "data_type": "forecast",
        "confidence_level": "Medium",
        "confidence_reason": "12-hour forecast, geomagnetic models carry a reasonable margin of error.",
    },
}

BASE = {
    "location_name": "Test Location",
    "local_time": "August 25, 2026, 09:00 local time",
    "forecast_window": "08:00-16:00 UTC",
    "source_name": "NOAA SWPC",
    "source_url": "https://www.swpc.noaa.gov/products/planetary-k-index",
}

if __name__ == "__main__":
    for role in ROLES:
        for scenario_name, scenario in SCENARIOS.items():
            data = dict(BASE, role=role, **scenario)
            print(f"\n{'=' * 70}\n{role.upper()} — {scenario_name}\n{'=' * 70}")
            output = call_translation_layer(role, data)
            if output.get("_is_fallback"):
                print("⚠️  FALLBACK — Groq call failed or validation rejected the output.")
            print(json.dumps(output, indent=2, ensure_ascii=False))