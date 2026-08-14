"""
scripts/validate_may_2024_storm.py — Historical validation #1 (GPS/G-scale).

Runs the full pipeline (location -> scoring -> AI translation) for the
farmer role using the May 10-11, 2024 "Gannon storm" (Kp=9, G5 Extreme) —
the strongest geomagnetic storm in ~20 years, documented to have disrupted
GPS-guided precision agriculture across North America. This is the
full-pipeline counterpart to test_validates_against_may_2024_storm() in
test_scoring.py (which only checks the scoring function in isolation), and
mirrors validate_nov_2025_flare.py's structure for the HF/R-scale side.

Requires GROQ_API_KEY in .env. Run: python3 -m scripts.validate_may_2024_storm
"""

import json

from pipeline import build_trust_panel_input
from ai_layer.translator import call_translation_layer

# High-latitude US location — the Gannon storm's documented precision-ag
# disruption was concentrated in North America at these latitudes.
RESOLVED_LOCATION = {
    "latitude": 46.8772,
    "longitude": -96.7898,
    "resolved_name": "Fargo, North Dakota, United States",
}

# Historical conditions from the May 10-11, 2024 event. No significant
# flare tied to this specific validation point — we're validating
# gps_impact_label here, not hf_blackout_risk_label (that's Nov 2025's job).
HISTORICAL_CONDITIONS = {
    "kp_index": 9,
    "kp_forecast": [],
    "flare_class": "None",
    "flare_time": None,
    "fetched_at": "2024-05-10T20:00:00+00:00",
}

if __name__ == "__main__":
    print("1) Building Trust Panel input from historical May 2024 storm data...")
    trust_input = build_trust_panel_input("farmer", RESOLVED_LOCATION, HISTORICAL_CONDITIONS)
    print(json.dumps(trust_input, indent=2, ensure_ascii=False))

    assert trust_input["geomagnetic_latitude_band"] == "High-Latitude/Auroral", (
        f"Expected 'High-Latitude/Auroral', got "
        f"{trust_input['geomagnetic_latitude_band']!r} — location classification "
        f"regression, stop and investigate before proceeding."
    )
    assert trust_input["gps_impact_label"] == "Critical", (
        f"Expected 'Critical', got {trust_input['gps_impact_label']!r} — "
        f"scoring regression, stop and investigate before proceeding."
    )
    print("\n✅ gps_impact_label correctly resolves to 'Critical' (G5 Extreme, high latitude).")

    print("\n2) Translating via Groq (role: farmer)...")
    output = call_translation_layer("farmer", trust_input)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if output.get("_is_fallback"):
        print("\n⚠️  This is a STATIC FALLBACK, not AI output. Check warning logs above.")
    else:
        print("\n✅ AI output passed validation — ready to screenshot as submission evidence.")