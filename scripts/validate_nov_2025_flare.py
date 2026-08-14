"""
scripts/validate_nov_2025_flare.py — Historical validation #2 (HF/R-scale).

Runs the full pipeline (location -> scoring -> AI translation) for the
ham_radio_operator role using the Nov 4, 2025 X1.8 flare (AR4274), which
triggered an R3 radio blackout across the Americas per NOAA SWPC. This
complements the May 2024 GPS/geomagnetic validation from Week 1.

Requires GROQ_API_KEY in .env. Run: python3 -m scripts.validate_nov_2025_flare
"""

import json

from pipeline import build_trust_panel_input
from ai_layer.translator import call_translation_layer

# A US location, since the Nov 2025 blackout was documented across the
# Americas — this keeps the demo geographically honest to the real event.
RESOLVED_LOCATION = {
    "latitude": 41.8781,
    "longitude": -87.6298,
    "resolved_name": "Chicago, Illinois, United States",
}

# Historical conditions from the Nov 4, 2025 event. kp_index is a
# plausible placeholder for that day (not the focus of this validation —
# the R-scale/HF side is) since we're validating hf_blackout_risk_label,
# not gps_impact_label, here.
HISTORICAL_CONDITIONS = {
    "kp_index": 4,
    "kp_forecast": [],
    "flare_class": "X1.8",
    "flare_time": "2025-11-04T12:00:00Z",
    "fetched_at": "2025-11-04T14:00:00+00:00",
}

if __name__ == "__main__":
    print("1) Building Trust Panel input from historical Nov 2025 event data...")
    trust_input = build_trust_panel_input("ham_radio_operator", RESOLVED_LOCATION, HISTORICAL_CONDITIONS)
    print(json.dumps(trust_input, indent=2, ensure_ascii=False))

    assert trust_input["hf_blackout_risk_label"] == "High", (
        f"Expected 'High', got {trust_input['hf_blackout_risk_label']!r} — "
        f"scoring regression, stop and investigate before proceeding."
    )
    print("\n✅ hf_blackout_risk_label correctly resolves to 'High' (R3 Strong).")

    print("\n2) Translating via Groq (role: ham_radio_operator)...")
    output = call_translation_layer("ham_radio_operator", trust_input)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if output.get("_is_fallback"):
        print("\n⚠️  This is a STATIC FALLBACK, not AI output. Check warning logs above.")
    else:
        print("\n✅ AI output passed validation — ready to screenshot as submission evidence.")