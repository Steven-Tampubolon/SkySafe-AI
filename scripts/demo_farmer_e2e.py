"""
scripts/demo_farmer_e2e.py — End-to-end demo: location -> geocoding ->
ingestion -> latitude-adjusted scoring -> ai_layer (role: farmer).

Requires real API keys (GROQ_API_KEY, NASA_DONKI_API_KEY) in .env.
Run: python3 -m scripts.demo_farmer_e2e "Nairobi, Kenya"
Or:  python3 -m scripts.demo_farmer_e2e "Fargo, North Dakota, USA"
(try both to see the latitude adjustment in action)
"""

import sys
import json

from ingestion.ingestion import get_current_conditions
from ingestion.geocoding import geocode_location
from pipeline import build_trust_panel_input
from ai_layer.translator import call_translation_layer


if __name__ == "__main__":
    location_name = sys.argv[1] if len(sys.argv) > 1 else "Nairobi, Kenya"

    print(f"0) Geocoding '{location_name}'...")
    resolved = geocode_location(location_name)
    print(json.dumps(resolved, indent=2))

    print("\n1) Fetching current conditions from NOAA + NASA DONKI...")
    conditions = get_current_conditions()
    print(json.dumps(conditions, indent=2))

    print("\n2) Computing latitude-adjusted deterministic scores...")
    trust_input = build_trust_panel_input("farmer", resolved, conditions)
    print(json.dumps(trust_input, indent=2, ensure_ascii=False))

    print("\n3) Translating via Groq (role: farmer)...")
    output = call_translation_layer("farmer", trust_input)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if output.get("_is_fallback"):
        print("\n⚠️  This is a STATIC FALLBACK, not AI output. Check warning logs above.")
    else:
        print("\n✅ End-to-end success — AI output passed validation.")