"""
scripts/demo_petani_e2e.py — Uji end-to-end alur lengkap:
ingestion -> scoring -> ai_layer (role petani).

PERLU API KEY ASLI (GROQ_API_KEY, NASA_DONKI_API_KEY) di .env.
Jalankan dari root project: python3 -m scripts.demo_petani_e2e
"""

import json

from ingestion.ingestion import get_current_conditions
from scoring.scoring import score_gps_impact, score_hf_risk, compute_confidence
from ai_layer.translator import call_translation_layer

def build_trust_panel_input(conditions: dict, location_name: str = "Karawang, Jawa Barat") -> dict:
    gps_score, gps_label = score_gps_impact(conditions["kp_index"])
    hf_label = score_hf_risk(conditions["flare_class"])

    data_type = "forecast" if conditions.get("_from_cache") else "real-time"
    confidence_level, confidence_reason = compute_confidence(data_type, forecast_horizon_hours=0)

    return {
        "role": "petani",
        "location_name": location_name,
        "local_time": conditions["fetched_at"],
        "kp_index": conditions["kp_index"],
        "solar_flare_class": conditions["flare_class"],
        "gps_impact_score": gps_score,
        "gps_impact_label": gps_label,
        "hf_blackout_risk_label": hf_label,
        "forecast_window": conditions["fetched_at"],
        "data_type": data_type,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "source_name": "NOAA SWPC",
        "source_url": "https://www.swpc.noaa.gov/products/planetary-k-index",
    }

if __name__ == "__main__":
    print("1) Ambil kondisi terkini dari NOAA + NASA DONKI...")
    conditions = get_current_conditions()
    print(json.dumps(conditions, indent=2))

    print("\n2) Hitung skor deterministik...")
    trust_input = build_trust_panel_input(conditions)
    print(json.dumps(trust_input, indent=2, ensure_ascii=False))

    print("\n3) Terjemahkan lewat groq (role: petani)...")
    output = call_translation_layer("petani", trust_input)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if output.get("_is_fallback"):
        print("\n⚠️  PERHATIAN: ini hasil FALLBACK statis, bukan dari AI. Cek log warning di atas.")
    else:
        print("\n✅ End-to-end sukses — output dari AI (Groq) lolos validasi-balik.")