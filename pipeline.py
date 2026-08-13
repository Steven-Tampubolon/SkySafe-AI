"""
pipeline.py — Shared orchestration: location -> conditions -> latitude-
adjusted scoring -> Trust Panel input dict. Used by both the CLI demo
script and the Streamlit UI so this logic lives in exactly one place.
"""

from scoring.scoring import (
    classify_latitude_band,
    score_gps_impact,
    score_hf_risk,
    compute_confidence,
)

NOAA_SOURCE_NAME = "NOAA SWPC"
NOAA_SOURCE_URL = "https://www.swpc.noaa.gov/products/planetary-k-index"


def build_trust_panel_input(role_key: str, resolved_location: dict, conditions: dict) -> dict:
    """
    Combine a geocoded location + current space weather conditions into the
    input schema expected by ai_layer.translator.call_translation_layer().

    Args:
        role_key: "farmer" | "surveyor" | "ham_radio_operator" | "general_public"
        resolved_location: output of ingestion.geocoding.geocode_location()
        conditions: output of ingestion.ingestion.get_current_conditions()
    """
    latitude_band = classify_latitude_band(resolved_location["latitude"], resolved_location["longitude"])
    gps_score, gps_label = score_gps_impact(conditions["kp_index"], latitude_band)
    hf_label = score_hf_risk(conditions["flare_class"])

    data_type = "forecast" if conditions.get("_from_cache") else "real-time"
    confidence_level, confidence_reason = compute_confidence(data_type, forecast_horizon_hours=0)

    return {
        "role": role_key,
        "location_name": resolved_location["resolved_name"],
        "local_time": conditions["fetched_at"],
        "kp_index": conditions["kp_index"],
        "solar_flare_class": conditions["flare_class"],
        "geomagnetic_latitude_band": latitude_band,
        "gps_impact_score": gps_score,
        "gps_impact_label": gps_label,
        "hf_blackout_risk_label": hf_label,
        "forecast_window": conditions["fetched_at"],
        "data_type": data_type,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "source_name": NOAA_SOURCE_NAME,
        "source_url": NOAA_SOURCE_URL,
    }