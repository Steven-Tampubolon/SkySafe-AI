"""
ui/app.py — SkySafe AI Streamlit UI.

Location search + role selector (Wed) + Trust Panel (Thu): AI Explanation
section and Raw Data & Sources section, rendered side by side with
color-coded badges for gps_impact_label / hf_blackout_risk_label /
confidence_label.

Run from repo root: streamlit run ui/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ingestion.geocoding import geocode_location, GeocodingError
from ingestion.ingestion import get_current_conditions, IngestionError
from scoring.scoring import ScoringError
from ai_layer.translator import call_translation_layer, TranslationError
from pipeline import build_trust_panel_input

st.set_page_config(page_title="SkySafe AI", page_icon="🛰️", layout="centered")

st.title("🛰️ SkySafe AI")
st.caption("Space weather impact, translated for you — wherever you are.")

ROLE_OPTIONS = {
    "Farmer": "farmer",
    "Surveyor": "surveyor",
    "Ham Radio Operator": "ham_radio_operator",
    "General Public": "general_public",
}

IMPACT_BADGE_COLORS = {
    "Low": "#2e7d32",
    "Moderate": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}
CONFIDENCE_BADGE_COLORS = {
    "High": "#2e7d32",
    "Medium": "#f9a825",
    "Low": "#c62828",
}


def render_labeled_badge(prefix: str, badge_text: str, color: str):
    st.markdown(
        f"{prefix} <span style='background-color:{color};color:white;"
        f"padding:3px 12px;border-radius:12px;font-size:0.85em;"
        f"font-weight:600;'>{badge_text}</span>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_get_current_conditions():
    return get_current_conditions()


@st.cache_data(ttl=300, show_spinner=False)
def cached_call_translation_layer(role_key: str, data: dict):
    return call_translation_layer(role_key, data)


# --- Location + role input ---

with st.form("location_form"):
    location_input = st.text_input(
        "Your location",
        placeholder="e.g. Nairobi, Kenya",
        help="Type any city, region, or country name — anywhere in the world.",
    )
    role_label = st.selectbox("I am a...", list(ROLE_OPTIONS.keys()))
    submitted = st.form_submit_button("Check space weather impact")

if submitted:
    if not location_input.strip():
        st.error("Please enter a location.")
    else:
        with st.spinner(f"Looking up '{location_input}'..."):
            try:
                resolved = geocode_location(location_input)
                st.session_state["resolved_location"] = resolved
                st.session_state["role_key"] = ROLE_OPTIONS[role_label]
                st.session_state["role_label"] = role_label
            except GeocodingError as e:
                st.session_state.pop("resolved_location", None)
                st.error(
                    f"Couldn't find that location. Try being more specific "
                    f"(e.g. add a country name). Details: {e}"
                )

# --- Trust Panel ---

if "resolved_location" in st.session_state:
    resolved = st.session_state["resolved_location"]
    role_key = st.session_state["role_key"]

    st.divider()
    st.success(f"📍 **{resolved['resolved_name']}**")
    st.caption(f"Coordinates: {resolved['latitude']:.4f}, {resolved['longitude']:.4f}")

    with st.spinner("Fetching current space weather conditions..."):
        try:
            conditions = cached_get_current_conditions()
        except IngestionError as e:
            conditions = None
            st.error(f"Could not fetch space weather data right now. Please try again shortly. ({e})")

    if conditions:
        try:
            trust_input = build_trust_panel_input(role_key, resolved, conditions)
        except ScoringError as e:
            trust_input = None
            st.error(f"Scoring error: {e}")

        if trust_input:
            with st.spinner("Generating your briefing..."):
                try:
                    ai_output = cached_call_translation_layer(role_key, trust_input)
                except TranslationError as e:
                    ai_output = None
                    st.error(f"Translation layer configuration error: {e}")

            if ai_output:
                if ai_output.get("_is_fallback"):
                    st.warning(
                        "⚠️ AI explanation unavailable right now — showing a "
                        "basic summary based on the raw data instead."
                    )

                st.subheader("AI Explanation")
                st.markdown(f"**{ai_output['headline']}**")
                st.write(ai_output["plain_explanation"])
                st.markdown(f"**Recommended action:** {ai_output['recommended_action']}")

                st.subheader("Raw Data & Sources")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kp Index", trust_input["kp_index"])
                    st.write(f"**Solar flare class:** {trust_input['solar_flare_class']}")
                    st.write(f"**Geomagnetic band:** {trust_input['geomagnetic_latitude_band']}")
                with col2:
                    st.write(f"**GPS impact score:** {trust_input['gps_impact_score']} / 100")
                    render_labeled_badge(
                        "**GPS impact:**",
                        trust_input["gps_impact_label"],
                        IMPACT_BADGE_COLORS[trust_input["gps_impact_label"]],
                    )
                    render_labeled_badge(
                        "**HF blackout risk:**",
                        trust_input["hf_blackout_risk_label"],
                        IMPACT_BADGE_COLORS[trust_input["hf_blackout_risk_label"]],
                    )

                render_labeled_badge(
                    "**Confidence:**",
                    ai_output["confidence_label"],
                    CONFIDENCE_BADGE_COLORS[ai_output["confidence_label"]],
                )
                st.caption(ai_output["why_confidence"])

                st.markdown(f"**Source:** [{trust_input['source_name']}]({trust_input['source_url']})")
                st.caption(f"Data fetched at: {trust_input['local_time']}")