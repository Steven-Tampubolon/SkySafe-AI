"""
ui/app.py — SkySafe AI Streamlit UI.

Two modes:
  - Live Location: free-text global geocoding + live NOAA/DONKI fetch.
  - Historical Validation: replays the two documented historical events
    (May 2024 "Gannon storm" GPS impact, Nov 2025 X1.8 flare HF blackout)
    through the EXACT SAME Trust Panel rendering path as live queries — no
    live API calls, fixed data, clearly labeled as historical/demo. This is
    what makes the replay an honest proof rather than a separate bespoke
    view: same code, same trust guarantees, just fixed input.

Historical event data is imported directly from scripts/validate_*.py so
there is exactly one source of truth, shared between the CLI validation
scripts and this dashboard — no risk of the two drifting apart.

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
from scripts.validate_may_2024_storm import (
    RESOLVED_LOCATION as MAY2024_LOCATION,
    HISTORICAL_CONDITIONS as MAY2024_CONDITIONS,
)
from scripts.validate_nov_2025_flare import (
    RESOLVED_LOCATION as NOV2025_LOCATION,
    HISTORICAL_CONDITIONS as NOV2025_CONDITIONS,
)

st.set_page_config(page_title="SkySafe AI", page_icon="🛰️", layout="centered")

if "form_version" not in st.session_state:
    st.session_state["form_version"] = 0

if st.button("🔄 Start Over", help="Clear the current result and start a new lookup."):
    for key in ("data_source", "resolved_location", "role_key", "historical_key"):
        st.session_state.pop(key, None)
    st.session_state["form_version"] += 1
    st.rerun()

FV = st.session_state["form_version"]

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

HISTORICAL_EVENTS = {
    "📜 Historical: May 2024 Storm (GPS/G-scale)": {
        "role_key": "farmer",
        "role_label": "Farmer",
        "resolved_location": MAY2024_LOCATION,
        "conditions": MAY2024_CONDITIONS,
        "narrative": (
            "On May 10, 2024, the strongest geomagnetic storm in two decades "
            "struck right in the middle of US corn planting season. "
            "GPS-guided tractors, planters, and spray equipment across the "
            "Midwest lost accuracy for hours — some receivers were reportedly "
            "off by up to 230 feet at the storm's peak, with effects "
            "lingering for up to two days. Agricultural economists at Kansas "
            "State University later estimated the storm cost Midwest farmers "
            "roughly $500-565 million in delayed planting and lost yield. "
            "An estimated seven in ten planted acres in the US now rely on "
            "GPS-guided equipment — this wasn't a minor inconvenience, it "
            "was a direct hit on food production, and most farmers had never "
            "even heard the term 'space weather' before their equipment "
            "stopped working that day."
        ),
        "sources": [
            {
                "name": "Space.com — May 2024 solar storm cost $500 million in damages to farmers",
                "url": "https://www.space.com/astronomy/sun/may-2024-solar-storm-cost-usd500-million-in-damages-to-farmers-new-study-reveals",
            },
            {
                "name": "Kansas State University — Could last year's Gannon storm impact farmers again?",
                "url": "https://www.ksre.k-state.edu/news-and-publications/news/stories/2025/04/agriculture-solar-weather-gps-outage.html",
            },
        ],
    },
    "📜 Historical: Nov 2025 Flare (HF/R-scale)": {
        "role_key": "ham_radio_operator",
        "role_label": "Ham Radio Operator",
        "resolved_location": NOV2025_LOCATION,
        "conditions": NOV2025_CONDITIONS,
        "narrative": (
            "On November 4, 2025, an X1.8-class solar flare erupted from "
            "sunspot region AR4274 and, within minutes, triggered a strong "
            "R3 radio blackout across most of North and South America. "
            "High-frequency radio — the backbone of aviation communication, "
            "maritime safety calls, and emergency and amateur radio networks "
            "— degraded across the entire sunlit side of the continent for "
            "up to an hour. For ham radio operators and emergency responders "
            "who depend on HF bands, that's not an abstract space-weather "
            "footnote — it's a communication blackout at exactly the moment "
            "it might matter most, often with little to no public-facing "
            "warning that it was coming."
        ),
        "sources": [
            {
                "name": "Space.com — Sun unleashes 2 colossal X-class solar flares, knocking out radio signals across the Americas",
                "url": "https://www.space.com/astronomy/sun/sun-unleashes-2-colossal-x-class-solar-flares-knocking-out-radio-signals-across-the-americas-and-pacific",
            },
        ],
    },
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


def render_trust_panel(role_key: str, resolved: dict, conditions: dict):
    """Shared rendering path for BOTH live and historical data — the same
    function, same Trust Panel, regardless of data source."""
    try:
        trust_input = build_trust_panel_input(role_key, resolved, conditions)
    except ScoringError as e:
        st.error(f"Scoring error: {e}")
        return

    with st.spinner("Generating your briefing..."):
        try:
            ai_output = cached_call_translation_layer(role_key, trust_input)
        except TranslationError as e:
            st.error(f"Translation layer configuration error: {e}")
            return

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
    st.caption(f"Data timestamp: {trust_input['local_time']}")


# --- Mode selector ---

mode = st.radio("Mode", ["Live Location"] + list(HISTORICAL_EVENTS.keys()), key=f"mode_radio_{FV}")

if mode == "Live Location":
    with st.form("location_form"):
        location_input = st.text_input(
            "Your location",
            placeholder="e.g. Nairobi, Kenya",
            help="Type any city, region, or country name — anywhere in the world.",
            key=f"location_input_field_{FV}",
        )
        role_label = st.selectbox(
            "I am a...", list(ROLE_OPTIONS.keys()), key=f"role_select_field_{FV}"
        )
        submitted = st.form_submit_button("Check space weather impact")

    if submitted:
        if not location_input.strip():
            st.error("Please enter a location.")
        else:
            with st.spinner(f"Looking up '{location_input}'..."):
                try:
                    resolved = geocode_location(location_input)
                    st.session_state["data_source"] = "live"
                    st.session_state["resolved_location"] = resolved
                    st.session_state["role_key"] = ROLE_OPTIONS[role_label]
                except GeocodingError as e:
                    st.session_state.pop("resolved_location", None)
                    st.error(
                        f"Couldn't find that location. Try being more specific "
                        f"(e.g. add a country name). Details: {e}"
                    )

    if st.session_state.get("data_source") == "live" and "resolved_location" in st.session_state:
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
            render_trust_panel(role_key, resolved, conditions)

else:
    event = HISTORICAL_EVENTS[mode]

    st.subheader("🗞️ Why This Matters")
    st.write(event["narrative"])
    st.markdown("**Sources:**")
    for src in event["sources"]:
        st.markdown(f"- [{src['name']}]({src['url']})")

    st.caption(
        "This replays the actual measured data from this event through the "
        "exact same scoring + AI pipeline used for live queries — no live "
        "API calls, fixed historical data, for validation purposes."
    )

    if st.button("Load this historical event"):
        st.session_state["data_source"] = "historical"
        st.session_state["historical_key"] = mode

    if (
        st.session_state.get("data_source") == "historical"
        and st.session_state.get("historical_key") == mode
    ):
        resolved = event["resolved_location"]
        role_key = event["role_key"]

        st.divider()
        st.warning(
            "📜 **Historical Validation Mode** — fixed data from a "
            "documented past event, not a live query."
        )
        st.success(f"📍 **{resolved['resolved_name']}** — Role: **{event['role_label']}**")

        render_trust_panel(role_key, resolved, event["conditions"])

st.divider()
st.caption(
    "Data sources: NOAA SWPC, NASA DONKI, Open-Meteo Geocoding. "
    "AI explanations generated via Groq (Llama 3.3)."
)