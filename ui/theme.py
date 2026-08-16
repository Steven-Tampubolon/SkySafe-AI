"""
ui/theme.py — "Cosmic Professional" design system for SkySafe AI.

Pure presentation layer: NO business logic imports (no ingestion/,
scoring/, ai_layer/), no data fetching, no session_state access. Every
function here takes already-computed values as arguments and returns
nothing but rendered Streamlit markup — this file has no opinion about
where the data came from.

CSS overrides for native Streamlit widgets (buttons, inputs, selects,
segmented control) target internal `data-testid` attributes, which are
undocumented and can shift between Streamlit versions. Pinned/tested
against Streamlit 1.61.1 — if a future Streamlit upgrade breaks the
widget styling, this is the first place to check.
"""

import streamlit as st

COLORS = {
    "background": "#051424",
    "surface": "#0d1c2d",
    "surface-variant": "#273647",
    "surface-container": "#122131",
    "surface-container-high": "#1c2b3c",
    "on-surface": "#d4e4fa",
    "on-surface-variant": "#94a3b8",
    "outline": "#334155",
    "outline-variant": "#45464d",
    "primary": "#bec6e0",
    "secondary": "#4edea3",
    "on-secondary": "#003824",
    "tertiary": "#ffb95f",
    "alert-moderate": "#f59e0b",
    "alert-high": "#f97316",
    "alert-critical": "#e11d48",
    "error": "#ffb4ab",
}

# Maps directly to the REAL labels produced by scoring.py / translator.py —
# gps_impact_label, hf_blackout_risk_label ("Low"|"Moderate"|"High"|"Critical")
# and confidence_label ("High"|"Medium"|"Low"). If scoring.py's label set
# ever changes, update here — this is the single source of truth for badge
# color across the whole UI.
IMPACT_BADGE_COLORS = {
    "Low": COLORS["secondary"],
    "Moderate": COLORS["alert-moderate"],
    "High": COLORS["alert-high"],
    "Critical": COLORS["alert-critical"],
}
CONFIDENCE_BADGE_COLORS = {
    "High": COLORS["secondary"],
    "Medium": COLORS["alert-moderate"],
    "Low": COLORS["alert-critical"],
}


def inject_global_styles():
    """Call once, right after st.set_page_config()."""
    st.markdown(
        f"""
    <style>
        /* @import is used instead of <link> tags because <link> placed
           outside <head> (which is unavoidable via st.markdown) is not
           reliably processed by all browsers — @import inside a <style>
           block works regardless of where that <style> tag ends up in
           the DOM. */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

        .stApp {{
            background-color: {COLORS['background']};
            background-image:
                radial-gradient(circle at 15% 50%, rgba(78, 222, 163, 0.03), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(190, 198, 224, 0.03), transparent 25%);
            color: {COLORS['on-surface']};
        }}
        /* .stApp is a stable Streamlit root class across versions, unlike
           the hashed st-emotion-cache-* class names — targeting
           ".stApp *" reliably reaches every descendant regardless of
           Streamlit's internal class naming. */
        .stApp, .stApp * {{ font-family: 'Inter', sans-serif !important; }}

        /* MUST come AFTER the ".stApp *" rule above — both use !important
           at equal specificity, so source order decides the winner. This
           rule needs to override the global Inter rule specifically for
           icon spans, or icon ligatures render as plain text instead of
           glyphs (this exact bug happened once already — don't reorder
           this block above the global font rule again). */
        .stApp .material-symbols-outlined {{
            font-family: 'Material Symbols Outlined' !important;
            font-weight: normal;
            font-style: normal;
            vertical-align: middle;
            line-height: 1;
        }}

        /* --- Glass card / metric primitives (used by render_* helpers) --- */
        .glass-card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(12px);
        }}
        .metric-label {{
            font-size: 0.75rem;
            color: {COLORS['on-surface-variant']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }}
        .metric-value {{
            font-family: monospace;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
            margin: 0.4rem 0;
        }}
        .action-box {{
            background: rgba(30, 41, 59, 0.5);
            border-left: 4px solid {COLORS['tertiary']};
            padding: 1.25rem;
            border-radius: 8px;
            margin-top: 1rem;
        }}
        .progress-track {{
            width: 100%;
            height: 4px;
            background: {COLORS['surface-variant']};
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
        }}

        /* --- Native widget restyling (best-effort, targets Streamlit
             internals — may need adjustment on Streamlit version bumps) --- */
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: {COLORS['background']} !important;
            border: 1px solid {COLORS['outline-variant']} !important;
            color: {COLORS['on-surface']} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stTextInput"] input:focus {{
            border-color: {COLORS['secondary']} !important;
            box-shadow: 0 0 0 1px {COLORS['secondary']} !important;
        }}

        [data-testid="stFormSubmitButton"] button {{
            background-color: {COLORS['secondary']} !important;
            color: {COLORS['on-secondary']} !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }}
        [data-testid="stFormSubmitButton"] button:hover {{
            opacity: 0.9;
        }}

        .stButton button {{
            background-color: transparent !important;
            color: {COLORS['on-surface']} !important;
            border: 1px solid {COLORS['outline-variant']} !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }}
        .stButton button:hover {{
            border-color: {COLORS['secondary']} !important;
            color: {COLORS['secondary']} !important;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(15, 23, 42, 0.4) !important;
            border-color: {COLORS['outline-variant']} !important;
            border-radius: 12px !important;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_icon(name: str, size: int = 18) -> str:
    """Returns an inline Material Symbols icon span. Embed directly in an
    f-string alongside other HTML — NOT a standalone st.markdown call."""
    return f"<span class='material-symbols-outlined' style='font-size:{size}px;'>{name}</span>"


# Icon is derived from the badge COLOR (not the label text), matching the
# severity-based color logic already agreed on: green=good regardless of
# whether the word is "Low" (impact) or "High" (confidence), etc. Adding
# this here means every existing render_badge() call site automatically
# gets an icon with zero changes required elsewhere.
SEVERITY_ICON_BY_COLOR = {
    COLORS["secondary"]: "✓",
    COLORS["alert-moderate"]: "⚠",
    COLORS["alert-high"]: "✕",
    COLORS["alert-critical"]: "✕",
}


def render_badge(label: str, value: str, color: str):
    """A small text label followed by a colored pill badge, prefixed with
    a severity icon (✓/⚠/✕) derived from `color` so meaning doesn't rely
    on color alone. `color` should come from IMPACT_BADGE_COLORS or
    CONFIDENCE_BADGE_COLORS — callers should never hardcode a color here."""
    icon = SEVERITY_ICON_BY_COLOR.get(color, "")
    st.markdown(
        f"<span style='color:{COLORS['on-surface-variant']};font-size:0.85rem;'>{label}</span> "
        f"<span style='background-color:{color}22;color:{color};border:1px solid {color}66;"
        f"padding:3px 12px;border-radius:9999px;font-size:0.8rem;font-weight:600;'>{icon} {value}</span>",
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str):
    st.markdown(
        f"""
    <div>
        <p class="metric-label">{label}</p>
        <p class="metric-value">{value}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_progress_bar(fraction: float, color: str):
    """fraction: 0.0-1.0, already computed by the caller from a REAL scale
    (e.g. kp_index/9, gps_impact_score/100) — this function never invents
    a number, it only draws the bar for a value passed in."""
    fraction = max(0.0, min(1.0, fraction))
    st.markdown(
        f"""
    <div class="progress-track">
        <div class="progress-fill" style="width:{fraction * 100:.1f}%;background-color:{color};"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Ordinal position of each real label — used only to derive a visual
# fraction (0.0-1.0) for progress bars from labels that don't have a raw
# numeric score (e.g. hf_blackout_risk_label). This is a presentation-only
# derivation from an EXISTING label, not a new claim about the data.
IMPACT_TIER_ORDER = ["Low", "Moderate", "High", "Critical"]


def tier_fraction(label: str) -> float:
    try:
        return IMPACT_TIER_ORDER.index(label) / (len(IMPACT_TIER_ORDER) - 1)
    except ValueError:
        return 0.0


def render_headline(text: str):
    st.markdown(
        f"<h1 style='font-size:2.75rem;font-weight:700;line-height:1.15;"
        f"margin-bottom:0.75rem;'>{text}</h1>",
        unsafe_allow_html=True,
    )


def render_teaser_card(title: str, text: str, hint: str):
    """Static, presentation-only teaser card — no click-to-navigate JS,
    since st.tabs() has no supported API for programmatic tab switching
    (unlike st.radio/st.selectbox, which bind to session_state). `hint`
    should point the user to where to go manually."""
    st.markdown(
        f"""
    <div class="glass-card" style="border-left: 3px solid {COLORS['tertiary']};">
        <p style="color:{COLORS['tertiary']};font-size:0.8rem;font-weight:700;
           text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">
            {render_icon('history', 16)} {title}
        </p>
        <p style="font-size:0.85rem;color:{COLORS['on-surface-variant']};margin-bottom:0.5rem;">
            {text}
        </p>
        <p style="font-size:0.8rem;color:{COLORS['secondary']};font-weight:600;margin-bottom:0;">
            {hint}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_status_pill(text: str, color: str):
    """Small uppercase pill with a pulsing dot — decorative status
    indicator (e.g. 'LIVE SYNC', 'HISTORICAL REPLAY'), not tied to any
    business data field."""
    st.markdown(
        f"<span style='background:{color}1a;color:{color};padding:4px 10px;"
        f"border-radius:9999px;font-size:0.7rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.05em;'>"
        f"<span style='display:inline-block;width:6px;height:6px;"
        f"border-radius:50%;background:{color};margin-right:6px;'></span>"
        f"{text}</span>",
        unsafe_allow_html=True,
    )


def render_action_box(action_text: str):
    st.markdown(
        f"""
    <div class="action-box">
        <p style="font-weight:700;margin-bottom:0.5rem;">{render_icon('warning', 18)} Recommended Action</p>
        <p style="font-size:0.9rem;margin-bottom:0;">{action_text}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )