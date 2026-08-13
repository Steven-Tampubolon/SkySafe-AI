"""
scoring.py — SkySafe AI deterministic scoring module.

CORE PRINCIPLE: every function here is a PURE FUNCTION — no AI, no API
calls, no randomness. Same input always produces the same output, and every
number can be traced back to an official table:

  G-scale (geomagnetic, based on Kp):
    Kp 0-4 -> G0 | Kp 5 -> G1 Minor | Kp 6 -> G2 Moderate
    Kp 7 -> G3 Strong | Kp 8 -> G4 Severe | Kp 9 -> G5 Extreme

  R-scale (radio blackout, based on X-ray flare class):
    <M1 -> R0 | M1-M4 -> R1 Minor | M5-X0.9 -> R2 Moderate
    X1-X9 -> R3 Strong | X10-X19 -> R4 Severe | >=X20 -> R5 Extreme

Reference: https://www.swpc.noaa.gov/noaa-scales-explanation

GLOBAL-AUDIENCE REVISION: the G-scale table above is calibrated for
high-latitude/auroral regions, where geomagnetic storms hit hardest. The
SAME Kp value has a much weaker practical GPS impact near the equator. So
score_gps_impact() applies a location-dependent adjustment on top of the
raw G-scale score, using GEOMAGNETIC latitude (not geographic latitude —
Earth's magnetic pole is offset from the geographic pole, so two cities at
the same geographic latitude can sit in very different latitude bands).

R-scale (HF blackout) is intentionally NOT latitude-adjusted: a solar flare
ionizes the entire sunlit hemisphere roughly uniformly, unlike geomagnetic
storms which concentrate near the poles.
"""

import re
import math
from dataclasses import dataclass


class ScoringError(Exception):
    """Raised for invalid input (out-of-range Kp, unrecognized flare class,
    or unknown latitude band)."""
    pass


# --- G-scale (Kp -> raw GPS impact score, high-latitude calibration) ---

@dataclass(frozen=True)
class GTier:
    kp_min: float
    kp_max: float
    g_scale: str
    score_min: float
    score_max: float


G_TIERS = [
    GTier(kp_min=5.0, kp_max=6.0, g_scale="G1 Minor",    score_min=20, score_max=40),
    GTier(kp_min=6.0, kp_max=7.0, g_scale="G2 Moderate", score_min=40, score_max=55),
    GTier(kp_min=7.0, kp_max=8.0, g_scale="G3 Strong",   score_min=55, score_max=70),
    GTier(kp_min=8.0, kp_max=9.0, g_scale="G4 Severe",   score_min=70, score_max=85),
    GTier(kp_min=9.0, kp_max=9.0, g_scale="G5 Extreme",  score_min=85, score_max=100),
]
G0_TIER = GTier(kp_min=0.0, kp_max=5.0, g_scale="G0", score_min=0, score_max=20)


def get_g_scale(kp_index: float) -> str:
    """Return just the G-scale code (e.g. 'G2 Moderate') for display/debug."""
    if kp_index < 0 or kp_index > 9:
        raise ScoringError(f"kp_index out of valid range [0, 9]: {kp_index}")
    if kp_index >= 9.0:
        return "G5 Extreme"
    if kp_index < 5.0:
        return G0_TIER.g_scale
    for t in G_TIERS:
        if t.kp_min <= kp_index < t.kp_max:
            return t.g_scale
    return "G5 Extreme"


def _raw_score_from_kp(kp_index: float) -> float:
    """Raw 0-100 score from Kp alone, high-latitude calibration, linear
    within each G-scale tier. Returned UNROUNDED — rounding happens only at
    final display time in score_gps_impact(), so tier-boundary labels
    (e.g. Kp=4.99) aren't misclassified by an intermediate rounding
    artifact (19.96 rounding up to 20.0 before the label check)."""
    if kp_index < 0 or kp_index > 9:
        raise ScoringError(f"kp_index out of valid range [0, 9]: {kp_index}")

    if kp_index >= 9.0:
        return 100.0

    tier = G0_TIER if kp_index < 5.0 else None
    if tier is None:
        for t in G_TIERS:
            if t.kp_min <= kp_index < t.kp_max:
                tier = t
                break

    tier_span = tier.kp_max - tier.kp_min
    position = (kp_index - tier.kp_min) / tier_span if tier_span > 0 else 0
    position = min(position, 1.0)
    score_span = tier.score_max - tier.score_min
    return tier.score_min + position * score_span  # unrounded


# --- Geomagnetic latitude (dipole approximation) ---

# North geomagnetic pole, dipole approximation (~2020 epoch, IGRF).
# Accurate enough to bucket a location into a latitude band — not intended
# for precision scientific use.
GEOMAGNETIC_NORTH_POLE_LAT = 80.7
GEOMAGNETIC_NORTH_POLE_LON = -72.7


def geomagnetic_latitude(geo_lat: float, geo_lon: float) -> float:
    """
    Approximate geomagnetic latitude from geographic coordinates. This is
    what actually determines how hard a geomagnetic storm hits a location —
    NOT geographic latitude. Two cities at the same geographic latitude can
    sit in different latitude bands depending on longitude, because the
    magnetic pole is offset from the geographic pole.
    """
    lat_r = math.radians(geo_lat)
    lon_r = math.radians(geo_lon)
    pole_lat_r = math.radians(GEOMAGNETIC_NORTH_POLE_LAT)
    pole_lon_r = math.radians(GEOMAGNETIC_NORTH_POLE_LON)

    sin_gmlat = (
        math.sin(lat_r) * math.sin(pole_lat_r)
        + math.cos(lat_r) * math.cos(pole_lat_r) * math.cos(lon_r - pole_lon_r)
    )
    sin_gmlat = max(-1.0, min(1.0, sin_gmlat))
    return math.degrees(math.asin(sin_gmlat))


def classify_latitude_band(geo_lat: float, geo_lon: float) -> str:
    """
    Classify coordinates into the latitude band used by the scoring model,
    based on ABSOLUTE geomagnetic latitude:
      >= 55 deg -> "High-Latitude/Auroral"
      30-55 deg -> "Mid-Latitude"
      <  30 deg -> "Equatorial/Low"
    Thresholds are simplified approximations of the auroral oval's typical
    equatorward extent, not exact scientific boundaries — sufficient for
    routing to the right adjustment factor below.
    """
    gmlat = abs(geomagnetic_latitude(geo_lat, geo_lon))
    if gmlat >= 55:
        return "High-Latitude/Auroral"
    if gmlat >= 30:
        return "Mid-Latitude"
    return "Equatorial/Low"


# --- GPS impact: raw score x latitude adjustment ---

LATITUDE_ADJUSTMENT_FACTOR = {
    "High-Latitude/Auroral": 1.0,   # original NOAA table calibration
    "Mid-Latitude": 0.6,
    "Equatorial/Low": 0.35,
}


def _label_from_adjusted_score(score: float) -> str:
    """
    Bucket boundaries are NOT arbitrary round numbers — they follow the
    official G-scale tier groupings directly, so every boundary is
    traceable back to the NOAA table:
      G0            (score 0-20)  -> Low
      G1 + G2       (score 20-55) -> Moderate
      G3 + G4       (score 55-85) -> High
      G5            (score 85-100)-> Critical
    """
    if score < 20:
        return "Low"
    if score < 55:
        return "Moderate"
    if score < 85:
        return "High"
    return "Critical"


def score_gps_impact(kp_index: float, geomagnetic_latitude_band: str) -> tuple[float, str]:
    """
    Compute the GPS impact score (0-100) and label, adjusted for the
    location's geomagnetic latitude band.

    Two-step, fully deterministic model:
    1. Raw score from Kp alone via the official G-scale table (calibrated
       for high-latitude/auroral regions).
    2. Multiply by a latitude adjustment factor: High-Latitude/Auroral=1.0,
       Mid-Latitude=0.7, Equatorial/Low=0.4 — since the same storm has a
       much weaker practical GPS impact near the equator.

    The label is determined from the full-precision adjusted score, and
    ONLY THEN is the score itself rounded to 1 decimal for display — this
    keeps tier boundaries exact even when the displayed number is rounded.

    Args:
        kp_index: 0-9.
        geomagnetic_latitude_band: "High-Latitude/Auroral" | "Mid-Latitude"
            | "Equatorial/Low" — no default, must always be passed
            explicitly (use classify_latitude_band() to derive it).

    Returns:
        (adjusted_score, label)
    """
    if geomagnetic_latitude_band not in LATITUDE_ADJUSTMENT_FACTOR:
        raise ScoringError(
            f"Unrecognized geomagnetic_latitude_band: {geomagnetic_latitude_band!r}. "
            f"Must be one of: {list(LATITUDE_ADJUSTMENT_FACTOR.keys())}"
        )

    raw_score = _raw_score_from_kp(kp_index)
    factor = LATITUDE_ADJUSTMENT_FACTOR[geomagnetic_latitude_band]
    adjusted_score_precise = raw_score * factor

    label = _label_from_adjusted_score(adjusted_score_precise)
    adjusted_score = round(adjusted_score_precise, 1)

    return adjusted_score, label


# --- R-scale (flare -> HF blackout risk, NOT latitude-adjusted) ---

FLARE_CLASS_PATTERN = re.compile(r"^([ABCMX])(\d+(?:\.\d+)?)?$", re.IGNORECASE)

R_LABEL_MAP = {
    "R0": "Low",
    "R1 Minor": "Moderate",
    "R2 Moderate": "Moderate",
    "R3 Strong": "High",
    "R4 Severe": "High",
    "R5 Extreme": "Critical",
}


def get_r_scale(flare_class: str) -> str:
    """Return just the R-scale code (e.g. 'R2 Moderate') for display/debug."""
    if flare_class is None or flare_class.strip().lower() in ("none", "tidak ada", ""):
        return "R0"

    match = FLARE_CLASS_PATTERN.match(flare_class.strip())
    if not match:
        raise ScoringError(f"Unrecognized flare_class format: {flare_class!r}")

    letter = match.group(1).upper()
    magnitude = float(match.group(2)) if match.group(2) else 1.0

    if letter in ("A", "B", "C"):
        return "R0"
    if letter == "M":
        return "R1 Minor" if magnitude < 5 else "R2 Moderate"
    if letter == "X":
        if magnitude < 1:
            return "R2 Moderate"
        if magnitude < 10:
            return "R3 Strong"
        if magnitude < 20:
            return "R4 Severe"
        return "R5 Extreme"

    raise ScoringError(f"Unrecognized flare class: {flare_class!r}")


def score_hf_risk(flare_class: str) -> str:
    """
    Determine HF radio blackout risk label from flare class, per the
    official R-scale table. Intentionally NOT latitude-adjusted — flares
    ionize the whole sunlit hemisphere roughly uniformly.
    """
    return R_LABEL_MAP[get_r_scale(flare_class)]


# --- Confidence ---

def compute_confidence(data_type: str, forecast_horizon_hours: float = 0) -> tuple[str, str]:
    """
    Args:
        data_type: "real-time" or "forecast"
        forecast_horizon_hours: hours ahead the forecast applies to
            (ignored when data_type == "real-time")

    Returns:
        (level, reason) — level is "High" | "Medium" | "Low"
    """
    if data_type not in ("real-time", "forecast"):
        raise ScoringError(f"data_type must be 'real-time' or 'forecast', got: {data_type!r}")

    if data_type == "real-time":
        return "High", "Data measured directly from NOAA/NASA instruments, not a prediction."

    if forecast_horizon_hours < 24:
        return (
            "Medium",
            f"{forecast_horizon_hours:g}-hour forecast; space weather models "
            f"carry a reasonable margin of error at this range.",
        )

    return (
        "Low",
        f"{forecast_horizon_hours:g}-hour forecast; geomagnetic/solar model "
        f"accuracy drops significantly beyond 24 hours.",
    )