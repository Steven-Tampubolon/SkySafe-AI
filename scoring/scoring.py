"""
scoring.py — Modul scoring deterministik SkySafe AI.

PRINSIP: semua fungsi di sini PURE FUNCTION — tidak ada AI, tidak ada
panggilan API, tidak ada randomness. Input sama selalu menghasilkan output
sama, dan setiap angka bisa ditelusuri balik ke tabel resmi NOAA berikut:

  G-scale (geomagnetik, berbasis Kp):
    Kp 0-4 -> G0      | Kp 5 -> G1 Minor  | Kp 6 -> G2 Moderate
    Kp 7   -> G3 Strong | Kp 8 -> G4 Severe | Kp 9 -> G5 Extreme

  R-scale (radio blackout, berbasis kelas flare X-ray):
    <M1 -> R0 | M1-M4 -> R1 Minor | M5-X0.9 -> R2 Moderate
    X1-X9 -> R3 Strong | X10-X19 -> R4 Severe | >=X20 -> R5 Extreme

Referensi resmi: https://www.swpc.noaa.gov/noaa-scales-explanation
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GTier:
    kp_min: float
    kp_max: float  # eksklusif di ujung atas, kecuali tier terakhir
    g_scale: str
    label: str
    score_min: float
    score_max: float


# Urutan PENTING: dicek dari atas ke bawah, tier pertama yang cocok dipakai.
G_TIERS = [
    GTier(kp_min=5.0, kp_max=6.0, g_scale="G1 Minor",    label="Rendah–Sedang", score_min=20, score_max=40),
    GTier(kp_min=6.0, kp_max=7.0, g_scale="G2 Moderate",  label="Sedang",        score_min=40, score_max=55),
    GTier(kp_min=7.0, kp_max=8.0, g_scale="G3 Strong",    label="Tinggi",        score_min=55, score_max=70),
    GTier(kp_min=8.0, kp_max=9.0, g_scale="G4 Severe",    label="Tinggi",        score_min=70, score_max=85),
    GTier(kp_min=9.0, kp_max=9.0, g_scale="G5 Extreme", label="Kritis",        score_min=85, score_max=100),
]
# G0 = fallback kalau Kp < 5 (tidak butuh tier eksplisit karena rentangnya lebar 0-4.99)
G0_TIER = GTier(kp_min=0.0, kp_max=5.0, g_scale="G0", label="Rendah", score_min=0, score_max=20)


class ScoringError(Exception):
    """Dilempar kalau input di luar rentang valid (mis. Kp negatif atau >9)."""
    pass


def score_gps_impact(kp_index: float) -> tuple[float, str]:
    """
    Hitung skor dampak GPS (0-100) dan label berdasarkan Kp index, sesuai
    tabel G-scale resmi NOAA. Skor linear DI DALAM tiap tier, supaya setiap
    angka bisa dijelaskan: "Kp=6.33 ada di tier G2 (Kp 6-7, skor 40-55),
    posisi 0.33 dari tier -> skor = 40 + 0.33*15 = 44.95".

    Catatan: Kp index resmi NOAA hanya punya rentang 0-9 (tidak ada Kp 10+),
    jadi Kp=9.0 (G5 Extreme) adalah TITIK MAKSIMUM, bukan awal rentang —
    di-special-case langsung ke skor 100, bukan dipaksa masuk formula
    linear generik seperti tier lain.

    Args:
        kp_index: nilai Kp, boleh float (mis. 6.33) atau int, range 0-9.

    Returns:
        (score, label) — score dibulatkan 1 desimal, label sesuai tabel.

    Raises:
        ScoringError: kalau kp_index di luar rentang valid [0, 9].
    """
    if kp_index < 0 or kp_index > 9:
        raise ScoringError(f"kp_index di luar rentang valid [0, 9]: {kp_index}")

    if kp_index >= 9.0:
        return 100.0, "Kritis"

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
    score = tier.score_min + position * score_span

    return round(score, 1), tier.label


def get_g_scale(kp_index: float) -> str:
    """Kembalikan kode G-scale saja (mis. 'G2 Moderate') untuk keperluan display/debug."""
    if kp_index < 0 or kp_index > 9:
        raise ScoringError(f"kp_index di luar rentang valid [0, 9]: {kp_index}")
    if kp_index >= 9.0:
        return "G5 Extreme"
    if kp_index < 5.0:
        return G0_TIER.g_scale
    for t in G_TIERS:
        if t.kp_min <= kp_index < t.kp_max:
            return t.g_scale
    return "G5 Extreme"


# --- R-scale (flare -> HF blackout risk) ---

FLARE_CLASS_PATTERN = re.compile(r"^([ABCMX])(\d+(?:\.\d+)?)?$", re.IGNORECASE)

# Urutan huruf kelas flare dari lemah ke kuat (dipakai untuk perbandingan)
FLARE_CLASS_ORDER = ["A", "B", "C", "M", "X"]


def score_hf_risk(flare_class: str) -> str:
    """
    Tentukan label risiko blackout HF radio berdasarkan kelas flare, sesuai
    tabel R-scale resmi NOAA.

    Args:
        flare_class: mis. "M4.1", "X2.0", "C2.4", atau "Tidak ada" (tanpa flare).

    Returns:
        label: "Rendah" | "Rendah–Sedang" | "Sedang" | "Tinggi" | "Kritis"

    Raises:
        ScoringError: kalau format flare_class tidak dikenali.
    """
    if flare_class is None or flare_class.strip().lower() in ("tidak ada", "none", ""):
        return "Rendah"  # R0 — tidak ada flare signifikan

    match = FLARE_CLASS_PATTERN.match(flare_class.strip())
    if not match:
        raise ScoringError(f"Format flare_class tidak dikenali: {flare_class!r}")

    letter = match.group(1).upper()
    magnitude = float(match.group(2)) if match.group(2) else 1.0

    if letter in ("A", "B", "C"):
        return "Rendah"  # R0: < M1
    if letter == "M":
        if magnitude < 5:
            return "Rendah–Sedang"  # R1 Minor: M1-M4(.9)
        return "Sedang"  # R2 Moderate: M5-X0.9 (M5+ termasuk sini)
    if letter == "X":
        if magnitude < 1:
            return "Sedang"  # R2 Moderate: masih di bawah X1 (mis. X0.9)
        if magnitude < 10:
            return "Tinggi"  # R3 Strong: X1-X9
        if magnitude < 20:
            return "Tinggi"  # R4 Severe: X10-X19 (sama label "Tinggi", beda tier resmi)
        return "Kritis"  # R5 Extreme: >= X20

    raise ScoringError(f"Kelas flare tidak dikenali: {flare_class!r}")


def get_r_scale(flare_class: str) -> str:
    """Kembalikan kode R-scale saja (mis. 'R2 Moderate') untuk keperluan display/debug."""
    if flare_class is None or flare_class.strip().lower() in ("tidak ada", "none", ""):
        return "R0"

    match = FLARE_CLASS_PATTERN.match(flare_class.strip())
    if not match:
        raise ScoringError(f"Format flare_class tidak dikenali: {flare_class!r}")

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

    raise ScoringError(f"Kelas flare tidak dikenali: {flare_class!r}")


# --- Confidence ---

def compute_confidence(data_type: str, forecast_horizon_hours: float = 0) -> tuple[str, str]:
    """
    Tentukan tingkat keyakinan berdasarkan jenis data dan jarak waktu forecast.

    Args:
        data_type: "real-time" atau "forecast"
        forecast_horizon_hours: berapa jam ke depan data forecast berlaku
            (diabaikan kalau data_type == "real-time")

    Returns:
        (level, reason)
    """
    if data_type not in ("real-time", "forecast"):
        raise ScoringError(f"data_type harus 'real-time' atau 'forecast', dapat: {data_type!r}")

    if data_type == "real-time":
        return "Tinggi", "Data terukur langsung dari instrumen NOAA/NASA, bukan prediksi."

    if forecast_horizon_hours < 24:
        return "Sedang", f"Forecast {forecast_horizon_hours:g} jam ke depan, model cuaca antariksa punya margin error wajar pada rentang ini."

    return "Rendah", f"Forecast {forecast_horizon_hours:g} jam ke depan, akurasi model geomagnetik/solar menurun signifikan di luar 24 jam."