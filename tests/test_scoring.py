"""
tests/test_scoring.py — Unit test untuk scoring.py, mencakup semua tier
G-scale (G0-G5) dan R-scale (R0-R5) sesuai tabel resmi NOAA.
"""

import pytest

from scoring.scoring import (
    score_gps_impact,
    get_g_scale,
    score_hf_risk,
    get_r_scale,
    compute_confidence,
    ScoringError,
)


class TestScoreGpsImpact:
    """Setiap tier G-scale dites di titik tengah DAN di batas (boundary)."""

    @pytest.mark.parametrize("kp,expected_label", [
        (0, "Rendah"), (2, "Rendah"), (4.99, "Rendah"),        # G0
        (5.0, "Rendah–Sedang"), (5.5, "Rendah–Sedang"),        # G1
        (6.0, "Sedang"), (6.5, "Sedang"),                       # G2
        (7.0, "Tinggi"), (7.5, "Tinggi"),                       # G3
        (8.0, "Tinggi"), (8.5, "Tinggi"),                       # G4
        (9.0, "Kritis"),                                        # G5
    ])
    def test_all_tiers_label(self, kp, expected_label):
        score, label = score_gps_impact(kp)
        assert label == expected_label

    @pytest.mark.parametrize("kp,expected_gscale", [
        (3, "G0"), (5, "G1 Minor"), (6, "G2 Moderate"),
        (7, "G3 Strong"), (8, "G4 Severe"), (9, "G5 Extreme"),
    ])
    def test_g_scale_mapping(self, kp, expected_gscale):
        assert get_g_scale(kp) == expected_gscale

    def test_score_is_linear_within_tier(self):
        # Kp=6.0 (awal tier G2) harus dapat skor 40 (batas bawah tier)
        score_start, _ = score_gps_impact(6.0)
        assert score_start == 40.0

        # Kp=6.5 (tengah tier G2, span Kp 6-7 -> skor 40-55) -> skor 47.5
        score_mid, _ = score_gps_impact(6.5)
        assert score_mid == 47.5

    def test_score_at_max_kp_is_100(self):
        score, label = score_gps_impact(9.0)
        assert score == 100.0
        assert label == "Kritis"

    def test_score_at_zero_kp_is_zero(self):
        score, label = score_gps_impact(0)
        assert score == 0.0
        assert label == "Rendah"

    def test_score_range_always_0_to_100(self):
        for kp in [i * 0.5 for i in range(0, 19)]:  # 0.0 sampai 9.0, step 0.5
            score, _ = score_gps_impact(kp)
            assert 0 <= score <= 100

    def test_out_of_range_raises(self):
        with pytest.raises(ScoringError):
            score_gps_impact(-1)
        with pytest.raises(ScoringError):
            score_gps_impact(9.5)


class TestScoreHfRisk:
    """Setiap tier R-scale (R0-R5), termasuk kasus 'Tidak ada' flare."""

    @pytest.mark.parametrize("flare_class,expected_label", [
        (None, "Rendah"),               # tidak ada flare -> R0
        ("Tidak ada", "Rendah"),
        ("A5.0", "Rendah"),             # R0
        ("B3.2", "Rendah"),             # R0
        ("C9.9", "Rendah"),             # R0
        ("M1.0", "Rendah–Sedang"),      # R1 Minor
        ("M4.9", "Rendah–Sedang"),      # R1 Minor
        ("M5.0", "Sedang"),             # R2 Moderate
        ("X0.5", "Sedang"),             # R2 Moderate (di bawah X1)
        ("X0.9", "Sedang"),             # R2 Moderate
        ("X1.0", "Tinggi"),             # R3 Strong
        ("X9.9", "Tinggi"),             # R3 Strong
        ("X10.0", "Tinggi"),            # R4 Severe (label sama "Tinggi")
        ("X19.9", "Tinggi"),            # R4 Severe
        ("X20.0", "Kritis"),            # R5 Extreme
        ("X30.0", "Kritis"),            # R5 Extreme
    ])
    def test_all_tiers_label(self, flare_class, expected_label):
        assert score_hf_risk(flare_class) == expected_label

    @pytest.mark.parametrize("flare_class,expected_rscale", [
        ("Tidak ada", "R0"),
        ("C2.4", "R0"),
        ("M4.1", "R1 Minor"),
        ("M5.0", "R2 Moderate"),
        ("X2.0", "R3 Strong"),
        ("X15.0", "R4 Severe"),
        ("X20.0", "R5 Extreme"),
    ])
    def test_r_scale_mapping(self, flare_class, expected_rscale):
        assert get_r_scale(flare_class) == expected_rscale

    def test_case_insensitive(self):
        assert score_hf_risk("m4.1") == "Rendah–Sedang"
        assert score_hf_risk("x2.0") == "Tinggi"

    def test_invalid_format_raises(self):
        with pytest.raises(ScoringError):
            score_hf_risk("Z9.9")
        with pytest.raises(ScoringError):
            score_hf_risk("random text")


class TestComputeConfidence:
    def test_realtime_is_high(self):
        level, reason = compute_confidence("real-time")
        assert level == "Tinggi"
        assert reason  # ada penjelasan, tidak kosong

    def test_forecast_under_24h_is_medium(self):
        level, _ = compute_confidence("forecast", forecast_horizon_hours=12)
        assert level == "Sedang"

    def test_forecast_over_24h_is_low(self):
        level, _ = compute_confidence("forecast", forecast_horizon_hours=48)
        assert level == "Rendah"

    def test_forecast_exactly_24h_is_low(self):
        # boundary: >= 24 jam dianggap Rendah (bukan Sedang)
        level, _ = compute_confidence("forecast", forecast_horizon_hours=24)
        assert level == "Rendah"

    def test_invalid_data_type_raises(self):
        with pytest.raises(ScoringError):
            compute_confidence("invalid_type")


class TestValidatesAgainstMay2024Storm:
    """
    Validasi terhadap event historis nyata: badai geomagnetik 10-11 Mei 2024
    ("Gannon storm"), Kp mencapai 9 (G5 Extreme) — badai terkuat dalam ~20
    tahun terakhir. Terdokumentasi luas mengganggu GPS presisi pertanian di
    Amerika Utara pada masa tanam, menyebabkan traktor RTK-GPS kehilangan
    akurasi hingga tidak bisa beroperasi presisi. Event ini jadi acuan utama
    demo "petani" SkySafe AI karena dampaknya nyata dan terdokumentasi resmi.

    Sumber: NOAA SWPC — https://www.swpc.noaa.gov/news/g5-extreme-geomagnetic-storm-conditions-observed
    """

    def test_validates_against_may_2024_storm(self):
        kp_historis = 9  # Kp aktual saat puncak badai Mei 2024
        score, label = score_gps_impact(kp_historis)

        assert label == "Kritis"
        assert score == 100.0

        g_scale = get_g_scale(kp_historis)
        assert g_scale == "G5 Extreme"

    def test_validates_moderate_tier_event(self):
        """
        Bukti tambahan bahwa scoring akurat bukan cuma di ekstrem: contoh
        badai G2 Moderate (Kp=6) yang jauh lebih umum terjadi dibanding G5.
        """
        score, label = score_gps_impact(6)
        assert label == "Sedang"
        assert get_g_scale(6) == "G2 Moderate"