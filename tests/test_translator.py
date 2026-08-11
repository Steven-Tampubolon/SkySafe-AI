"""
tests/test_translator.py — Unit test untuk ai_layer/translator.py, semua
panggilan Groq API di-mock.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from ai_layer.translator import (
    call_translation_layer,
    TranslationError,
    _validate_output,
    _build_static_fallback,
)

SAMPLE_INPUT = {
    "role": "petani",
    "location_name": "Karawang, Jawa Barat",
    "local_time": "10 Agustus 2026, 09:00 WIB",
    "kp_index": 6,
    "solar_flare_class": "M4.1",
    "gps_impact_score": 62,
    "gps_impact_label": "Sedang",
    "hf_blackout_risk_label": "Rendah",
    "forecast_window": "08:00–16:00 UTC",
    "data_type": "forecast",
    "confidence_level": "Sedang",
    "confidence_reason": "forecast 12 jam ke depan, model geomagnetik punya margin error wajar",
    "source_name": "NOAA SWPC",
    "source_url": "https://www.swpc.noaa.gov/products/planetary-k-index",
}

VALID_LLM_OUTPUT = {
    "headline": "Akurasi GPS traktor Anda berpotensi terganggu sedang hari ini",
    "plain_explanation": "Ada badai geomagnetik sedang yang diperkirakan berlangsung 08:00-16:00 UTC.",
    "recommended_action": "Pertimbangkan tunda operasi presisi atau cek ulang titik acuan RTK.",
    "confidence_label": "Sedang",
    "why_confidence": "Ini masih prediksi 12 jam ke depan, bukan pengukuran real-time.",
    "source_citation": "NOAA Space Weather Prediction Center — swpc.noaa.gov",
}


def _mock_groq_response(content_dict_or_str):
    content = (
        content_dict_or_str if isinstance(content_dict_or_str, str)
        else json.dumps(content_dict_or_str)
    )
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_resp


class TestValidateOutput:
    def test_valid_output_passes(self):
        assert _validate_output(VALID_LLM_OUTPUT, SAMPLE_INPUT) is True

    def test_missing_key_fails(self):
        bad = dict(VALID_LLM_OUTPUT)
        del bad["headline"]
        assert _validate_output(bad, SAMPLE_INPUT) is False

    def test_confidence_mismatch_fails(self):
        bad = dict(VALID_LLM_OUTPUT)
        bad["confidence_label"] = "Tinggi"  # LLM mengarang, beda dari input "Sedang"
        assert _validate_output(bad, SAMPLE_INPUT) is False

    def test_empty_string_value_fails(self):
        bad = dict(VALID_LLM_OUTPUT)
        bad["headline"] = ""
        assert _validate_output(bad, SAMPLE_INPUT) is False


class TestStaticFallback:
    def test_fallback_has_all_required_keys(self):
        fallback = _build_static_fallback(SAMPLE_INPUT)
        assert fallback["_is_fallback"] is True
        assert fallback["confidence_label"] == "Sedang"
        assert "NOAA SWPC" in fallback["source_citation"]


class TestCallTranslationLayer:
    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_success_returns_ai_output(self, mock_getenv, mock_post):
        mock_post.return_value = _mock_groq_response(VALID_LLM_OUTPUT)

        result = call_translation_layer("petani", SAMPLE_INPUT)

        assert result["_is_fallback"] is False
        assert result["confidence_label"] == "Sedang"
        assert result["headline"] == VALID_LLM_OUTPUT["headline"]

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "llama-3.3-70b-versatile"
        assert kwargs["json"]["temperature"] == 0.2

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_llm_changes_number_triggers_fallback(self, mock_getenv, mock_post):
        tampered = dict(VALID_LLM_OUTPUT)
        tampered["confidence_label"] = "Tinggi"  # LLM "mengarang" -> harus ditolak
        mock_post.return_value = _mock_groq_response(tampered)

        result = call_translation_layer("petani", SAMPLE_INPUT)

        assert result["_is_fallback"] is True
        assert result["confidence_label"] == "Sedang"  # tetap dari input asli

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_invalid_json_from_llm_triggers_fallback(self, mock_getenv, mock_post):
        mock_post.return_value = _mock_groq_response("ini bukan JSON valid {{{")

        result = call_translation_layer("petani", SAMPLE_INPUT)
        assert result["_is_fallback"] is True

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_api_down_triggers_fallback(self, mock_getenv, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        result = call_translation_layer("petani", SAMPLE_INPUT)
        assert result["_is_fallback"] is True

    @patch("ai_layer.translator.os.getenv", return_value=None)
    def test_no_api_key_raises(self, mock_getenv):
        with pytest.raises(TranslationError):
            call_translation_layer("petani", SAMPLE_INPUT)

    def test_unsupported_role_raises(self):
        with pytest.raises(TranslationError):
            call_translation_layer("surveyor", SAMPLE_INPUT)  # belum didukung Minggu 1