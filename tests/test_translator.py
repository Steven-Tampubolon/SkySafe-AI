"""
tests/test_translator.py — Unit tests for ai_layer/translator.py. All Groq
API calls are mocked.
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
    ROLE_TEMPLATES,
)

SAMPLE_INPUT = {
    "role": "farmer",
    "location_name": "Fargo, North Dakota, USA",
    "local_time": "August 17, 2026, 09:00 local time",
    "kp_index": 6,
    "solar_flare_class": "M4.1",
    "geomagnetic_latitude_band": "High-Latitude/Auroral",
    "gps_impact_score": 68,
    "gps_impact_label": "Moderate",
    "hf_blackout_risk_label": "Low",
    "forecast_window": "08:00-16:00 UTC",
    "data_type": "forecast",
    "confidence_level": "Medium",
    "confidence_reason": "12-hour forecast, geomagnetic models carry a reasonable margin of error",
    "source_name": "NOAA SWPC",
    "source_url": "https://www.swpc.noaa.gov/products/planetary-k-index",
}

VALID_LLM_OUTPUT = {
    "headline": "Your tractor's GPS may see moderate accuracy loss today",
    "plain_explanation": "A moderate geomagnetic storm is forecast between 08:00-16:00 UTC.",
    "recommended_action": "Consider shifting precision planting to tomorrow morning.",
    "confidence_label": "Medium",
    "why_confidence": "This is a 12-hour forecast, not a real-time measurement.",
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


class TestRoleTemplatesFormatCleanly:
    """Regression guard: every role template must format cleanly with no
    leftover/broken braces once input_json is substituted."""

    @pytest.mark.parametrize("role", list(ROLE_TEMPLATES.keys()))
    def test_template_formats_without_error(self, role):
        prompt = ROLE_TEMPLATES[role].format(input_json=json.dumps(SAMPLE_INPUT))
        assert "{input_json}" not in prompt
        assert "{{" not in prompt and "}}" not in prompt


class TestValidateOutput:
    def test_valid_output_passes(self):
        assert _validate_output(VALID_LLM_OUTPUT, SAMPLE_INPUT) is True

    def test_missing_key_fails(self):
        bad = dict(VALID_LLM_OUTPUT)
        del bad["headline"]
        assert _validate_output(bad, SAMPLE_INPUT) is False

    def test_confidence_mismatch_fails(self):
        bad = dict(VALID_LLM_OUTPUT)
        bad["confidence_label"] = "High"  # LLM invented a different value
        assert _validate_output(bad, SAMPLE_INPUT) is False


class TestStaticFallback:
    def test_fallback_has_all_required_keys(self):
        fallback = _build_static_fallback(SAMPLE_INPUT)
        assert fallback["_is_fallback"] is True
        assert fallback["confidence_label"] == "Medium"
        assert "NOAA SWPC" in fallback["source_citation"]


class TestCallTranslationLayer:
    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_success_returns_ai_output(self, mock_getenv, mock_post):
        mock_post.return_value = _mock_groq_response(VALID_LLM_OUTPUT)

        result = call_translation_layer("farmer", SAMPLE_INPUT)

        assert result["_is_fallback"] is False
        assert result["confidence_label"] == "Medium"

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "llama-3.3-70b-versatile"
        assert kwargs["json"]["temperature"] == 0.2

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_llm_changes_value_triggers_fallback(self, mock_getenv, mock_post):
        tampered = dict(VALID_LLM_OUTPUT)
        tampered["confidence_label"] = "High"
        mock_post.return_value = _mock_groq_response(tampered)

        result = call_translation_layer("farmer", SAMPLE_INPUT)

        assert result["_is_fallback"] is True
        assert result["confidence_label"] == "Medium"

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_invalid_json_triggers_fallback(self, mock_getenv, mock_post):
        mock_post.return_value = _mock_groq_response("not valid json {{{")
        result = call_translation_layer("farmer", SAMPLE_INPUT)
        assert result["_is_fallback"] is True

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_api_down_triggers_fallback(self, mock_getenv, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")
        result = call_translation_layer("farmer", SAMPLE_INPUT)
        assert result["_is_fallback"] is True

    @patch("ai_layer.translator.os.getenv", return_value=None)
    def test_no_api_key_raises(self, mock_getenv):
        with pytest.raises(TranslationError):
            call_translation_layer("farmer", SAMPLE_INPUT)

    def test_old_indonesian_role_key_now_unsupported(self):
        # "petani" was Week 1's role key — it's intentionally gone now.
        with pytest.raises(TranslationError):
            call_translation_layer("petani", SAMPLE_INPUT)