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

ALL_ROLES = ["farmer", "surveyor", "ham_radio_operator", "general_public"]

SCENARIO_LOW = {
    "kp_index": 2,
    "solar_flare_class": "C2.4",
    "geomagnetic_latitude_band": "Equatorial/Low",
    "gps_impact_score": 3.0,
    "gps_impact_label": "Low",
    "hf_blackout_risk_label": "Low",
    "data_type": "real-time",
    "confidence_level": "High",
    "confidence_reason": "Data measured directly from NOAA/NASA instruments, not a prediction.",
}

SCENARIO_HIGH = {
    "kp_index": 8,
    "solar_flare_class": "X2.0",
    "geomagnetic_latitude_band": "High-Latitude/Auroral",
    "gps_impact_score": 77.0,
    "gps_impact_label": "High",
    "hf_blackout_risk_label": "High",
    "data_type": "forecast",
    "confidence_level": "Medium",
    "confidence_reason": "12-hour forecast, geomagnetic models carry a reasonable margin of error.",
}


def _build_scenario_input(role: str, scenario: dict) -> dict:
    base = {
        "role": role,
        "location_name": "Test Location",
        "local_time": "August 25, 2026, 09:00 local time",
        "forecast_window": "08:00-16:00 UTC",
        "source_name": "NOAA SWPC",
        "source_url": "https://www.swpc.noaa.gov/products/planetary-k-index",
    }
    base.update(scenario)
    return base


def _mock_output_for(data: dict) -> dict:
    return {
        "headline": "Test headline",
        "plain_explanation": "Test explanation covering the situation.",
        "recommended_action": "Test recommended action.",
        "confidence_label": data["confidence_level"],
        "why_confidence": "Test reasoning.",
        "source_citation": f"{data['source_name']} — {data['source_url']}",
    }


class TestAllRolesAllScenarios:
    """Closes the Week 2 coverage gap: at least 2 scenarios (Low, High)
    per role, all 4 roles, all returning valid English JSON that passes
    validation. 4 roles x 2 scenarios = 8 parametrized cases, matching the
    original Week 2 Monday DoD."""

    @pytest.mark.parametrize("role", ALL_ROLES)
    @pytest.mark.parametrize("scenario_name,scenario", [("low", SCENARIO_LOW), ("high", SCENARIO_HIGH)])
    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_role_scenario_combination(self, mock_getenv, mock_post, role, scenario_name, scenario):
        data = _build_scenario_input(role, scenario)
        mock_post.return_value = _mock_groq_response(_mock_output_for(data))

        result = call_translation_layer(role, data)

        assert result["_is_fallback"] is False
        assert result["confidence_label"] == data["confidence_level"]

class TestNormalizeOutput:
    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_source_citation_always_deterministic_format(self, mock_getenv, mock_post):
        """Regression test for the inconsistent citation formatting flagged
        in external Week 2 review (comma vs dash vs no separator)."""
        messy_output = dict(VALID_LLM_OUTPUT)
        messy_output["source_citation"] = "noaa swpc (weird format, no url)"
        mock_post.return_value = _mock_groq_response(messy_output)

        result = call_translation_layer("farmer", SAMPLE_INPUT)

        expected = f"{SAMPLE_INPUT['source_name']} — {SAMPLE_INPUT['source_url']}"
        assert result["source_citation"] == expected

    @patch("ai_layer.translator.requests.post")
    @patch("ai_layer.translator.os.getenv", return_value="fake_groq_key")
    def test_missing_terminal_punctuation_is_fixed(self, mock_getenv, mock_post):
        """Regression test for run-on/missing-period sentences flagged in
        the General Public - High Impact scenario during external review."""
        no_period_output = dict(VALID_LLM_OUTPUT)
        no_period_output["headline"] = "GPS impact expected today"
        mock_post.return_value = _mock_groq_response(no_period_output)

        result = call_translation_layer("farmer", SAMPLE_INPUT)

        assert result["headline"].endswith(".")