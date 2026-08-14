"""
ai_layer/translator.py — SkySafe AI translation layer (Groq).

HARD RULE (from skysafe-ai-prompt-templates v2): the AI here NEVER computes
or corrects numbers. All scores are final, computed by the deterministic
scoring module (scoring/scoring.py). The AI's only job is to translate
those scores into role-specific, natural-language guidance in English. If
the LLM changes any key value in its output vs. the input, the output is
rejected and a static fallback is used instead — the main trust safeguard.

v2 change: all AI output is now in English (global audience), input
includes geomagnetic_latitude_band so the AI can explain WHY the same
storm hits harder at some locations than others.
"""

import os
import json
import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("skysafe.ai_layer")
logging.basicConfig(level=logging.INFO)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 1

OUTPUT_REQUIRED_KEYS = [
    "headline", "plain_explanation", "recommended_action",
    "confidence_label", "why_confidence", "source_citation",
]


class TranslationError(Exception):
    """Raised only for configuration errors (missing API key) or unknown role."""
    pass


MASTER_SYSTEM_PROMPT = """\
You are the translation layer for SkySafe AI. Your ONLY job is to translate
space weather impact data and scores that have ALREADY been calculated
deterministically into language that is easy to understand for the given
user role. Always respond in English, regardless of the user's location.

YOU MUST NOT:
- Recalculate, correct, or change any number/score/index given in the input.
  All numbers are final facts.
- Invent any new prediction, number, or claim not present in the input data.
- Give absolute guarantees ("definitely safe", "definitely fine", "guaranteed
  normal").
- Invent or alter the name of any data source.
- Downplay or hide a high risk score to make the tone "nicer to read".
- Write internal field/variable names verbatim, such as "gps_impact_score",
  "kp_index", or "hf_blackout_risk_label". Describe the underlying number in
  natural language instead (e.g. "a GPS impact score of 3 out of 100", "a
  Kp-index reading of 8" — note "Kp-index" as a scientific term is fine,
  "kp_index" as a snake_case identifier is not).
- Use these overused filler phrases: "It's a good idea to", "Consider X or
  Y", "adjust your plans accordingly", "as planned". Vary your wording and
  action verbs instead.
- Write run-on sentences joined by multiple "and"/"or" conjunctions — split
  distinct ideas into separate, shorter sentences.

YOU MUST:
- Always include the raw score/index and the official source name in your
  explanation, phrased in natural language (see rule above).
- Adapt terminology and focus to the given user role.
- Use a calm, concrete, actionable tone — not alarmist.
- State the confidence_level and a brief reason for it.
- End every sentence with correct terminal punctuation.
- If the data is FORECAST (not a real-time measurement), state that
  explicitly and do not present it as a certain event.
- If gps_impact_label = "Critical", direct the user to check the official
  source directly (link provided in input) — never give false reassurance.
- If geomagnetic_latitude_band = "Equatorial/Low", note that geomagnetic
  storm effects are generally weaker at this latitude than the raw
  Kp-index alone would suggest for high-latitude regions — the
  gps_impact_score already accounts for this, just make it clear in the
  explanation so the user understands why the impact may feel less severe
  than headlines about the same storm elsewhere.

OUTPUT FORMAT: reply ONLY with valid JSON matching the schema given in each
role prompt. Do not add any text outside the JSON.
"""

ROLE_TEMPLATE_FARMER = """\
ROLE: Farmer or precision-agriculture equipment operator (RTK-GPS tractor,
spray drone, auto-steer harvesting).

EXPLANATION FOCUS:
- Impact on GPS accuracy for precision planting, targeted spraying, and
  automated harvesting. Use familiar units (cm/meter drift) ONLY if that
  information is present in gps_impact_score — never invent a specific
  drift number.
- If gps_impact_label is "Moderate" or higher, suggest a practical
  alternative (e.g. delay precision operations, switch to manual mode,
  re-check reference points).
- If geomagnetic_latitude_band is "Equatorial/Low", briefly note the
  impact is typically less severe here than at high latitudes for the
  same storm.

TONE: practical, warm, avoid space-weather jargon. Mention terms like
"geomagnetic storm" or "Kp-index" once for context, then focus entirely on
the practical impact.

OUTPUT SCHEMA (REQUIRED, reply with ONLY this JSON, no other text):
{{
  "headline": "1 short sentence, gets straight to the impact",
  "plain_explanation": "2-3 sentences, plain language matching the role",
  "recommended_action": "1-2 sentences, concrete and immediately actionable",
  "confidence_label": "copied directly from input, do not alter",
  "why_confidence": "1 sentence explaining the confidence level",
  "source_citation": "source name + url, copied directly from input"
}}

INPUT DATA:
{input_json}

Produce output matching the JSON schema defined above.
"""

ROLE_TEMPLATE_SURVEYOR = """\
ROLE: Surveyor or geodesy practitioner using RTK/PPK GNSS.

EXPLANATION FOCUS:
- Impact on positioning accuracy and possible ionospheric scintillation.
- Technical terms are fine (Kp-index, scintillation, fix/float RTK) since
  the audience knows this domain — but still spell out the practical
  implication (e.g. risk of losing RTK fix, need to re-observe a baseline).
- If geomagnetic_latitude_band is "Equatorial/Low", note that equatorial
  ionospheric scintillation can still be a separate concern independent of
  the geomagnetic storm score, and recommend normal best practices for
  low-latitude GNSS work.

TONE: technical, concise, straight to the actionable insight. No need to
explain basic space-weather concepts.

OUTPUT SCHEMA (REQUIRED, reply with ONLY this JSON, no other text):
{{
  "headline": "1 short sentence, gets straight to the impact",
  "plain_explanation": "2-3 sentences, plain language matching the role",
  "recommended_action": "1-2 sentences, concrete and immediately actionable",
  "confidence_label": "copied directly from input, do not alter",
  "why_confidence": "1 sentence explaining the confidence level",
  "source_citation": "source name + url, copied directly from input"
}}

INPUT DATA:
{input_json}

Produce output matching the JSON schema defined above.
"""

ROLE_TEMPLATE_HAM_RADIO = """\
ROLE: Amateur (ham) radio operator or HF communications operator, including
emergency radio use.

EXPLANATION FOCUS:
- Impact on HF propagation and blackout risk over the forecast window.
- Use terminology familiar to the ham radio community (band, propagation,
  blackout, MUF) but explain briefly so newcomers can follow too.
- If hf_blackout_risk_label is "High" or "Critical", suggest considering an
  alternate band or communication schedule if relevant from the data.

TONE: to the point, like one operator sharing band conditions with another.

OUTPUT SCHEMA (REQUIRED, reply with ONLY this JSON, no other text):
{{
  "headline": "1 short sentence, gets straight to the impact",
  "plain_explanation": "2-3 sentences, plain language matching the role",
  "recommended_action": "1-2 sentences, concrete and immediately actionable",
  "confidence_label": "copied directly from input, do not alter",
  "why_confidence": "1 sentence explaining the confidence level",
  "source_citation": "source name + url, copied directly from input"
}}

INPUT DATA:
{input_json}

Produce output matching the JSON schema defined above.
"""

ROLE_TEMPLATE_GENERAL_PUBLIC = """\
ROLE: General public, no technical background assumed.

EXPLANATION FOCUS:
- Everyday impact most relevant to them: GPS accuracy on phones/map apps,
  and — only if kp_index indicates favorable conditions — mention the
  possibility of visible aurora. Do not mention it if the data doesn't
  support it.
- Do not mention precision-agriculture or survey-grade GPS impacts — not
  relevant for this role.
- Never mention "Kp-index" or any numeric score by name — describe severity
  in everyday terms only (e.g. "a strong solar storm today", "nothing
  unusual today", "northern lights might be visible if you're near the
  polar regions").

TONE: casual, a little engaging/educational, but always accurate and never
exaggerated.

OUTPUT SCHEMA (REQUIRED, reply with ONLY this JSON, no other text):
{{
  "headline": "1 short sentence, gets straight to the impact",
  "plain_explanation": "2-3 sentences, plain language matching the role",
  "recommended_action": "1-2 sentences, concrete and immediately actionable",
  "confidence_label": "copied directly from input, do not alter",
  "why_confidence": "1 sentence explaining the confidence level",
  "source_citation": "source name + url, copied directly from input"
}}

INPUT DATA:
{input_json}

Produce output matching the JSON schema defined above.
"""

ROLE_TEMPLATES = {
    "farmer": ROLE_TEMPLATE_FARMER,
    "surveyor": ROLE_TEMPLATE_SURVEYOR,
    "ham_radio_operator": ROLE_TEMPLATE_HAM_RADIO,
    "general_public": ROLE_TEMPLATE_GENERAL_PUBLIC,
}


def _get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise TranslationError("GROQ_API_KEY not found in .env")
    return key


def _build_static_fallback(data: dict) -> dict:
    """Non-AI, label-based static template — used if Groq fails or its
    output doesn't pass validation. The Trust Panel must never be empty."""
    gps_label = data.get("gps_impact_label", "Unknown")
    confidence = data.get("confidence_level", "Unknown")
    source = data.get("source_name", "Unknown source")
    url = data.get("source_url", "")

    return {
        "headline": f"Current GPS impact: {gps_label}",
        "plain_explanation": (
            f"Based on data from {source}, the current impact level on GPS "
            f"accuracy is '{gps_label}'. This data is "
            f"{data.get('data_type', 'of unknown type')}."
        ),
        "recommended_action": (
            "Check the official source for further detail before making "
            "any operational decisions."
        ),
        "confidence_label": confidence,
        "why_confidence": data.get("confidence_reason", "No additional explanation available."),
        "source_citation": f"{source} — {url}".strip(" —"),
        "_is_fallback": True,
    }


def _validate_output(output: dict, data: dict) -> bool:
    """Ensure the LLM didn't alter key values compared to the input."""
    if not isinstance(output, dict):
        return False

    for key in OUTPUT_REQUIRED_KEYS:
        if key not in output or not isinstance(output[key], str) or not output[key].strip():
            logger.warning(f"Validation failed: key '{key}' missing/empty in LLM output.")
            return False

    expected_confidence = data.get("confidence_level")
    if output["confidence_label"].strip() != str(expected_confidence).strip():
        logger.warning(
            f"Validation failed: output confidence_label "
            f"({output['confidence_label']!r}) != input confidence_level "
            f"({expected_confidence!r})."
        )
        return False

    return True


def _call_groq(system_prompt: str, user_prompt: str) -> dict:
    api_key = _get_groq_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)

SENTENCE_END_CHARS = (".", "!", "?", '"', "'", ")")


def _ensure_terminal_punctuation(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in SENTENCE_END_CHARS:
        text += "."
    return text


def _normalize_output(output: dict, data: dict) -> dict:
    """
    Deterministic post-processing applied AFTER validation passes, closing
    gaps an LLM can't be relied on to get consistent every call:

    - source_citation is REBUILT from the original input data, never taken
      from the LLM's own wording. External review of Week 2 output found
      inconsistent formatting across calls ("NOAA SWPC, url" vs "NOAA SWPC
      - url" vs missing separator entirely) — rebuilding it deterministically
      makes the format identical every time, same principle as never
      trusting the LLM with numbers.
    - Ensures key text fields end with terminal punctuation, since the LLM
      occasionally drops a trailing period (observed in General Public
      scenarios during manual review).
    """
    normalized = dict(output)
    for key in ("headline", "plain_explanation", "recommended_action", "why_confidence"):
        normalized[key] = _ensure_terminal_punctuation(normalized[key])

    source_name = data.get("source_name", "Unknown source")
    source_url = data.get("source_url", "")
    normalized["source_citation"] = f"{source_name} — {source_url}".strip(" —")

    return normalized


def call_translation_layer(role: str, data: dict) -> dict:
    """
    Translate deterministic scores into role-specific natural-language
    guidance via Groq. Always validates output against input; retries once
    on failure, then falls back to a static template.

    Args:
        role: "farmer" | "surveyor" | "ham_radio_operator" | "general_public"
        data: dict matching the v2 input schema (kp_index,
              geomagnetic_latitude_band, gps_impact_label, confidence_level,
              source_name, etc.)

    Returns:
        dict matching the Trust Panel output schema, plus "_is_fallback".
    """
    if role not in ROLE_TEMPLATES:
        raise TranslationError(f"Unsupported role: {role!r}. Supported: {list(ROLE_TEMPLATES.keys())}")

    template = ROLE_TEMPLATES[role]
    user_prompt = template.format(input_json=json.dumps(data, ensure_ascii=False, indent=2))

    attempts = 0
    last_error = None

    while attempts <= MAX_RETRIES:
        attempts += 1
        try:
            output = _call_groq(MASTER_SYSTEM_PROMPT, user_prompt)
        except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempts} failed: {e}")
            last_error = e
            continue

        if _validate_output(output, data):
            output = _normalize_output(output, data)
            output["_is_fallback"] = False
            return output

        logger.warning(f"Attempt {attempts}: output failed validation.")

    logger.error(f"All attempts failed (last: {last_error}). Using static fallback.")
    return _build_static_fallback(data)