# SkySafe AI

Space weather impact translator for precision agriculture, land
surveying, HF radio operators, and the general public — built for [IBM hackathon].

## What it does

SkySafe AI fetches real-time and forecast space weather data (geomagnetic
Kp-index, solar flare class) from NOAA SWPC and NASA DONKI, scores the
practical impact using a **deterministic, rule-based engine** (no AI
involved in the numbers), then translates that score into role-specific,
plain-language guidance via an LLM (Groq) — in English, for a global
audience.

**Core design principle:** the AI layer never computes or invents numbers.
All scores (GPS impact, HF blackout risk, confidence level) come from a
pure-function scoring module based on official NOAA G-scale / R-scale
tables. The LLM's only job is translation and contextual recommendation —
every AI output is validated against the original input before being shown
to the user, and falls back to a static template if validation fails. This
separation is what keeps the system trustworthy and auditable to judges.

**Global-audience calibration:** the official NOAA G-scale table is
calibrated for high-latitude/auroral regions, where geomagnetic storms hit
hardest. The same storm has a much weaker practical GPS impact near the
equator. SkySafe AI geocodes any location on Earth, computes its
**geomagnetic latitude** (not geographic — Earth's magnetic pole is offset
from the geographic pole), classifies it into a latitude band
(High-Latitude/Auroral, Mid-Latitude, Equatorial/Low), and applies a
transparent adjustment factor before scoring — so a farmer in Nairobi
doesn't see the same "Critical" alert as a farmer in Fargo for the same
storm. HF blackout risk is intentionally left un-adjusted, since solar
flares ionize the sunlit hemisphere roughly uniformly, unlike geomagnetic
storms.

## Architecture

Location name → geocoding/ → lat/lon → geomagnetic latitude band
│
NOAA SWPC / NASA DONKI → ingestion/ → scoring/ (latitude-adjusted) → ai_layer/ → Trust Panel (ui/)
(raw data) (fetch+cache) (deterministic) (Groq LLM) (per-role, English)

- **`ingestion/geocoding.py`** — resolves any place name on Earth into
  coordinates via the free Open-Meteo Geocoding API (no API key needed).
- **`ingestion/ingestion.py`** — fetches Kp-index (real-time + 3-day
  forecast) and solar flare data, with error handling and local cache
  fallback if NOAA/NASA is unreachable.
- **`scoring/scoring.py`** — pure functions: Kp-index → geomagnetic
  latitude → latitude-adjusted GPS impact score/label (Low/Moderate/
  High/Critical), flare class → HF blackout risk label, confidence level.
  Fully deterministic, fully unit-tested, every number traceable back to
  official NOAA tables plus a documented latitude adjustment factor.
- **`ai_layer/translator.py`** — sends deterministic scores + role +
  geomagnetic latitude band to Groq (`llama-3.3-70b-versatile`) to
  generate an English natural-language explanation and recommendation for
  one of 4 roles (farmer, surveyor, ham_radio_operator, general_public).
  Validates that the LLM did not alter any key numbers; falls back to a
  static template otherwise.

## Validated against a real historical event

Scoring logic is validated against the **May 2024 "Gannon storm"**
(Kp = 9, G5 Extreme) — the strongest geomagnetic storm in ~20 years,
documented to have disrupted GPS-guided precision agriculture across North
America. See `tests/test_scoring.py::TestValidatesAgainstMay2024Storm` and
`docs/submission-evidence/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in GROQ_API_KEY and NASA_DONKI_API_KEY in .env
```

## Running tests

```bash
python3 -m pytest -v                    # all unit tests (no real API calls)
python3 -m pytest tests/test_scoring.py -v -k may_2024   # historical validation
```

## Running the end-to-end demo (real API calls)

```bash
python3 -m scripts.demo_petani_e2e
```

Fetches live conditions → computes scores → generates AI explanation for
the "petani" (farmer) role, and prints each stage.

## Project status

- [x] Week 1 — Data ingestion + deterministic scoring, validated against
      May 2024 storm, connected to Groq for the "petani" role.
- [x] Week 1.5 — Global-audience revision: English AI output, global
      geocoding, geomagnetic-latitude-adjusted GPS scoring, all 4 roles
      wired up (farmer, surveyor, ham_radio_operator, general_public).
- [ ] Week 2 — Trust Panel UI, connect all roles to the frontend.
- [ ] Week 3 — Polish, submission materials.

## Data sources

- [NOAA SWPC Planetary K-index](https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json)
- [NOAA SWPC 3-Day Forecast](https://services.swpc.noaa.gov/text/3-day-forecast.txt)
- [NASA DONKI Solar Flare API](https://api.nasa.gov/DONKI/FLR)
- [NOAA Space Weather Scales](https://www.swpc.noaa.gov/noaa-scales-explanation)
- [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api) (free, no API key)

## Tech stack

Python 3.12, `requests`, `pytest`, Groq API (OpenAI-compatible,
`llama-3.3-70b-versatile`).