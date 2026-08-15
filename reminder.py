"""
reminder.py — Generates a personalized, recurring daily .ics calendar
reminder that links back into SkySafe AI. Deliberately has ZERO Streamlit
dependency so it can be imported by both ui/app.py and the test suite
without needing an active Streamlit script run context (mirrors why
pipeline.py is a standalone module, not embedded in ui/app.py).

No backend, no scheduler, no account system needed — this rides on the
calendar app the user already trusts (Google/Apple/Outlook) for the actual
daily notification.

DELIBERATE LIMITATION: the event text is static — "remember to check
SkySafe AI today", not today's actual risk score, since a live score would
require a server-side feed running independently of the user's browser,
which is out of scope for this hackathon. The deep link back into the app
(which pre-fills location + role via query params) is what keeps the
reminder useful day to day instead of going stale.
"""

from datetime import datetime, timedelta
from urllib.parse import quote

import pytz
from icalendar import Alarm, Calendar, Event
from timezonefinder import TimezoneFinder

# TODO: replace with the actual Streamlit Cloud URL after deploying.
APP_BASE_URL = "https://skysafe-ai.streamlit.app"

_tf = TimezoneFinder()  # loads its lookup data once, reused across calls


def generate_reminder_ics(
    location_name: str, role_key: str, lat: float, lng: float, reminder_hour: int = 7
) -> bytes:
    """
    Generate a daily recurring .ics reminder pointing back to SkySafe AI.

    Args:
        location_name: resolved location name (e.g. "Nairobi, Kenya"),
            used both in the deep link and as a human-readable label.
        role_key: internal role key (e.g. "farmer"), embedded in the deep
            link so returning users skip straight to their Trust Panel.
        lat, lng: coordinates used to auto-detect the local timezone via
            timezonefinder, so the reminder fires at a sensible local hour
            without asking the user to pick a timezone manually.
        reminder_hour: local hour (24h) the daily reminder should fire.

    Returns:
        Raw .ics file content as bytes, ready for a Streamlit download
        button. RFC 5545 compliance (line folding, character escaping) is
        handled by the icalendar library — never hand-build this format as
        a plain string.
    """
    tz_name = _tf.timezone_at(lat=lat, lng=lng) or "UTC"
    tz = pytz.timezone(tz_name)

    now = datetime.now(tz)
    dtstart = tz.localize(datetime(now.year, now.month, now.day, reminder_hour, 0))

    cal = Calendar()
    cal.add("prodid", "-//SkySafe AI//Daily Space Weather Check//EN")
    cal.add("version", "2.0")

    event = Event()
    event.add("summary", "Check SkySafe AI — Space Weather Update")

    deep_link = f"{APP_BASE_URL}/?location={quote(location_name)}&role={quote(role_key)}"
    event.add(
        "description",
        "Space weather can affect your work today (GPS accuracy, HF radio, etc).\n\n"
        f"Open your personalized brief: {deep_link}",
    )
    event.add("dtstart", dtstart)
    event.add("dtend", dtstart + timedelta(minutes=15))
    event.add("rrule", {"freq": "daily"})

    uid_slug = location_name.replace(" ", "-").replace(",", "").lower()
    event.add("uid", f"skysafe-{uid_slug}-{role_key}@skysafe.ai")

    alarm = Alarm()
    alarm.add("action", "DISPLAY")
    alarm.add("description", "SkySafe AI daily check")
    alarm.add("trigger", timedelta(minutes=-5))
    event.add_component(alarm)

    cal.add_component(event)
    return cal.to_ical()