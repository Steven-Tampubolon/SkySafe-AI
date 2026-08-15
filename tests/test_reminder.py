"""
tests/test_reminder.py — Unit tests for reminder.generate_reminder_ics():
RFC 5545 structure (delegated to the icalendar library), recurrence rule,
alarm timing, deep link encoding, and timezone auto-detection.
"""

from datetime import timedelta
from unittest.mock import patch

from icalendar import Calendar

from reminder import generate_reminder_ics


def _get_vevent(ics_bytes: bytes):
    cal = Calendar.from_ical(ics_bytes)
    return next(c for c in cal.walk() if c.name == "VEVENT")


class TestGenerateReminderIcs:
    def test_produces_valid_ics(self):
        ics_bytes = generate_reminder_ics("Nairobi, Kenya", "farmer", -1.28, 36.82)
        assert Calendar.from_ical(ics_bytes) is not None  # raises if malformed

    def test_daily_recurrence_rule(self):
        event = _get_vevent(generate_reminder_ics("Nairobi, Kenya", "farmer", -1.28, 36.82))
        assert event["rrule"]["freq"] == ["DAILY"]

    def test_has_alarm_5_minutes_before(self):
        event = _get_vevent(generate_reminder_ics("Nairobi, Kenya", "farmer", -1.28, 36.82))
        alarms = [c for c in event.walk() if c.name == "VALARM"]
        assert len(alarms) == 1
        assert alarms[0]["trigger"].dt.total_seconds() == -300

    def test_deep_link_contains_url_encoded_location_and_role(self):
        event = _get_vevent(generate_reminder_ics("Nairobi, Kenya", "farmer", -1.28, 36.82))
        description = str(event["description"])
        assert "location=Nairobi%2C%20Kenya" in description
        assert "role=farmer" in description

    def test_timezone_detected_for_reykjavik(self):
        # Atlantic/Reykjavik = UTC+0 with no DST, so icalendar (RFC 5545)
        # serializes it as a bare 'Z' (UTC) timestamp — the TZID string
        # is lost on round-trip, but the offset and wall-clock hour
        # are still correct.
        event = _get_vevent(
            generate_reminder_ics("Reykjavik, Iceland", "ham_radio_operator", 64.1466, -21.9426)
        )
        dt = event["dtstart"].dt
        assert dt.utcoffset() == timedelta(0)
        assert dt.hour == 7

    def test_timezone_detected_for_nairobi(self):
        event = _get_vevent(generate_reminder_ics("Nairobi, Kenya", "farmer", -1.28, 36.82))
        assert str(event["dtstart"].dt.tzinfo) == "Africa/Nairobi"

    def test_unknown_coordinates_fall_back_to_utc(self):
        # Modern timezonefinder has full ocean coverage, so it almost
        # never returns None for real coordinates — patch the class
        # method (it's a compiled/read-only slot on the instance) so
        # the `or "UTC"` fallback branch in reminder.py is exercised.
        with patch("reminder.TimezoneFinder.timezone_at", return_value=None):
            event = _get_vevent(generate_reminder_ics("Nowhere", "general_public", 0.0, -150.0))
        dt = event["dtstart"].dt
        assert dt.utcoffset() == timedelta(0)
        assert dt.hour == 7