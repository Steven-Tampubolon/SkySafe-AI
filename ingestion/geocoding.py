"""
ingestion/geocoding.py — Resolve any place name in the world into
coordinates, using the free Open-Meteo Geocoding API (no API key needed).
Needed so SkySafe AI can classify ANY location on Earth into a geomagnetic
latitude band, instead of assuming a fixed country/region.
"""

import logging

import requests

logger = logging.getLogger("skysafe.geocoding")

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
REQUEST_TIMEOUT = 10


class GeocodingError(Exception):
    """Raised when a location can't be resolved, or the API call fails."""
    pass


def geocode_location(location_name: str) -> dict:
    """
    Resolve a free-text location name (city, region, country — any
    language, any place on Earth) into coordinates.

    Args:
        location_name: e.g. "Nairobi, Kenya", "Fargo North Dakota", "Jakarta".

    Returns:
        {"latitude": float, "longitude": float, "resolved_name": str}

    Raises:
        GeocodingError: location not found, or the API call failed.
    """
    if not location_name or not location_name.strip():
        raise GeocodingError("location_name is empty")

    params = {"name": location_name.strip(), "count": 1, "language": "en", "format": "json"}

    try:
        resp = requests.get(GEOCODING_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        raise GeocodingError(f"Geocoding API failed: {e}") from e

    results = data.get("results")
    if not results:
        raise GeocodingError(f"Location not found: {location_name!r}")

    top = results[0]
    name_parts = [top.get("name", ""), top.get("admin1"), top.get("country")]

    return {
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "resolved_name": ", ".join(p for p in name_parts if p),
    }