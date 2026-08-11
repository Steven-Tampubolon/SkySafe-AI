"""
ingestion.py — Modul pengambilan data cuaca antariksa dari NOAA SWPC & NASA DONKI.
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("skysafe.ingestion")
logging.basicConfig(level=logging.INFO)

NOAA_KP_REALTIME_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_KP_FORECAST_URL = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"
NASA_DONKI_FLR_URL = "https://api.nasa.gov/DONKI/FLR"

REQUEST_TIMEOUT = 10  # detik
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "last_conditions.json"


class IngestionError(Exception):
    """Dilempar kalau semua sumber data gagal DAN tidak ada cache fallback."""
    pass


def _get_nasa_api_key() -> str:
    key = os.getenv("NASA_DONKI_API_KEY")
    if not key:
        raise IngestionError("NASA_DONKI_API_KEY tidak ditemukan di .env")
    return key


def fetch_kp_index() -> dict:
    """
    Ambil Kp index real-time + forecast 3 hari dari NOAA SWPC.

    Returns:
        {
          "kp_index": int,               # nilai Kp real-time terbaru
          "kp_forecast": [{"time": ISO8601, "kp": float}, ...]
        }
    """
    kp_index = _fetch_kp_realtime()
    kp_forecast = _fetch_kp_forecast()
    return {"kp_index": kp_index, "kp_forecast": kp_forecast}


def _fetch_kp_realtime() -> int:
    try:
        resp = requests.get(NOAA_KP_REALTIME_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Gagal fetch Kp real-time: {e}")
        raise IngestionError(f"Gagal fetch Kp real-time: {e}") from e

    # Format asli NOAA: list of dict, BUKAN list-of-lists dengan header row.
    # Contoh: [{"time_tag": "...", "Kp": 1.33, "a_running": 5, "station_count": 8}, ...]
    if not data:
        raise IngestionError("Response Kp real-time NOAA kosong/tidak sesuai format")

    last_row = data[-1]
    try:
        kp_raw = float(last_row["Kp"])
    except (KeyError, TypeError, ValueError) as e:
        raise IngestionError(f"Tidak bisa parse nilai Kp dari response: {last_row}") from e

    return round(kp_raw)


def _fetch_kp_forecast() -> list:
    """
    Parse file teks forecast 3 hari NOAA. Formatnya semi-tabular dan BISA
    berubah sewaktu-waktu (ini API publik pemerintah) — kalau parsing gagal,
    kembalikan list kosong (bukan crash), supaya kp_index real-time tetap
    bisa dipakai.
    """
    try:
        resp = requests.get(NOAA_KP_FORECAST_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
    except requests.exceptions.RequestException as e:
        logger.warning(f"Gagal fetch Kp forecast: {e}")
        return []

    try:
        return _parse_kp_forecast_text(text)
    except Exception as e:
        logger.warning(f"Gagal parse Kp forecast text, kembalikan list kosong: {e}")
        return []


def _parse_kp_forecast_text(text: str) -> list:
    """
    Parse baris seperti:
        00-03UT       2.33         2.00         3.33
        06-09UT       2.33         5.00 (G1)    2.67
    Label "(Gx)" HANYA muncul kalau Kp >= 5 (G1+), jadi jangan diwajibkan ada
    di semua kolom — buang saja anotasinya, lalu ambil 3 angka numerik pertama
    yang tersisa per baris.
    """
    lines = text.splitlines()

    date_line = None
    for line in lines:
        if re.search(r"[A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{1,2}", line):
            date_line = line
            break

    dates = re.findall(r"[A-Za-z]{3}\s+\d{1,2}", date_line) if date_line else []

    row_start_pattern = re.compile(r"^(\d{2})-(\d{2})UT\s+(.*)$")
    year = datetime.now(timezone.utc).year
    forecast = []

    for raw_line in lines:
        m = row_start_pattern.match(raw_line.strip())
        if not m:
            continue

        start_hour = m.group(1)
        rest = m.group(3)
        rest_clean = re.sub(r"\(G\d\)", "", rest)  # buang label G-scale kalau ada

        kp_values = []
        for tok in rest_clean.split():
            try:
                kp_values.append(float(tok))
            except ValueError:
                continue
            if len(kp_values) == 3:
                break

        for i, kp_val in enumerate(kp_values):
            if i >= len(dates):
                continue
            try:
                dt = datetime.strptime(f"{dates[i]} {year}", "%b %d %Y")
                dt = dt.replace(hour=int(start_hour), tzinfo=timezone.utc)
                forecast.append({"time": dt.isoformat(), "kp": kp_val})
            except ValueError:
                continue

    return forecast

def fetch_flare_data(days_back: int = 3) -> dict:
    """
    Ambil flare terbaru dari NASA DONKI dalam rentang `days_back` hari terakhir.

    Returns:
        {"flare_class": str | None, "flare_time": str | None}
        flare_class = None kalau tidak ada flare (akan dikonversi jadi
        "Tidak ada" di get_current_conditions() sesuai konvensi skema AI layer).
    """
    from datetime import timedelta

    api_key = _get_nasa_api_key()
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "api_key": api_key,
    }

    try:
        resp = requests.get(NASA_DONKI_FLR_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        flares = resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Gagal fetch data flare DONKI: {e}")
        raise IngestionError(f"Gagal fetch data flare DONKI: {e}") from e

    if not flares:
        return {"flare_class": None, "flare_time": None}

    # Ambil flare paling baru berdasarkan peakTime (fallback beginTime)
    def sort_key(f):
        return f.get("peakTime") or f.get("beginTime") or ""

    latest = sorted(flares, key=sort_key)[-1]

    return {
        "flare_class": latest.get("classType"),
        "flare_time": latest.get("peakTime") or latest.get("beginTime"),
    }


def get_current_conditions() -> dict:
    """
    Gabungkan Kp index + flare data jadi satu schema sesuai kontrak dokumen sprint.
    Kalau salah satu sumber gagal total, coba fallback ke cache lokal terakhir.
    """
    try:
        kp_data = fetch_kp_index()
        flare_data = fetch_flare_data()

        result = {
            "kp_index": kp_data["kp_index"],
            "kp_forecast": kp_data["kp_forecast"],
            "flare_class": flare_data["flare_class"] or "Tidak ada",
            "flare_time": flare_data["flare_time"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_cache(result)
        return result

    except IngestionError as e:
        logger.error(f"get_current_conditions gagal total: {e}. Coba fallback cache...")
        cached = _load_cache()
        if cached:
            cached["_from_cache"] = True
            return cached
        raise IngestionError(
            "Semua sumber data gagal dan tidak ada cache fallback tersedia."
        ) from e


def _save_cache(data: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.warning(f"Gagal simpan cache: {e}")


def _load_cache() -> dict | None:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Gagal baca cache: {e}")
        return None


if __name__ == "__main__":
    # Quick manual check (pakai API asli) — jalankan: python -m ingestion.ingestion
    import pprint
    pprint.pprint(get_current_conditions())