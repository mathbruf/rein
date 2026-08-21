"""Phase 5: historical daily weather from the Open-Meteo ERA5 archive (no API key).

CLAUDE.md names MET Frost for historical weather; we use Open-Meteo's free archive
(ERA5 reanalysis) instead so validation isn't blocked on a Frost client ID, and
because gridded ERA5 sampled at the field is a defensible spatial match for this
remote alpine field (the nearest Frost station is a valley site far away and much
lower). Frost remains a documented upgrade path.

Daytime aggregation (06-18 local) mirrors weather/forecast.py, so a historical
WeatherDay and a forecast WeatherDay are computed the same way and feed the same
scorer.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import requests

from reindeer.model.score import WeatherDay

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "reindeer-heatmap/0.1 (reindeer movement-ecology research)"

_ROOT = Path(__file__).resolve().parents[3]
CACHE = _ROOT / "data" / "raw" / "archive"


def fetch_archive(lat: float, lon: float, start: str, end: str,
                  session: requests.Session | None = None) -> dict:
    """Fetch hourly temp/wind/precip for [start, end] at (lat, lon); cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    if cache.exists() and cache.stat().st_size > 200:
        return json.loads(cache.read_text(encoding="utf-8"))
    s = session or requests.Session()
    r = s.get(ARCHIVE, params={
        "latitude": round(lat, 3), "longitude": round(lon, 3),
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "wind_speed_unit": "ms", "timezone": "Europe/Oslo",
    }, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo archive error: {data}")
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def weather_by_date(archive: dict, day_start: int = 6, day_end: int = 18) -> dict[str, WeatherDay]:
    """Aggregate the archive into {YYYY-MM-DD: WeatherDay} over daytime hours."""
    h = archive["hourly"]
    buckets: dict[str, list] = {}
    for i, ts in enumerate(h["time"]):
        hr = int(ts[11:13])
        if not (day_start <= hr <= day_end):
            continue
        buckets.setdefault(ts[:10], []).append(
            (h["temperature_2m"][i], h["wind_speed_10m"][i], h["precipitation"][i]))
    out: dict[str, WeatherDay] = {}
    for date, rows in buckets.items():
        temps = [t for t, _, _ in rows if t is not None]
        winds = [w for _, w, _ in rows if w is not None]
        precip = sum(p for _, _, p in rows if p is not None)
        if not temps:
            continue
        out[date] = WeatherDay(
            temp_c=round(sum(temps) / len(temps), 1),
            wind_ms=round(sum(winds) / len(winds), 1) if winds else 0.0,
            precip_mm=round(precip, 1))
    return out


def day_weather(archive: dict, date, day_start: int = 6, day_end: int = 18) -> WeatherDay:
    key = date.isoformat() if isinstance(date, dt.date) else str(date)
    wd = weather_by_date(archive, day_start, day_end).get(key)
    if wd is None:
        raise ValueError(f"no archive data for {key}")
    return wd
