"""Phase 4: fetch the daily forecast from MET Locationforecast (api.met.no).

Free, no key, but requires a descriptive User-Agent. v0 uses one area-wide forecast
(sampled at the field centroid) for the whole grid; spatial interpolation + an
elevation lapse-rate are Phase-4 refinements (see IDEAS). A "day" is aggregated over
daytime hours (reindeer move by day): temp = daytime mean, wind = daytime mean,
precip = daytime sum.
"""
from __future__ import annotations

import datetime as dt

import requests

from reindeer.model.score import WeatherDay

API = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
USER_AGENT = "reindeer-heatmap/0.1 (reindeer movement-ecology research)"


def fetch_forecast(lat: float, lon: float, session: requests.Session | None = None) -> dict:
    s = session or requests.Session()
    r = s.get(API, params={"lat": round(lat, 4), "lon": round(lon, 4)},
              headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


def _steps(forecast: dict):
    for e in forecast["properties"]["timeseries"]:
        t = dt.datetime.fromisoformat(e["time"].replace("Z", "+00:00"))
        yield t, e["data"]


def available_dates(forecast: dict) -> list[dt.date]:
    return sorted({t.date() for t, _ in _steps(forecast)})


def day_weather(forecast: dict, date: dt.date,
                day_start: int = 6, day_end: int = 18) -> WeatherDay:
    """Aggregate one day's daytime forecast into a WeatherDay."""
    temps, winds, precip = [], [], 0.0
    for t, data in _steps(forecast):
        if t.date() != date or not (day_start <= t.hour <= day_end):
            continue
        det = data["instant"]["details"]
        if "air_temperature" in det:
            temps.append(det["air_temperature"])
        if "wind_speed" in det:
            winds.append(det["wind_speed"])
        n1 = data.get("next_1_hours", {}).get("details", {})
        if "precipitation_amount" in n1:
            precip += n1["precipitation_amount"]
    if not temps:
        raise ValueError(f"no daytime forecast steps for {date}")
    return WeatherDay(temp_c=round(sum(temps) / len(temps), 1),
                      wind_ms=round(sum(winds) / len(winds), 1),
                      precip_mm=round(precip, 1))


def next_day_weather(forecast: dict, after: dt.date | None = None) -> tuple[dt.date, WeatherDay]:
    """First forecast day strictly after `after` (default: today) that has daytime data."""
    after = after or dt.date.today()
    for d in available_dates(forecast):
        if d <= after:
            continue
        try:
            return d, day_weather(forecast, d)
        except ValueError:
            continue
    raise ValueError("no usable forecast day found")
