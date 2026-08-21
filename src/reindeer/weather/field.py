"""Phase 6: a REAL, spatially-varying per-cell weather field (the reconstruction).

The v1 scorer took ONE area reading and *synthesised* a per-cell field with fixed
constants — a 6.5 C/km lapse rate (wrong sign under autumn inversions) and a TPI
wind proxy, and it never fetched wind direction at all. This module replaces that
synthetic downscaling with actual data:

  1. build a lattice of sample points over the grid bbox (a few km apart);
  2. fetch REAL weather at every lattice point from Open-Meteo — temperature, wind
     speed, **wind direction**, precipitation — using MET Norway's 1 km metno_nordic
     model for forecasts and the ERA5 archive for historical validation dates;
  3. interpolate that lattice to every 250 m grid cell:
       - wind      -> interpolate the u/v VECTOR components (so 350 deg and 10 deg
                      average to 0, not 180), recover per-cell speed + direction;
       - temperature -> fit a DATA-DRIVEN lapse (T ~ a + b*elevation) from the real
                      lattice, apply it to each cell's real DTM elevation, plus an
                      IDW residual for horizontal structure. b is whatever the data
                      says — negative (normal) or positive (a valley inversion), so
                      the inversion case is handled by the data, not a fixed sign;
       - precipitation -> inverse-distance weighting.

The result is a `WeatherField`: per-cell arrays the scorer consumes directly. No
API key, no sign-up; Open-Meteo data is free for non-commercial use (CC-BY 4.0:
MET Norway for metno_nordic, Copernicus ERA5 for the archive).

Daytime aggregation (06-18 local) matches weather/forecast.py & historical.py, so a
forecast field and a historical field are built the same way and feed one scorer.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "reindeer-heatmap/0.2 (reindeer movement-ecology research)"
_HOURLY = "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation"

_ROOT = Path(__file__).resolve().parents[3]
CACHE = _ROOT / "data" / "raw" / "wfield"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


@dataclass
class PointObs:
    """One lattice point's daytime-aggregated weather + its model elevation."""
    lat: float
    lon: float
    elev_m: float
    temp_c: float
    wind_ms: float
    wind_dir_deg: float   # meteorological "from" direction
    precip_mm: float


@dataclass
class WeatherField:
    """Per-cell weather arrays (aligned to the cells passed to build_field).

    Every array has one entry per grid cell. This is the real, spatially varying
    replacement for the old single WeatherDay + synthetic downscaling.
    """
    temp_c: np.ndarray
    wind_ms: np.ndarray
    wind_dir_deg: np.ndarray   # meteorological "from" direction, per cell
    precip_mm: np.ndarray
    lapse_c_per_m: float       # the data-driven lapse actually used (sign is honest)

    def area_summary(self) -> tuple[float, float, float]:
        """(median temp, median wind, mean precip) over the field — for reporting."""
        return (float(np.nanmedian(self.temp_c)),
                float(np.nanmedian(self.wind_ms)),
                float(np.nanmean(self.precip_mm)))


# --- wind vector helpers (meteorological "from" convention) ----------------------
def wind_uv(speed, dir_deg):
    """(speed, from-direction deg) -> (u, v) 'toward' components. Vector-safe."""
    r = np.radians(np.asarray(dir_deg, float))
    s = np.asarray(speed, float)
    return -s * np.sin(r), -s * np.cos(r)   # toward-vector


def wind_from_uv(u, v):
    """(u, v) 'toward' components -> (speed, from-direction deg in [0,360))."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    speed = np.hypot(u, v)
    dir_deg = (np.degrees(np.arctan2(-u, -v))) % 360.0
    return speed, dir_deg


def circular_mean_dir(dirs, speeds=None):
    """Circular (vector) mean of 'from' directions, optionally speed-weighted."""
    dirs = np.asarray(dirs, float)
    if len(dirs) == 0:
        return 0.0
    w = np.ones_like(dirs) if speeds is None else np.asarray(speeds, float)
    u, v = wind_uv(np.ones_like(dirs), dirs)
    _, d = wind_from_uv(float(np.average(u, weights=_safe_w(w))),
                        float(np.average(v, weights=_safe_w(w))))
    return float(d)


def _safe_w(w):
    w = np.asarray(w, float)
    return w if w.sum() > 0 else np.ones_like(w)


# --- lattice construction --------------------------------------------------------
def build_lattice(east: np.ndarray, north: np.ndarray, spacing_m: float):
    """Regular lattice of sample points over the bbox of the given cells.

    Returns (pt_east, pt_north, lat, lon) arrays. Snapped to the bbox with a one-cell
    margin so every grid cell is inside the lattice hull (no extrapolation).
    """
    e0, e1 = float(np.min(east)) - spacing_m, float(np.max(east)) + spacing_m
    n0, n1 = float(np.min(north)) - spacing_m, float(np.max(north)) + spacing_m
    ex = np.arange(e0, e1 + 1, spacing_m)
    nx = np.arange(n0, n1 + 1, spacing_m)
    EE, NN = np.meshgrid(ex, nx)
    pe, pn = EE.ravel(), NN.ravel()
    lon, lat = _to_wgs.transform(pe, pn)
    return pe, pn, np.asarray(lat), np.asarray(lon)


# --- daytime aggregation of one point's hourly series ----------------------------
def _aggregate_point(times, temp, wind, wdir, precip, date: str,
                     day_start=6, day_end=18):
    idx = [i for i, ts in enumerate(times)
           if ts[:10] == date and day_start <= int(ts[11:13]) <= day_end]
    if not idx:
        return None
    T = [temp[i] for i in idx if temp[i] is not None]
    W = [wind[i] for i in idx if wind[i] is not None]
    D = [wdir[i] for i in idx if wdir[i] is not None]
    P = [precip[i] for i in idx if precip[i] is not None]
    if not T:
        return None
    return (round(sum(T) / len(T), 2),
            round(sum(W) / len(W), 2) if W else 0.0,
            circular_mean_dir(D, W if len(W) == len(D) else None) if D else 0.0,
            round(sum(P), 2) if P else 0.0)


# --- fetch a whole lattice in one request (cached) -------------------------------
def _get_json(session, api, params, tries=6):
    """GET with polite exponential backoff on 429/5xx (Open-Meteo rate limits the
    heavier multi-point archive calls). Honours Retry-After when present."""
    delay = 5.0
    for attempt in range(tries):
        r = session.get(api, params=params, headers={"User-Agent": USER_AGENT}, timeout=120)
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt == tries - 1:
                r.raise_for_status()
            wait = float(r.headers.get("Retry-After", delay))
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def _cache_path(source: str, date: str, lat, lon) -> Path:
    key = hashlib.md5(
        f"{source}|{date}|{lat[0]:.3f},{lon[0]:.3f}|{len(lat)}|"
        f"{lat[-1]:.3f},{lon[-1]:.3f}".encode()).hexdigest()[:16]
    return CACHE / f"{source}_{date}_{key}.json"


def fetch_lattice(source: str, date: str, lat, lon,
                  session: requests.Session | None = None) -> list[PointObs]:
    """Fetch daytime weather at every lattice point for one date. Cached on disk.

    source='forecast' -> metno_nordic (1 km); source='archive' -> ERA5 (validation).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(source, date, lat, lon)
    if cache.exists() and cache.stat().st_size > 200:
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        s = session or requests.Session()
        lats = ",".join(f"{x:.4f}" for x in lat)
        lons = ",".join(f"{x:.4f}" for x in lon)
        if source == "forecast":
            api, params = FORECAST_API, {"latitude": lats, "longitude": lons,
                                         "hourly": _HOURLY, "models": "metno_nordic",
                                         "wind_speed_unit": "ms",
                                         "timezone": "Europe/Oslo", "forecast_days": 3}
        else:
            api, params = ARCHIVE_API, {"latitude": lats, "longitude": lons,
                                        "hourly": _HOURLY, "wind_speed_unit": "ms",
                                        "timezone": "Europe/Oslo",
                                        "start_date": date, "end_date": date}
        raw = _get_json(s, api, params)
        cache.write_text(json.dumps(raw), encoding="utf-8")
        time.sleep(8.0)  # polite base throttle to stay under the per-minute weighted limit

    locs = raw if isinstance(raw, list) else [raw]
    obs: list[PointObs] = []
    for loc in locs:
        h = loc.get("hourly")
        if not h:
            continue
        agg = _aggregate_point(h["time"], h["temperature_2m"], h["wind_speed_10m"],
                               h["wind_direction_10m"], h["precipitation"], date)
        if agg is None:
            continue
        t, w, d, p = agg
        obs.append(PointObs(loc["latitude"], loc["longitude"],
                            float(loc.get("elevation", np.nan)), t, w, d, p))
    if not obs:
        raise ValueError(f"no daytime lattice data for {date} ({source})")
    return obs


def available_forecast_dates(lat, lon, session=None) -> list[str]:
    """Dates (YYYY-MM-DD) present in a metno_nordic forecast at the first point."""
    s = session or requests.Session()
    r = s.get(FORECAST_API, params={"latitude": f"{lat[0]:.4f}", "longitude": f"{lon[0]:.4f}",
                                     "hourly": "temperature_2m", "models": "metno_nordic",
                                     "timezone": "Europe/Oslo", "forecast_days": 3},
              headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    times = r.json()["hourly"]["time"]
    return sorted({t[:10] for t in times})


# --- interpolation: lattice -> every cell ----------------------------------------
def _idw_weights(cell_e, cell_n, pt_e, pt_n, power=2.0, eps=1.0):
    """Inverse-distance weights, shape (n_cells, n_pts). Vectorised."""
    dx = cell_e[:, None] - pt_e[None, :]
    dy = cell_n[:, None] - pt_n[None, :]
    d2 = dx * dx + dy * dy
    w = 1.0 / np.power(d2 + eps * eps, power / 2.0)
    return w / w.sum(axis=1, keepdims=True)


def interpolate_field(obs: list[PointObs], pt_e, pt_n,
                      cell_e, cell_n, cell_elev) -> WeatherField:
    """Interpolate lattice PointObs onto grid cells -> WeatherField."""
    pt_e = np.asarray(pt_e, float)
    pt_n = np.asarray(pt_n, float)
    cell_e = np.asarray(cell_e, float)
    cell_n = np.asarray(cell_n, float)
    cell_elev = np.asarray(cell_elev, float)

    Tp = np.array([o.temp_c for o in obs])
    Wp = np.array([o.wind_ms for o in obs])
    Dp = np.array([o.wind_dir_deg for o in obs])
    Pp = np.array([o.precip_mm for o in obs])
    Ep = np.array([o.elev_m for o in obs])

    W = _idw_weights(cell_e, cell_n, pt_e, pt_n)

    # temperature: data-driven lapse fit (T = a + b*elev) over lattice points, then
    # apply to each cell's real DTM elevation + IDW of the fit residual.
    good = np.isfinite(Ep) & np.isfinite(Tp)
    if good.sum() >= 3 and np.ptp(Ep[good]) > 50:
        b, a = np.polyfit(Ep[good], Tp[good], 1)   # b = lapse (degC per m), signed
    else:
        b, a = 0.0, float(np.nanmean(Tp))
    resid = Tp - (a + b * Ep)
    temp = a + b * cell_elev + (W * resid[None, :]).sum(axis=1)

    # wind: interpolate the u/v vector components, recover speed + direction.
    up, vp = wind_uv(Wp, Dp)
    u = (W * up[None, :]).sum(axis=1)
    v = (W * vp[None, :]).sum(axis=1)
    wind, wdir = wind_from_uv(u, v)

    precip = np.clip((W * Pp[None, :]).sum(axis=1), 0.0, None)

    return WeatherField(temp_c=temp, wind_ms=np.clip(wind, 0.0, None),
                        wind_dir_deg=wdir, precip_mm=precip, lapse_c_per_m=float(b))


# --- one-call convenience --------------------------------------------------------
def build_field(source: str, date, cell_e, cell_n, cell_elev,
                spacing_m: float | None = None,
                session: requests.Session | None = None) -> WeatherField:
    """Fetch + interpolate a real per-cell weather field for `date`.

    source='forecast' (metno_nordic 1 km) or 'archive' (ERA5). spacing defaults to
    4 km for forecasts (fine model) and 6 km for the archive (ERA5 is coarse anyway).
    """
    date = date.isoformat() if isinstance(date, dt.date) else str(date)
    if spacing_m is None:
        # forecast: a fine mesoscale model, sample densely. archive: ERA5 is coarse
        # (~25 km) and its multi-point calls are rate-limited, so sample lightly.
        # Both spacings live in config/area.json.
        from reindeer.area import AREA
        spacing_m = (AREA.weather_lattice_forecast_m if source == "forecast"
                     else AREA.weather_lattice_archive_m)
    cell_e = np.asarray(cell_e, float)
    cell_n = np.asarray(cell_n, float)
    pe, pn, lat, lon = build_lattice(cell_e, cell_n, spacing_m)
    obs = fetch_lattice(source, date, lat, lon, session=session)
    # keep only lattice points that came back, matched to their planned e/n by index
    # (Open-Meteo returns points in request order); align by nearest to be safe.
    ope = np.array([_nearest_e(o, pe, pn, lat, lon) for o in obs])
    opn = np.array([_nearest_n(o, pe, pn, lat, lon) for o in obs])
    return interpolate_field(obs, ope, opn, cell_e, cell_n, cell_elev)


def _nearest_e(o: PointObs, pe, pn, lat, lon):
    i = int(((lat - o.lat) ** 2 + (lon - o.lon) ** 2).argmin())
    return pe[i]


def _nearest_n(o: PointObs, pe, pn, lat, lon):
    i = int(((lat - o.lat) ** 2 + (lon - o.lon) ** 2).argmin())
    return pn[i]
