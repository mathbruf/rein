"""Transparent per-report validation: real weather on the exact report date vs the
reported position. One row per sighting, nothing assumed except the geocoding of the
named landmark (reports carry no GPS). Writes data/processed/validation_detail.csv and
prints a readable sample so every data point can be inspected.

For each report:
  exact date -> ERA5 historical weather THAT day (temp/wind/precip, Open-Meteo archive)
  reported landmark/direction -> position (EPSG:25832 -> lat/lon)
  model score at that position + its rank within that day's field (percentile)

Usage (repo root, venv active):
    python scripts/validation_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pyproj import Transformer

from reindeer.geocode.gazetteer import load_gazetteer
from reindeer.geocode.positions import resolve_position
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA
from reindeer.model.score import score_cells
from reindeer.model.validation import percentile_rank
from reindeer.weather.historical import fetch_archive, weather_by_date
from shapely.geometry import Point
from shapely.prepared import prep

PROC = _ROOT / "data" / "processed"
OUT = PROC / "validation_detail.csv"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def main() -> None:
    grid = pd.read_csv(PROC / "grid_250m.csv")
    g = pd.read_csv(PROC / "terrain_250m.csv").merge(
        grid[["cell_id", "in_lordalen", "in_dalsida"]], on="cell_id")
    g = g.merge(pd.read_csv(PROC / "disturbance_250m.csv")[["cell_id", "dist_disturb_m"]], on="cell_id")
    g = g.merge(pd.read_csv(PROC / "forage_250m.csv")[["cell_id", "forage"]], on="cell_id")
    gaz = load_gazetteer()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    gx, gy = g.east.to_numpy(), g.north.to_numpy()

    obs = pd.read_csv(_ROOT / "data" / "interim" / "observations.csv")
    cx, cy = g.east.mean(), g.north.mean()
    lon, lat = _to_wgs.transform(cx, cy)
    wx = weather_by_date(fetch_archive(lat, lon, obs.date.min(), obs.date.max()))

    elev, slope, tpi = g.elevation_m.to_numpy(), g.slope_deg.to_numpy(), g.tpi_m.to_numpy()
    dist, forage = g.dist_disturb_m.to_numpy(), g.forage.to_numpy()
    score_cache: dict[str, np.ndarray] = {}

    rows = []
    for _, r in obs.iterrows():
        if pd.isna(r["landmark_phrases"]):
            continue
        pos = resolve_position(r["landmark_phrases"], r["direction_hints"], gaz, 3000.0)
        if pos is None:
            continue
        e, n, method = pos
        if not inside.contains(Point(e, n)):
            continue
        d = r["date"]
        w = wx.get(d)
        if w is None:
            continue
        if d not in score_cache:
            score_cache[d] = score_cells(elev, slope, tpi, w,
                                         disturb_dist=dist, forage=forage)["score_raw"]
        s = score_cache[d]
        idx = int(((gx - e) ** 2 + (gy - n) ** 2).argmin())
        plon, plat = _to_wgs.transform(g.iloc[idx].east, g.iloc[idx].north)
        pr = percentile_rank(s[idx], s)
        lms = json.loads(r["landmark_phrases"])
        rows.append({
            "date": d,
            "report": (str(r["observation_text"])[:60]),
            "landmark": lms[0] if lms else "",
            "pos_method": method,
            "lat": round(plat, 4), "lon": round(plon, 4),
            "temp_C": w.temp_c, "wind_ms": w.wind_ms, "precip_mm": w.precip_mm,
            "model_score": round(float(s[idx]), 3),
            "percentile": round(float(pr), 3),
            "in_top20pct": pr >= 0.80,
        })

    det = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    det.to_csv(OUT, index=False, encoding="utf-8")

    hit20 = det["in_top20pct"].mean()
    half = (det["percentile"] >= 0.5).mean()
    print(f"{len(det)} reports checked: real weather on the exact date vs reported position")
    print(f"  in model's favored half: {half*100:.0f}%  | in top 20%: {hit20*100:.0f}%  "
          f"| mean percentile: {det['percentile'].mean():.3f}\n")
    show = det[["date", "landmark", "pos_method", "temp_C", "wind_ms", "precip_mm",
                "percentile", "in_top20pct"]]
    with pd.option_context("display.max_rows", 60, "display.width", 200):
        print(show.to_string(index=False))
    print(f"\n-> {OUT}  (full detail incl. report text + lat/lon)")


if __name__ == "__main__":
    main()
