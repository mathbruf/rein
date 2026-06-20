"""Phase 4: produce tomorrow's live map + ranked top-zones list from a live forecast.

Fetches the next-day MET Locationforecast for the field centroid, runs the tuned
scorer over the Lordalen cells with all static layers, then writes:
  - a scored CSV (data/processed/score_live_<date>.csv),
  - a 0..1 heatmap PNG with the top zones marked (data/processed/maps/live_<date>.png),
  - a printed ranked top-zones list with a per-zone reason and nearest named landmark.

Usage (repo root, venv active):
    python scripts/daily_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # print Norwegian landmark names correctly

import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

from reindeer.weather.forecast import fetch_forecast, next_day_weather  # noqa: E402
from reindeer.model.score import score_cells, explain_cell  # noqa: E402
from reindeer.viz.render import render_heatmap  # noqa: E402
from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def nearest_landmark(east, north, entries):
    """Nearest gazetteer landmark to a point (EPSG:25832) -> (name, dist_km)."""
    best, best_d = None, float("inf")
    for g in entries:
        d = ((g.east - east) ** 2 + (g.north - north) ** 2) ** 0.5
        if d < best_d:
            best, best_d = g, d
    return (best.name if best else "?"), best_d / 1000.0


def load_field() -> pd.DataFrame:
    grid = pd.read_csv(PROCESSED / "grid_250m.csv")
    df = pd.read_csv(PROCESSED / "terrain_250m.csv").merge(
        grid[["cell_id", "in_lordalen"]], on="cell_id")
    for fn, col in (("disturbance_250m.csv", "dist_disturb_m"),
                    ("forage_250m.csv", "forage")):
        p = PROCESSED / fn
        if p.exists():
            df = df.merge(pd.read_csv(p)[["cell_id", col]], on="cell_id")
    return df[df["in_lordalen"] == 1].copy()


def main() -> None:
    field = load_field()
    cx, cy = field["east"].mean(), field["north"].mean()
    lon, lat = _to_wgs.transform(cx, cy)

    forecast = fetch_forecast(lat, lon)
    date, w = next_day_weather(forecast)
    print(f"Field centroid {lat:.4f}N {lon:.4f}E  |  forecast for {date}")
    print(f"  daytime: temp={w.temp_c} C  wind={w.wind_ms} m/s  precip={w.precip_mm} mm")

    disturb = field["dist_disturb_m"] if "dist_disturb_m" in field else None
    forage = field["forage"] if "forage" in field else None
    res = score_cells(field["elevation_m"], field["slope_deg"], field["tpi_m"], w,
                      disturb_dist=disturb, forage=forage)
    field["score"] = res["score"]
    print(f"  regime: insect_pressure={res['insect_pressure'][0]:.2f}  "
          f"shelter_pressure={res['shelter_pressure'][0]:.2f}")

    out = PROCESSED / f"score_live_{date}.csv"
    cols = ["cell_id", "east", "north", "elevation_m", "tpi_m", "score"]
    field[cols].to_csv(out, index=False, encoding="utf-8")

    top = field.nlargest(12, "score")
    png = render_heatmap(
        field["east"], field["north"], field["score"],
        PROCESSED / "maps" / f"live_{date}.png",
        title=f"Lordalen presence - {date} ({w.temp_c}C {w.wind_ms}m/s {w.precip_mm}mm)",
        top=(top["east"].to_numpy(), top["north"].to_numpy()))

    entries = list(load_gazetteer().values())
    p_ins = res["insect_pressure"][0]
    p_shl = res["shelter_pressure"][0]
    print(f"\nTop 12 zones for {date} (lat/lon for the field notebook):")
    for _, r in top.iterrows():
        plon, plat = _to_wgs.transform(r["east"], r["north"])
        name, dkm = nearest_landmark(r["east"], r["north"], entries)
        reason = explain_cell(r["elevation_m"], r["tpi_m"], p_ins, p_shl,
                              dist_disturb=r.get("dist_disturb_m"),
                              forage=r.get("forage"))
        print(f"  score {r['score']:.2f}  {plat:.4f}N {plon:.4f}E  "
              f"elev {r['elevation_m']:.0f} m  TPI {r['tpi_m']:+.0f}  "
              f"| {reason}  | near {name} ({dkm:.1f} km)")
    print(f"\n-> {out}  (QGIS: X=east, Y=north, CRS EPSG:25832)")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
