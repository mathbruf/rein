"""Phase 4 (partial): produce tomorrow's live scored grid + ranked top-zones list.

Fetches the next-day MET Locationforecast for the field centroid, runs the tuned
scorer over the Lordalen cells with all static layers, writes a scored CSV, and
prints the top zones. Heatmap rendering (matplotlib/folium) is still TODO.

Usage (repo root, venv active):
    python scripts/daily_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

from reindeer.weather.forecast import fetch_forecast, next_day_weather  # noqa: E402
from reindeer.model.score import score_cells  # noqa: E402
from reindeer.viz.render import render_heatmap  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


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

    print(f"\nTop 12 zones for {date} (lat/lon for the field notebook):")
    for _, r in top.iterrows():
        plon, plat = _to_wgs.transform(r["east"], r["north"])
        print(f"  score {r['score']:.2f}  {plat:.4f}N {plon:.4f}E  "
              f"elev {r['elevation_m']:.0f} m  TPI {r['tpi_m']:+.0f}")
    print(f"\n-> {out}  (QGIS: X=east, Y=north, CRS EPSG:25832)")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
