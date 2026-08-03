"""Render the presence heatmap for a past date using that day's real ERA5 weather.

Same scorer + layers as the live daily map, but the weather comes from the Open-Meteo
ERA5 archive for the given date instead of a forecast. Produces the redesigned,
human-readable map (shaded relief + named zones + plain-language panel).

Usage (repo root, venv active):
    python scripts/historical_map.py            # defaults to 2025-09-22
    python scripts/historical_map.py 2025-09-22
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

from reindeer.weather.historical import fetch_archive, day_weather  # noqa: E402
from reindeer.model.score import score_cells, explain_cell  # noqa: E402
from reindeer.viz.render import render_heatmap, cluster_top_zones  # noqa: E402
from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
DTM = _ROOT / "data" / "raw" / "dem" / "dtm_50m_25833.tif"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def nearest_landmark(east, north, entries):
    best, best_d = None, float("inf")
    for g in entries:
        d = ((g.east - east) ** 2 + (g.north - north) ** 2) ** 0.5
        if d < best_d:
            best, best_d = g, d
    return (best.name if best else "?"), best_d / 1000.0


def regime_sentence(p_ins, p_shl):
    if p_ins >= 0.3 and p_ins >= p_shl:
        return "Driver: insect-escape — animals climb to cool, breezy, exposed ground."
    if p_shl >= 0.3:
        return "Driver: shelter — animals seek leeward, sheltered, lower ground."
    return ("Driver: weak — calm/mild day, little weather push; map leans on the "
            "high-ground preference, forage and distance from access.")


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-09-22"

    grid = pd.read_csv(PROCESSED / "grid_250m.csv")
    df = pd.read_csv(PROCESSED / "terrain_250m.csv").merge(
        grid[["cell_id", "in_lordalen"]], on="cell_id")
    for fn, col in (("disturbance_250m.csv", "dist_disturb_m"), ("forage_250m.csv", "forage")):
        p = PROCESSED / fn
        if p.exists():
            df = df.merge(pd.read_csv(p)[["cell_id", col]], on="cell_id")
    field = df[df["in_lordalen"] == 1].copy()

    cx, cy = field["east"].mean(), field["north"].mean()
    lon, lat = _to_wgs.transform(cx, cy)
    w = day_weather(fetch_archive(lat, lon, date, date), date)
    print(f"{date}  daytime weather: temp={w.temp_c} C  wind={w.wind_ms} m/s  precip={w.precip_mm} mm")

    disturb = field["dist_disturb_m"] if "dist_disturb_m" in field else None
    forage = field["forage"] if "forage" in field else None
    res = score_cells(field["elevation_m"], field["slope_deg"], field["tpi_m"], w,
                      disturb_dist=disturb, forage=forage)
    field["score"] = res["score"]
    p_ins, p_shl = res["insect_pressure"][0], res["shelter_pressure"][0]
    print(f"  regime: insect_pressure={p_ins:.2f}  shelter_pressure={p_shl:.2f}")

    out_csv = PROCESSED / f"score_hist_{date}.csv"
    field[["cell_id", "east", "north", "elevation_m", "tpi_m", "score"]].to_csv(
        out_csv, index=False, encoding="utf-8")

    # cluster the hot cells into distinct 'go here' zones, name + explain each
    entries = list(load_gazetteer().values())
    pts = cluster_top_zones(field["east"], field["north"], field["score"],
                            n=6, min_sep_m=2500.0)
    fe, fn = field["east"].to_numpy(), field["north"].to_numpy()
    zones, print_rows = [], []
    for i, (ze, zn, sc) in enumerate(pts, 1):
        r = field.iloc[int(((fe - ze) ** 2 + (fn - zn) ** 2).argmin())]
        name, dkm = nearest_landmark(ze, zn, entries)
        reason = explain_cell(r["elevation_m"], r["tpi_m"], p_ins, p_shl,
                              dist_disturb=r.get("dist_disturb_m"), forage=r.get("forage"))
        zones.append((ze, zn, f"{name} · {r['elevation_m']:.0f} m ({dkm:.1f} km)\n{reason}"))
        plon, plat = _to_wgs.transform(ze, zn)
        print_rows.append(f"  {i}. score {sc:.2f}  {plat:.4f}N {plon:.4f}E  "
                          f"elev {r['elevation_m']:.0f} m  | {reason}  | near {name} ({dkm:.1f} km)")

    weather_text = (f"{date}\n{w.temp_c:g} °C  ·  wind {w.wind_ms:g} m/s  ·  "
                    f"{w.precip_mm:g} mm rain")
    png = render_heatmap(
        field["east"], field["north"], field["score"],
        PROCESSED / "maps" / f"hist_{date}.png",
        title="Reindeer presence — Lordalen (historical)",
        subtitle=date, dtm_path=DTM, zones=zones,
        weather_text=weather_text, regime_text=regime_sentence(p_ins, p_shl))

    print(f"\nTop zones for {date}:")
    print("\n".join(print_rows))
    print(f"\n-> {png}\n-> {out_csv}")


if __name__ == "__main__":
    main()
