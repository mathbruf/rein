"""Phase 4: produce tomorrow's live map + ranked top-zones list from a live forecast.

Fetches the next-day forecast (real per-cell metno_nordic field), runs the tuned
scorer over the Lordalen cells with all static layers, then writes:
  - the heatmap PNG   -> output/forecast/<date>.png
  - the scored grid   -> output/forecast/<date>.csv
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

from reindeer.weather import field as WF  # noqa: E402
from reindeer.model.score import score_cells, explain_cell  # noqa: E402
from reindeer.viz.render import render_heatmap, cluster_top_zones  # noqa: E402
from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402
from reindeer.paths import outdir  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
DTM = _ROOT / "data" / "raw" / "dem" / "dtm_50m_25833.tif"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def regime_sentence(p_ins, p_shl):
    if p_ins >= 0.3 and p_ins >= p_shl:
        return "Driver: insect-escape — animals climb to cool, breezy, exposed ground."
    if p_shl >= 0.3:
        return "Driver: shelter — animals seek leeward, sheltered, lower ground."
    return ("Driver: weak — calm/mild day, little weather push; map leans on the "
            "high-ground preference, forage and distance from access.")


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
    ce, cn, cz = field["east"].to_numpy(), field["north"].to_numpy(), field["elevation_m"].to_numpy()

    # next available forecast day, real per-cell metno_nordic (1 km) field
    _, _, lat, lon = WF.build_lattice(ce, cn, 4000.0)
    dates = WF.available_forecast_dates(lat, lon)
    today = __import__("datetime").date.today().isoformat()
    date = next((d for d in dates if d > today), dates[-1])
    wf = WF.build_field("forecast", date, ce, cn, cz)
    t_med, w_med, p_mean = wf.area_summary()
    w_dir = WF.circular_mean_dir(wf.wind_dir_deg, wf.wind_ms)
    print(f"Forecast for {date} (real metno_nordic field over {len(ce)} cells)")
    print(f"  daytime: temp≈{t_med:.1f} C  wind≈{w_med:.1f} m/s from {w_dir:.0f}°  "
          f"precip≈{p_mean:.1f} mm  | data lapse {wf.lapse_c_per_m*1000:+.1f} C/km")

    disturb = field["dist_disturb_m"] if "dist_disturb_m" in field else None
    forage = field["forage"] if "forage" in field else None
    res = score_cells(field["elevation_m"], field["slope_deg"], field["tpi_m"],
                      wfield=wf, aspect=field["aspect_deg"],
                      disturb_dist=disturb, forage=forage)
    field["score"] = res["score"]
    p_ins, p_shl = res["insect_pressure"][0], res["shelter_pressure"][0]
    print(f"  regime: insect_pressure={p_ins:.2f}  shelter_pressure={p_shl:.2f}")

    out = outdir("forecast") / f"{date}.csv"
    cols = ["cell_id", "east", "north", "elevation_m", "tpi_m", "score"]
    field[cols].to_csv(out, index=False, encoding="utf-8")

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
                          f"elev {r['elevation_m']:.0f} m  TPI {r['tpi_m']:+.0f}  "
                          f"| {reason}  | near {name} ({dkm:.1f} km)")

    weather_text = (f"forecast for {date}\n{t_med:.1f} °C  ·  wind {w_med:.1f} m/s from {w_dir:.0f}°  ·  "
                    f"{p_mean:.1f} mm rain")
    png = render_heatmap(
        field["east"], field["north"], field["score"],
        outdir("forecast") / f"{date}.png",
        title="Reindeer presence — Lordalen (tomorrow)",
        subtitle=date, dtm_path=DTM, zones=zones,
        weather_text=weather_text, regime_text=regime_sentence(p_ins, p_shl),
        wind_dir_deg=w_dir, wind_label=f"wind {w_med:.0f} m/s")

    print(f"\nTop zones for {date} (lat/lon for the field notebook):")
    print("\n".join(print_rows))
    print(f"\n-> {out}  (QGIS: X=east, Y=north, CRS EPSG:25832)")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
