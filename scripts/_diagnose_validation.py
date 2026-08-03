"""Diagnostic audit of the Phase-5 validation chain (bug hunt; does NOT change the model).

Checks, with evidence, the things that could drag the hit-rate below chance for reasons
other than the model being biologically wrong:
  1 grid integrity (row count, duplicate cell_ids, NaNs per layer)
  2 NaN contamination in per-date scores (corrupts percentile)
  3 date -> weather matching (every obs date present? values sane? regime mix)
  4 observed-cell terrain vs field (are obs still systematically in low/near-road cells?)
  5 percentile by positioning method (at-landmark vs offset vs mot)
  6 background definition (full grid vs the observation's own subfield)
  7 hand-recompute one observed score to confirm the lookup is correct
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
from reindeer.model.score import score_cells, insect_pressure, shelter_pressure
from reindeer.model import validation as V
from reindeer.weather.historical import fetch_archive, weather_by_date
from shapely.geometry import Point
from shapely.prepared import prep

PROC = _ROOT / "data" / "processed"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def load_grid():
    grid = pd.read_csv(PROC / "grid_250m.csv")
    df = pd.read_csv(PROC / "terrain_250m.csv").merge(
        grid[["cell_id", "in_lordalen", "in_dalsida"]], on="cell_id")
    df = df.merge(pd.read_csv(PROC / "disturbance_250m.csv")[["cell_id", "dist_disturb_m"]], on="cell_id")
    df = df.merge(pd.read_csv(PROC / "forage_250m.csv")[["cell_id", "forage"]], on="cell_id")
    return df


def main():
    g = load_grid()
    print("=" * 70)
    print("[1] GRID INTEGRITY")
    print(f"  rows={len(g)}  unique cell_id={g.cell_id.nunique()}  "
          f"dup={len(g)-g.cell_id.nunique()}")
    for c in ["east", "north", "elevation_m", "slope_deg", "tpi_m", "dist_disturb_m", "forage"]:
        print(f"  NaNs in {c:15s}: {int(g[c].isna().sum())}")

    gaz = load_gazetteer()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    gx, gy = g["east"].to_numpy(), g["north"].to_numpy()

    obs = pd.read_csv(_ROOT / "data" / "interim" / "observations.csv")
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
        idx = int(((gx - e) ** 2 + (gy - n) ** 2).argmin())
        rows.append({"date": r["date"], "idx": idx, "method": method,
                     "lm": next((lm for lm in json.loads(r["landmark_phrases"]) if lm in gaz), None)})
    used = pd.DataFrame(rows)

    cx, cy = g["east"].mean(), g["north"].mean()
    lon, lat = _to_wgs.transform(cx, cy)
    wx = weather_by_date(fetch_archive(lat, lon, min(used.date), max(used.date)))

    elev, slope, tpi = g.elevation_m.to_numpy(), g.slope_deg.to_numpy(), g.tpi_m.to_numpy()
    dist, forage = g.dist_disturb_m.to_numpy(), g.forage.to_numpy()

    print("\n[3] DATE -> WEATHER MATCHING")
    miss = sorted(set(used.date) - set(wx))
    print(f"  obs dates={used.date.nunique()}  missing weather={len(miss)} {miss[:5]}")
    regi = []
    for d in sorted(set(used.date)):
        if d in wx:
            w = wx[d]
            regi.append((d, w.temp_c, w.wind_ms, w.precip_mm,
                         round(insect_pressure(w), 2), round(shelter_pressure(w), 2)))
    rdf = pd.DataFrame(regi, columns=["date", "temp", "wind", "precip", "p_ins", "p_shl"])
    print(f"  temp range {rdf.temp.min()}..{rdf.temp.max()} (mean {rdf.temp.mean():.1f})")
    print(f"  wind range {rdf.wind.min()}..{rdf.wind.max()} (mean {rdf.wind.mean():.1f})")
    print(f"  precip mean {rdf.precip.mean():.1f}  | insect>0 days: {(rdf.p_ins>0).sum()}  "
          f"shelter>0.5 days: {(rdf.p_shl>0.5).sum()} / {len(rdf)}")

    per_date = {d: (score_cells(elev, slope, tpi, wx[d], disturb_dist=dist, forage=forage)["score_raw"]
                    if d in wx else None) for d in sorted(set(used.date))}
    print("\n[2] NaN CONTAMINATION IN SCORES")
    anan = [d for d, s in per_date.items() if s is not None and np.isnan(s).any()]
    print(f"  dates with any NaN score cell: {len(anan)} {anan[:5]}")
    if anan:
        s0 = per_date[anan[0]]
        print(f"  e.g. {anan[0]}: {int(np.isnan(s0).sum())}/{len(s0)} cells NaN")

    used = used[used.date.map(lambda d: per_date.get(d) is not None)].copy()
    obs_sc = np.array([per_date[r.date][r.idx] for r in used.itertuples()])
    obs_bg = [per_date[r.date] for r in used.itertuples()]
    pct = V.date_matched_percentiles(obs_sc, obs_bg)
    used["pct"] = pct

    print("\n[4] OBSERVED CELLS vs FIELD")
    oc = g.iloc[used.idx.to_numpy()]
    fld = g
    print(f"  obs   elev {oc.elevation_m.mean():.0f} | tpi {oc.tpi_m.mean():+.0f} | "
          f"dist {oc.dist_disturb_m.mean():.0f} | forage {oc.forage.mean():.2f}")
    print(f"  field elev {fld.elevation_m.mean():.0f} | tpi {fld.tpi_m.mean():+.0f} | "
          f"dist {fld.dist_disturb_m.mean():.0f} | forage {fld.forage.mean():.2f}")
    print(f"  observed percentile: mean {pct.mean():.3f} median {np.median(pct):.3f} "
          f"min {pct.min():.3f} max {pct.max():.3f}")

    print("\n[5] PERCENTILE BY POSITIONING METHOD")
    print(used.groupby("method")["pct"].agg(["count", "mean"]).to_string())
    print("  top landmarks (count, mean pct):")
    lm = used.groupby("lm")["pct"].agg(["count", "mean"]).sort_values("count", ascending=False).head(8)
    print(lm.to_string())

    print("\n[6] BACKGROUND DEFINITION: full grid vs own subfield")
    in_l = g.in_lordalen.to_numpy().astype(bool)
    in_d = g.in_dalsida.to_numpy().astype(bool)
    pct_sub = []
    for r in used.itertuples():
        s = per_date[r.date]
        mask = in_l if in_l[r.idx] else in_d
        pct_sub.append(V.percentile_rank(s[r.idx], s[mask]))
    pct_sub = np.array(pct_sub)
    print(f"  full-grid background mean pct : {pct.mean():.3f}")
    print(f"  own-subfield background mean  : {pct_sub.mean():.3f}")
    print(f"  Lordalen vs Dalsida score means per a sample date:")
    d0 = used.date.iloc[0]; s = per_date[d0]
    print(f"    {d0}: Lordalen mean {s[in_l].mean():.3f}  Dalsida mean {s[in_d].mean():.3f}")

    print("\n[7] HAND-CHECK one observed lookup")
    r = used.iloc[0]
    s = per_date[r.date]
    rank = (s < s[r.idx]).mean()
    print(f"  obs0 date={r.date} idx={r.idx} lm={r.lm} score={s[r.idx]:.4f} "
          f"recomputed pct={rank:.3f} stored={r.pct:.3f}  (match={abs(rank-r.pct)<0.02})")
    plon, plat = _to_wgs.transform(g.iloc[r.idx].east, g.iloc[r.idx].north)
    print(f"  obs0 cell at {plat:.4f}N {plon:.4f}E elev {g.iloc[r.idx].elevation_m:.0f} "
          f"tpi {g.iloc[r.idx].tpi_m:+.0f} dist {g.iloc[r.idx].dist_disturb_m:.0f}")


if __name__ == "__main__":
    main()
