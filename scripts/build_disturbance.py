"""Phase 3: build the distance-to-disturbance layer onto the 250 m grid.

Reads data/processed/grid_250m.csv, the OSM disturbance dump, and the KML cabins/
camping, and writes per-cell nearest distance:
    data/processed/disturbance_250m.csv   cell_id,east,north,dist_disturb_m

Reproducible output (gitignored). Usage (repo root, venv active):
    python scripts/build_disturbance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reindeer.terrain.disturbance import (  # noqa: E402
    load_osm_disturbance, load_kml_disturbance, nearest_distance,
)

PROCESSED = _ROOT / "data" / "processed"
GRID_CSV = PROCESSED / "grid_250m.csv"
OUT = PROCESSED / "disturbance_250m.csv"


def main() -> None:
    if not GRID_CSV.exists():
        raise SystemExit("grid_250m.csv missing - run scripts/build_grid.py first")

    osm = load_osm_disturbance()
    kml = load_kml_disturbance()
    geoms = osm + kml
    print(f"disturbance features: {len(osm)} OSM + {len(kml)} KML = {len(geoms)}")

    grid = pd.read_csv(GRID_CSV)
    dist = nearest_distance(grid["east"].to_numpy(), grid["north"].to_numpy(), geoms)

    out = grid[["cell_id", "east", "north"]].copy()
    out["dist_disturb_m"] = np.round(dist, 1)
    out.to_csv(OUT, index=False, encoding="utf-8")

    lord = grid["in_lordalen"] == 1
    d_all, d_lord = out["dist_disturb_m"], out.loc[lord.values, "dist_disturb_m"]
    print(f"Disturbance -> {OUT}  ({len(out)} cells)")
    for label, s in (("all cells     ", d_all), ("Lordalen field", d_lord)):
        print(f"  {label}: nearest dist (m) min/median/mean/max = "
              f"{s.min():.0f} / {s.median():.0f} / {s.mean():.0f} / {s.max():.0f}")
    for thr in (250, 500, 1000, 2000):
        frac = (d_lord <= thr).mean()
        print(f"    Lordalen cells within {thr:5d} m of disturbance: {frac*100:4.1f}%")


if __name__ == "__main__":
    main()
