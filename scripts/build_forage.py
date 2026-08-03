"""Phase 3: build the forage-quality layer onto the 250 m grid (NIBIO AR50).

Reads data/processed/grid_250m.csv and the AR50 GML (downloads it if missing),
assigns each cell the forage value of the land-cover polygon it falls in, and writes:
    data/processed/forage_250m.csv   cell_id,east,north,arealtype,forage

Reproducible output (gitignored). Usage (repo root, venv active):
    python scripts/build_forage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from reindeer.terrain.forage import (  # noqa: E402
    download_ar50, load_ar50_polygons, forage_to_grid,
)

PROCESSED = _ROOT / "data" / "processed"
GRID_CSV = PROCESSED / "grid_250m.csv"
OUT = PROCESSED / "forage_250m.csv"

_ARNAME = {50: "open-alpine", 60: "mire", 30: "forest", 20: "agri",
           10: "built", 70: "snow/ice", 81: "freshwater", 82: "sea",
           99: "unmapped", -1: "none"}


def main() -> None:
    if not GRID_CSV.exists():
        raise SystemExit("grid_250m.csv missing - run scripts/build_grid.py first")

    gml = download_ar50()
    polys, types = load_ar50_polygons(gml)
    print(f"AR50 polygons: {len(polys)}")

    grid = pd.read_csv(GRID_CSV)
    at, forage = forage_to_grid(grid["east"].to_numpy(), grid["north"].to_numpy(),
                                polys, types)
    out = grid[["cell_id", "east", "north"]].copy()
    out["arealtype"] = at
    out["forage"] = forage.round(3)
    out.to_csv(OUT, index=False, encoding="utf-8")

    lord = grid["in_lordalen"] == 1
    fl = out.loc[lord.values]
    print(f"Forage -> {OUT}  ({len(out)} cells)")
    print(f"  Lordalen forage min/mean/max = "
          f"{fl.forage.min():.2f} / {fl.forage.mean():.2f} / {fl.forage.max():.2f}")
    print("  Lordalen land-cover mix:")
    vc = fl["arealtype"].value_counts()
    for code, n in vc.items():
        print(f"    {_ARNAME.get(int(code), code):12s} ({int(code):>3}): "
              f"{n:5d} cells ({n/len(fl)*100:4.1f}%)")


if __name__ == "__main__":
    main()
