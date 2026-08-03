"""Phase 3: materialise the 250 m scoring grid to CSV.

Builds the grid clipped to the Lordalen Statsallmenning polygon (+ Dalsida buffer)
from the cached KML and writes one row per cell:
    data/processed/grid_250m.csv   cell_id,col,row,east,north,in_lordalen,in_dalsida

The CSV is in EPSG:25832 and is the substrate the scoring engine consumes. It is a
reproducible output (gitignored): rebuild any time with this script.

QGIS: Add Delimited Text Layer -> X=east, Y=north, CRS = EPSG:25832.

Usage (repo root, venv active):
    python scripts/build_grid.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from reindeer.terrain.grid import (  # noqa: E402
    build_grid, load_field_polygons, CELL_SIZE_M, LORDALEN, DALSIDA,
)

PROCESSED = _ROOT / "data" / "processed"
OUT = PROCESSED / "grid_250m.csv"


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    polys = load_field_polygons()
    cells = build_grid()

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "col", "row", "east", "north",
                    "in_lordalen", "in_dalsida"])
        for c in cells:
            w.writerow([c.cell_id, c.col, c.row, f"{c.east:.1f}", f"{c.north:.1f}",
                        int(c.in_lordalen), int(c.in_dalsida)])

    n = len(cells)
    n_lord = sum(c.in_lordalen for c in cells)
    n_dals = sum(c.in_dalsida for c in cells)
    n_both = sum(c.in_lordalen and c.in_dalsida for c in cells)
    cell_km2 = (CELL_SIZE_M / 1000.0) ** 2

    print(f"Grid -> {OUT}")
    print(f"  cell size: {CELL_SIZE_M} m  ({cell_km2:.4f} km^2/cell)")
    print(f"  cells total : {n:6d}  (~{n*cell_km2:7.1f} km^2)")
    print(f"  in Lordalen : {n_lord:6d}  (~{n_lord*cell_km2:7.1f} km^2)  "
          f"<- primary reported field")
    print(f"  in Dalsida  : {n_dals:6d}  (~{n_dals*cell_km2:7.1f} km^2)  "
          f"<- extended buffer")
    print(f"  in both     : {n_both:6d}")
    for name in (LORDALEN, DALSIDA):
        a = polys[name].area / 1e6
        print(f"  polygon area {name:28s}: {a:7.1f} km^2")


if __name__ == "__main__":
    main()
