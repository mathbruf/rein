"""Phase 3: derive static terrain layers onto the 250 m grid.

Reads data/processed/grid_250m.csv and the cached DTM (downloads it if missing),
computes per-cell elevation / slope / ruggedness / TPI, and writes:
    data/processed/terrain_250m.csv   cell_id,east,north,elevation_m,slope_deg,
                                       ruggedness_m,tpi_m

Reproducible output (gitignored). Usage (repo root, venv active):
    python scripts/build_terrain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from reindeer.terrain.dem import download_dtm  # noqa: E402
from reindeer.terrain import terrain as T  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
GRID_CSV = PROCESSED / "grid_250m.csv"
OUT = PROCESSED / "terrain_250m.csv"


def main() -> None:
    if not GRID_CSV.exists():
        raise SystemExit("grid_250m.csv missing - run scripts/build_grid.py first")

    dtm = download_dtm(50)
    Z, valid, transform, res = T.load_dtm(dtm)
    print(f"DTM loaded: {Z.shape} @ {res:.1f} m")
    slope = T.slope_deg(Z, res)
    tpi_r = T.tpi(Z, valid, res, window_m=1000.0)

    grid = pd.read_csv(GRID_CSV)
    attrs = T.sample_to_grid(grid["east"].to_numpy(), grid["north"].to_numpy(),
                             Z, slope, tpi_r, transform)

    out = grid[["cell_id", "east", "north"]].copy()
    for k, v in attrs.items():
        out[k] = v
    out.to_csv(OUT, index=False, encoding="utf-8")

    n = len(out)
    miss = int(out["elevation_m"].isna().sum())
    print(f"Terrain -> {OUT}  ({n} cells, {miss} without DTM coverage)")
    for col in ("elevation_m", "slope_deg", "ruggedness_m", "tpi_m"):
        s = out[col]
        print(f"  {col:13s} min/mean/max = "
              f"{s.min():8.1f} / {s.mean():8.1f} / {s.max():8.1f}")


if __name__ == "__main__":
    main()
