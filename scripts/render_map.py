"""Phase 4: render scored grid CSVs to heatmap PNGs.

Renders every output/demo/*.csv (the fixed weather-scenario sanity grids from
score_demo.py) to output/demo/<name>.png.

Usage (repo root, venv active):
    python scripts/render_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from reindeer.viz.render import render_heatmap  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
DTM = _ROOT / "data" / "raw" / "dem" / "dtm_50m_25833.tif"


def main() -> None:
    from reindeer.paths import outdir
    csvs = sorted(outdir("demo").glob("*.csv"))
    if not csvs:
        raise SystemExit("no output/demo/*.csv found - run score_demo.py first")
    for csv in csvs:
        df = pd.read_csv(csv)
        name = csv.stem
        out = render_heatmap(df["east"], df["north"], df["score"],
                             outdir("demo") / f"{name}.png",
                             title="Reindeer presence — Lordalen", subtitle=name,
                             dtm_path=DTM)
        print(f"  {csv.name} -> {out}")
    print(f"\n{len(csvs)} map(s) -> {outdir('demo')}")


if __name__ == "__main__":
    main()
