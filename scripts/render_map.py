"""Phase 4: render scored grid CSVs to heatmap PNGs.

Renders every data/processed/score_*.csv (the demo scenarios + any live map) to
data/processed/maps/<name>.png.

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
MAPS = PROCESSED / "maps"
DTM = _ROOT / "data" / "raw" / "dem" / "dtm_50m_25833.tif"


def main() -> None:
    csvs = sorted(PROCESSED.glob("score_*.csv"))
    if not csvs:
        raise SystemExit("no score_*.csv found - run score_demo.py or daily_map.py first")
    for csv in csvs:
        df = pd.read_csv(csv)
        name = csv.stem.replace("score_", "")
        out = render_heatmap(df["east"], df["north"], df["score"],
                             MAPS / f"{name}.png",
                             title="Reindeer presence — Lordalen", subtitle=name,
                             dtm_path=DTM)
        print(f"  {csv.name} -> {out}")
    print(f"\n{len(csvs)} map(s) -> {MAPS}")


if __name__ == "__main__":
    main()
