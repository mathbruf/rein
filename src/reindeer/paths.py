"""One place for the human-facing output layout (2026-08-09 restructure).

Everything a person actually LOOKS AT lands under a single top-level `output/`
folder with self-explaining subfolders and date-named files:

    output/
      forecast/    tomorrow's map + scored grid        2026-08-10.png / .csv
      historical/  past-date maps (real ERA5 weather)  2025-09-22.png / .csv
      analysis/    validation & hit-rate charts        hit_analysis.png
      demo/        fixed weather-scenario sanity maps  warm_calm.png / .csv ...

`data/` keeps only pipeline inputs and intermediates (raw caches, grid/terrain
layers). Reports stay in `docs/`. All of output/ is reproducible and gitignored.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "output"
FORECAST_DIR = OUTPUT / "forecast"
HISTORICAL_DIR = OUTPUT / "historical"
ANALYSIS_DIR = OUTPUT / "analysis"
DEMO_DIR = OUTPUT / "demo"


def outdir(kind: str) -> Path:
    """Return (and create) the output subfolder for `kind`."""
    d = {"forecast": FORECAST_DIR, "historical": HISTORICAL_DIR,
         "analysis": ANALYSIS_DIR, "demo": DEMO_DIR}[kind]
    d.mkdir(parents=True, exist_ok=True)
    return d
