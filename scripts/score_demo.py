"""Phase 3 DoD demo: given a sample weather input, produce a reproducible 0..1 grid.

Runs the v0 scorer over the Lordalen cells for two contrasting forecasts and writes
a scored CSV per scenario, then prints how the favored ground shifts between regimes
(warm+calm -> high/exposed; cold+windy -> low/sheltered).

Usage (repo root, venv active):
    python scripts/score_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from reindeer.model.score import WeatherDay, score_cells  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"

SCENARIOS = {
    "warm_calm": WeatherDay(temp_c=19.0, wind_ms=2.0, precip_mm=0.0),
    "cold_windy": WeatherDay(temp_c=1.0, wind_ms=12.0, precip_mm=6.0),
}


def main() -> None:
    grid = pd.read_csv(PROCESSED / "grid_250m.csv")
    terr = pd.read_csv(PROCESSED / "terrain_250m.csv")
    df = terr.merge(grid[["cell_id", "in_lordalen"]], on="cell_id")
    field = df[df["in_lordalen"] == 1].copy()   # reported surface = Lordalen field
    print(f"Scoring {len(field)} Lordalen cells\n")

    for name, w in SCENARIOS.items():
        res = score_cells(field["elevation_m"], field["slope_deg"],
                          field["tpi_m"], w)
        out = field[["cell_id", "east", "north", "elevation_m", "tpi_m"]].copy()
        out["score"] = res["score"]
        out.to_csv(PROCESSED / f"score_{name}.csv", index=False, encoding="utf-8")

        top = out.nlargest(max(1, len(out) // 5), "score")   # top 20%
        print(f"[{name}]  temp={w.temp_c}C wind={w.wind_ms}m/s precip={w.precip_mm}mm")
        print(f"   insect_pressure={res['insect_pressure'][0]:.2f}  "
              f"shelter_pressure={res['shelter_pressure'][0]:.2f}")
        print(f"   top-20% cells: mean elev={top.elevation_m.mean():6.0f} m  "
              f"mean TPI={top.tpi_m.mean():+5.0f} m   "
              f"(field mean elev={out.elevation_m.mean():.0f}, "
              f"TPI={out.tpi_m.mean():+.0f})")
        print(f"   -> data/processed/score_{name}.csv\n")


if __name__ == "__main__":
    main()
