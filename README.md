# Reindeer Heatmap (Lordalen / Reinheimen)

A daily, forecast-driven **probability heatmap** of where wild reindeer (*villrein*) are
likely to be one day ahead, in the **Lordalen** hunting field of **Reinheimen
villreinområde** (Lesja, Innlandet, Norway). Using the next day's weather forecast and the
fixed landscape (terrain, forage, snow, distance from disturbance), it scores every grid
cell from 0 ("avoid / unlikely") to 1 ("strongly favored") so a hunter can decide, the
night before, where to walk in and where to glass from.

> **This is a search-narrowing tool, not a GPS oracle.** It does not predict the herd's
> exact position. Reindeer move as a social herd and tomorrow depends heavily on where they
> are today (usually unknown). The output is an honest probability surface that narrows the
> search and complements glassing and fieldcraft — it does not replace them.

## Quickstart

Create a virtual environment and install the (light) phase-1 dependencies.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project docs
- **`CLAUDE.md`** — persistent project context (the full infosheet) and the working rules every contributor follows.
- **`ROADMAP.md`** — the phased build plan and each phase's Definition of Done.
- **`PROGRESS.md`** — onboarding + running work log; read this to see the current phase and next action.
- **`IDEAS.md`** — backlog of proposed improvements (the human promotes these to the roadmap).

## Pipeline (run from the repo root, venv active)
```bash
python scripts/harvest_jaktinfo.py   # Phase 1: scrape + parse sightings (validation data)
python scripts/build_gazetteer.py    # Phase 2: landmark names -> coordinates (SSR)
python scripts/build_grid.py         # Phase 3: 250 m grid clipped to the field
python scripts/build_terrain.py      # Phase 3: DTM -> elevation/slope/ruggedness/TPI
python scripts/build_disturbance.py  # Phase 3: distance to roads/trails/cabins (OSM + KML)
python scripts/build_forage.py       # Phase 3: forage value from NIBIO AR50 land cover
python scripts/score_demo.py         # Phase 3: score sample forecasts (sanity demo)
python scripts/daily_map.py          # Phase 4: tomorrow's live forecast -> scored top-zones
python scripts/render_map.py         # Phase 4: render scored CSVs to heatmap PNGs
python scripts/validate.py           # Phase 5: test the scorer vs held-out sightings
```
Outputs land in `data/processed/*.csv` (EPSG:25832; import into QGIS as delimited text);
maps in `data/processed/maps/`; the validation write-up in `docs/validation_report.md`.

## Status
**Full pipeline built (phases 0–5).** `python scripts/daily_map.py` fetches tomorrow's
live MET forecast and produces, in one command, a scored grid CSV, a heatmap PNG with the
top zones marked, and a ranked top-zone list with per-zone reasons and nearest landmark.

**Honest validation result (`docs/validation_report.md`):** tested against 52 held-out
hunting-season sightings, the current scorer ranks them *below* chance (date-matched
AUC 0.32) — it is **anti-correlated** with where reindeer were reported in autumn.
Ablation shows the summer high-ground baseline and the disturbance penalty cause this
(reports are autumn, with insects gone and animals lower, and are effort-biased toward
accessible terrain). The weights were **not** retuned to the test set; the fix — a
seasonal (autumn) profile and an effort-aware re-test under cross-validation — is the next
cycle (IDEAS 008–010). So this is an honest, validated **prototype**, not yet a v1.0.
See `PROGRESS.md`.

## Last session
This is the last claude-code session: claude --resume 4ba2afce-adc5-4f3a-a435-c6ff4fb59d8c
