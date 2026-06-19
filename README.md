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
```
Outputs land in `data/processed/*.csv` (EPSG:25832; import into QGIS as delimited text).

## Status
Phases 0–3 complete: the rule-based scorer turns a forecast + the static layers
(terrain, disturbance, forage) into a 0–1 grid over the field, with weights tuned to the
hunter's field experience. **Phase 4 in progress** — `daily_map.py` produces tomorrow's
scored grid + ranked top-zones from a live MET forecast; heatmap rendering is the remaining
Phase-4 piece. Phase 5 (validation against harvested sightings) needs a free MET Frost key.
See `PROGRESS.md` for the exact next action.

## Last session
This is the last claude-code session: claude --resume 4ba2afce-adc5-4f3a-a435-c6ff4fb59d8c
