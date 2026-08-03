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
**Full pipeline built (phases 0–5), scorer v1 + redesigned map (2026-07-07).**
`python scripts/daily_map.py` fetches tomorrow's live MET forecast and produces, in one
command, a scored grid CSV and a **human-readable heatmap**: a shaded-relief terrain
background, a translucent red→green probability wash (red = avoid, green = favoured), up to
six clustered "go here" zones each anchored to the nearest named landmark, and a side panel
with the date, the day's weather in plain language, which behavioural driver is active, and
the ranked zone list — plus a plain-language legend, scale bar and north arrow.

**Scorer v1 — per-cell weather (physics, not fitted).** The single area forecast is now
*downscaled* to every cell: temperature by the standard lapse rate (~6.5 °C/km) and wind by
terrain exposure (ridges windier, hollows calmer). This fixes the v0 failure where a calm,
cool autumn valley reading switched **both** behavioural regimes off and the map went flat —
now the shelter regime engages because the tops are modeled genuinely cold and windy.

**Honest validation result (`docs/validation_report.md`, `docs/validation_bug_audit.md`):**
tested against held-out hunting-season sightings using **real ERA5 weather on each report's
exact date** and **human-pinned real positions** (evaluated over a 2.5 km zone, since
"nord for X" names an area), scorer v1 places **57% of the readings in the model's favoured
half** (up from 51% for v0; date-matched AUC 0.484 → 0.497). The direction is now correct,
but on 37 held-out reports this is still **within sampling noise** (p≈0.51) — honestly
~chance, not yet a proven edge. Earlier worse-than-chance numbers (AUC 0.32) were artifacts
of a fuzzy-geocoding bug (a valley matched to a stream, now fixed) and a positional-offset
assumption (replaced by the pins). The ablation now flags the **effort-bias confound**:
removing the disturbance penalty *raises* the apparent AUC, the expected signature of
reports being biased toward accessible terrain (not a reason to drop the rule — IDEA 009).
The weights were **never** fitted to the sightings, and raising the validated hit-rate
further needs more data + cross-validation (IDEAS 002, 009, 010), not hand-tuning these 37
points. So this is an honest, validated **prototype**, not yet a v1.0. See `PROGRESS.md`.

## Last session
This is the last claude-code session: claude --resume 4ba2afce-adc5-4f3a-a435-c6ff4fb59d8c
