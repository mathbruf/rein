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

## Status
Phase 0 (scaffold) complete. Currently entering **Phase 1 — harvest & structure presence
data** from `villreinutvalet.no/jaktinfo`. See `PROGRESS.md` for the exact next action.
