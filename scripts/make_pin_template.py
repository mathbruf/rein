"""Make the manual-pin template for validation positions.

Writes data/gazetteer/manual_positions.csv: one row per (landmark, method) group of
in-grid reports, ranked by how many reports depend on it, pre-filled with my *assumed*
position (assumed_lat/lon) and BLANK real_lat/real_lon for you to fill from the map.
Reports carry no GPS, so filling real_lat/real_lon for the top rows removes the
positional assumption from the validation (resolve_position then uses your pin).

Read off coordinates in Google Maps: right-click the spot -> the lat,lon at the top of
the menu -> paste into real_lat, real_lon. You mainly need the top row(s).

Usage (repo root, venv active):
    python scripts/make_pin_template.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402
from reindeer.geocode.positions import resolve_position, MANUAL_PINS_CSV  # noqa: E402
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA  # noqa: E402
from shapely.geometry import Point  # noqa: E402
from shapely.prepared import prep  # noqa: E402

_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def main() -> None:
    gaz = load_gazetteer()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    obs = pd.read_csv(_ROOT / "data" / "interim" / "observations.csv")

    groups: dict = defaultdict(lambda: {"n": 0, "phrase": "", "pos": None})
    for _, r in obs.iterrows():
        if pd.isna(r["landmark_phrases"]):
            continue
        pos = resolve_position(r["landmark_phrases"], r["direction_hints"], gaz, 3000.0)
        if pos is None:
            continue
        e, n, method = pos
        if not inside.contains(Point(e, n)):
            continue
        lms = json.loads(r["landmark_phrases"])
        hints = json.loads(r["direction_hints"]) if not pd.isna(r["direction_hints"]) else []
        anchor = next((lm for lm in lms if lm in gaz), lms[0])
        rel = next((h["relation"] for h in hints if h.get("landmark") == anchor and h.get("relation")), "")
        mot = next((h["landmark"] for h in hints if h.get("relation") == "mot"), "")
        phrase = f"{rel + ' ' if rel else ''}{anchor}{' mot ' + mot if mot else ''}"
        key = (anchor, method)
        g = groups[key]
        g["n"] += 1
        if not g["phrase"]:
            g["phrase"] = phrase
            g["pos"] = (e, n)

    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["n"])
    with MANUAL_PINS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["landmark", "method", "n_reports", "sample_phrase",
                    "assumed_lat", "assumed_lon", "real_lat", "real_lon", "note"])
        for (anchor, method), g in ranked:
            lon, lat = _to_wgs.transform(*g["pos"])
            w.writerow([anchor, method, g["n"], g["phrase"],
                        round(lat, 4), round(lon, 4), "", "", ""])

    print(f"-> {MANUAL_PINS_CSV}  ({len(ranked)} groups; fill real_lat/real_lon for the top ones)\n")
    print(f"{'landmark':15s}{'method':12s}{'n':>3}  sample phrase")
    cum = 0
    for (anchor, method), g in ranked:
        cum += g["n"]
        print(f"{anchor:15s}{method:12s}{g['n']:>3}  {g['phrase']}")
    print(f"\nTotal in-grid reports: {cum}. The top 3 rows cover "
          f"{sum(g['n'] for _, g in ranked[:3])} of them.")


if __name__ == "__main__":
    main()
