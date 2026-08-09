"""Make/refresh the manual-pin template for validation positions (MERGE, never lose pins).

Updates data/gazetteer/manual_positions.csv: one row per (landmark, method) group of
in-grid reports, ranked by how many reports depend on it, pre-filled with my *assumed*
position (assumed_lat/lon) and BLANK real_lat/real_lon for you to fill from the map.
Reports carry no GPS, so filling real_lat/real_lon for the top rows removes the
positional assumption from the validation (resolve_position then uses your pin).

MERGE SEMANTICS (2026-08-09): existing rows keep their human-filled real_lat/real_lon/
note untouched; only NEW (landmark, method) groups (e.g. from a newly harvested season)
are appended. Rerunning this script is always safe.

`at-landmark` rows are the VAGUE tier ("i området X" — the geocoder locates the *name*,
usually a valley/lake, not the herd) and are marked NEEDS-PIN in the printout: pinning
them moves their reports into the position-confident validation tier.

Read off coordinates in Google Maps: right-click the spot -> the lat,lon at the top of
the menu -> paste into real_lat, real_lon.

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

    # MERGE: never overwrite human-filled pins. Load the existing file (if any),
    # keep every existing row's real_lat/real_lon/note verbatim, refresh its report
    # count, and append only genuinely new (landmark, method) groups.
    existing: dict = {}
    if MANUAL_PINS_CSV.exists():
        with MANUAL_PINS_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[(row["landmark"], row["method"])] = row

    out_rows, n_new = [], 0
    for (anchor, method), g in ranked:
        lon, lat = _to_wgs.transform(*g["pos"])
        old = existing.get((anchor, method))
        if old is not None:
            old["n_reports"] = str(g["n"])   # refresh count; pins/notes untouched
            out_rows.append(old)
        else:
            n_new += 1
            out_rows.append({"landmark": anchor, "method": method,
                             "n_reports": str(g["n"]), "sample_phrase": g["phrase"],
                             "assumed_lat": round(lat, 4), "assumed_lon": round(lon, 4),
                             "real_lat": "", "real_lon": "", "note": ""})
    # keep any existing rows whose group vanished (e.g. parser change) — pins are precious
    for key, row in existing.items():
        if key not in groups:
            out_rows.append(row)

    fields = ["landmark", "method", "n_reports", "sample_phrase",
              "assumed_lat", "assumed_lon", "real_lat", "real_lon", "note"]
    with MANUAL_PINS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    n_pinned = sum(1 for r in out_rows if str(r.get("real_lat", "")).strip())
    print(f"-> {MANUAL_PINS_CSV}  ({len(out_rows)} groups: {n_pinned} pinned, "
          f"{n_new} new; existing pins preserved)\n")
    print(f"{'landmark':15s}{'method':12s}{'n':>3}  flag       sample phrase")
    cum = 0
    for (anchor, method), g in ranked:
        cum += g["n"]
        pinned = str(existing.get((anchor, method), {}).get("real_lat", "")).strip()
        flag = ("pinned    " if pinned else
                "NEEDS-PIN " if method == "at-landmark" else "          ")
        print(f"{anchor:15s}{method:12s}{g['n']:>3}  {flag} {g['phrase']}")
    print(f"\nTotal in-grid reports: {cum}. NEEDS-PIN rows are the vague tier "
          "('i området X'): pinning them moves their reports into the "
          "position-confident validation headline.")


if __name__ == "__main__":
    main()
