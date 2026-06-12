"""Phase 2: build the place-name gazetteer from Phase-1 landmarks via Kartverket SSR.

Reads the distinct landmark_phrases out of data/interim/observations.csv, geocodes
each to EPSG:25832 (radius-only disambiguation around the Lordalen anchor), and writes:
    data/gazetteer/gazetteer.csv      name -> east,north,uncertainty_m,... (version-controlled)
    data/gazetteer/unresolved.txt     landmarks with no in-area SSR match (manual review)

Usage (repo root, venv active):
    python scripts/build_gazetteer.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from reindeer.geocode.ssr import (  # noqa: E402
    geocode, result_to_row, _session, DEFAULT_RADIUS_KM,
)

INTERIM = _ROOT / "data" / "interim"
GAZ = _ROOT / "data" / "gazetteer"


def distinct_landmarks() -> tuple[Counter, set]:
    """Return (mention counts per landmark, set of landmarks seen in Reinheimen)."""
    df = pd.read_csv(INTERIM / "observations.csv")
    c: Counter = Counter()
    reinheimen: set = set()
    for _, row in df.iterrows():
        if pd.isna(row["landmark_phrases"]):
            continue
        lms = json.loads(row["landmark_phrases"])
        for lm in lms:
            c[lm] += 1
            if str(row["region"]).lower().startswith("reinheimen"):
                reinheimen.add(lm)
    return c, reinheimen


def main() -> None:
    GAZ.mkdir(parents=True, exist_ok=True)
    counts, reinheimen = distinct_landmarks()
    names = sorted(counts, key=lambda n: (-counts[n], n))
    print(f"Geocoding {len(names)} distinct landmarks...", flush=True)

    session = _session()
    rows, unresolved = [], []
    for i, name in enumerate(names, 1):
        res = geocode(name, session)
        row = result_to_row(res)
        row["n_observations"] = counts[name]
        rows.append(row)
        if res.status in ("not_found", "needs_review"):
            cand = f" (weak candidate: {res.matched_name})" if res.matched_name else ""
            unresolved.append(f"{counts[name]:3}  {name:22} [{res.status}]{cand}")
        flag = "" if res.status == "ok" else f"  <{res.status}>"
        print(f"  [{i}/{len(names)}] {name:22} -> {res.status:9} "
              f"{res.matched_name or '-'} ({res.kommune or '-'}){flag}", flush=True)

    df = pd.DataFrame(rows)
    cols = ["query", "status", "matched_name", "feature_type", "kommune",
            "east", "north", "dist_km", "uncertainty_m", "n_in_area", "n_observations"]
    df = df[cols]
    df.to_csv(GAZ / "gazetteer.csv", index=False, encoding="utf-8")
    (GAZ / "unresolved.txt").write_text(
        ("Landmarks with no SSR hit within the radius (manual review):\n\n"
         + "\n".join(unresolved)) if unresolved
        else "All landmarks resolved to an in-area SSR point.\n",
        encoding="utf-8")

    resolved_states = ("ok", "ambiguous", "fuzzy")

    def cov(sub: pd.DataFrame, label: str) -> None:
        n = len(sub)
        if n == 0:
            return
        res = sub.status.isin(resolved_states).sum()
        vc = sub.status.value_counts().to_dict()
        print(f"  {label}: {res}/{n} resolved ({res/n*100:.0f}%)  "
              f"[ok={vc.get('ok',0)}, ambiguous={vc.get('ambiguous',0)}, "
              f"fuzzy={vc.get('fuzzy',0)}, needs_review={vc.get('needs_review',0)}, "
              f"not_found={vc.get('not_found',0)}]")

    print(f"\nGazetteer -> {GAZ/'gazetteer.csv'}")
    cov(df, "all landmarks      ")
    cov(df[df["query"].isin(reinheimen)], "Reinheimen landmarks")
    print(f"  (Reinheimen is the target field; Breheimen names sit outside the "
          f"{int(DEFAULT_RADIUS_KM)} km Lordalen radius.)")


if __name__ == "__main__":
    main()
