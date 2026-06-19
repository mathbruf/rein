"""Phase 2->3 bridge: load the resolved gazetteer into an in-memory name->coordinate map.

The gazetteer (data/gazetteer/gazetteer.csv, produced in Phase 2) resolves every
observed landmark to an EPSG:25832 point with an uncertainty estimate. Phase 3 uses
this map to attach observations to grid cells for validation, and to sanity-check the
scoring surface against where animals were actually reported.

Locatability tiers (docs/phase3_starting_point.md):
    P1 ok        exact, single in-area hit
    P2 ambiguous exact, nearest of several in-area
    P3 fuzzy     name-verified dialect spelling
    P4 needs_review / not_found   -> excluded by default (down-prioritised)

By default we load only the resolved entries (P1-P3).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
GAZETTEER_CSV = _ROOT / "data" / "gazetteer" / "gazetteer.csv"

RESOLVED_STATES = ("ok", "ambiguous", "fuzzy")  # P1-P3


@dataclass(frozen=True)
class GazEntry:
    name: str            # the landmark phrase as observed (gazetteer `query`)
    east: float          # EPSG:25832
    north: float
    uncertainty_m: float
    status: str          # ok | ambiguous | fuzzy | needs_review | not_found
    feature_type: str | None
    matched_name: str | None


def load_gazetteer(path: Path = GAZETTEER_CSV,
                   include_unresolved: bool = False) -> dict[str, GazEntry]:
    """Return {landmark_name -> GazEntry} in EPSG:25832.

    Only P1-P3 (ok/ambiguous/fuzzy) are loaded unless include_unresolved=True.
    Rows without coordinates (not_found) are always skipped.
    """
    out: dict[str, GazEntry] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row["status"]
            if not include_unresolved and status not in RESOLVED_STATES:
                continue
            if not row["east"] or not row["north"]:
                continue
            out[row["query"]] = GazEntry(
                name=row["query"],
                east=float(row["east"]),
                north=float(row["north"]),
                uncertainty_m=float(row["uncertainty_m"]) if row["uncertainty_m"] else 0.0,
                status=status,
                feature_type=row["feature_type"] or None,
                matched_name=row["matched_name"] or None,
            )
    return out
