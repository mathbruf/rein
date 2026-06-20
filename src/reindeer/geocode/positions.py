"""Phase 5: turn an observation's landmark + directional phrase into a position.

The sightings describe animals *relative* to a named landmark, not *at* it:
"nord for Lordalen mot Tverrfjellet" = north of the Lordalen valley, towards
Tverrfjellet. Placing the observation at the landmark point is systematically wrong
(named features that resolve well are mostly valleys/lakes, which sit low and near
roads), so this module applies the directional phrase:

  - "mot Y" (towards Y, Y resolves)      -> midpoint between the anchor and Y
  - "<dir> for X" / bearing "<dir>over"  -> offset `offset_m` from the anchor in <dir>
  - "i området" / "ved" / "innanfor" / none -> at the anchor (no offset)

`offset_m` is fixed a priori (not tuned to the validation score); the validation
script reports a sensitivity sweep over it to show the result isn't cherry-picked.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pyproj import Transformer

DEFAULT_OFFSET_M = 3000.0

_ROOT = Path(__file__).resolve().parents[3]
MANUAL_PINS_CSV = _ROOT / "data" / "gazetteer" / "manual_positions.csv"
_to_utm = Transformer.from_crs(4326, 25832, always_xy=True)


def load_manual_pins(path: Path = MANUAL_PINS_CSV) -> dict:
    """Human-pinned real positions, keyed (landmark.lower(), method) -> (east, north).

    The template has assumed_lat/lon (mine) and real_lat/real_lon (yours, blank until
    filled). Only rows with both real_lat and real_lon set are used. A method of 'any'
    matches every method for that landmark.
    """
    pins: dict = {}
    p = Path(path)
    if not p.exists():
        return pins
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rlat, rlon = row.get("real_lat", "").strip(), row.get("real_lon", "").strip()
            if not rlat or not rlon:
                continue
            east, north = _to_utm.transform(float(rlon), float(rlat))
            pins[(row["landmark"].lower(), row.get("method", "any").strip() or "any")] = (east, north)
    return pins

# compass unit vectors in EPSG:25832 (east, north), keys folded (ø->o, å->a, æ->ae)
_DIAG = math.sqrt(0.5)
_COMPASS = {
    "nord": (0.0, 1.0), "sor": (0.0, -1.0), "aust": (1.0, 0.0), "vest": (-1.0, 0.0),
    "nordaust": (_DIAG, _DIAG), "nordvest": (-_DIAG, _DIAG),
    "soraust": (_DIAG, -_DIAG), "sorvest": (-_DIAG, -_DIAG),
}


def _fold(s: str) -> str:
    s = (s or "").lower().strip()
    for a, b in (("ø", "o"), ("å", "a"), ("æ", "ae")):
        s = s.replace(a, b)
    return s


def _direction(token: str) -> tuple[float, float] | None:
    """Map a relation/bearing token to a compass unit vector, or None."""
    t = _fold(token)
    if not t:
        return None
    t = t.replace("over", "")              # austover -> aust, nordover -> nord
    t = t.split(" ")[0]                    # "nord for" -> "nord"
    return _COMPASS.get(t)


def resolve_position(landmark_phrases, direction_hints, gaz,
                     offset_m: float = DEFAULT_OFFSET_M, pins: dict | None = None):
    """Return (east, north, method) for an observation, or None if no anchor resolves.

    method is one of: 'mot', 'offset', 'at-landmark' (suffixed '-pinned' when a manual
    pin overrode the derived position). If `pins` is given and the anchor landmark has a
    human-pinned real position (by method or 'any'), that exact position is used instead
    of the directional heuristic — removing the offset assumption.
    """
    lms = landmark_phrases if isinstance(landmark_phrases, list) else json.loads(landmark_phrases or "[]")
    hints = direction_hints if isinstance(direction_hints, list) else json.loads(direction_hints or "[]")

    anchor = next((gaz[lm] for lm in lms if lm in gaz), None)
    if anchor is None:
        return None
    ax, ay = anchor.east, anchor.north

    # determine the derived method first (so a pin can be keyed by it)
    method, pos = "at-landmark", (ax, ay)
    for h in hints:
        if _fold(h.get("relation", "")) == "mot":
            tgt = gaz.get(h.get("landmark"))
            if tgt is not None and (tgt.east, tgt.north) != (ax, ay):
                method, pos = "mot", (ax + 0.5 * (tgt.east - ax), ay + 0.5 * (tgt.north - ay))
                break
    else:
        for h in hints:
            v = _direction(h.get("relation", "")) or _direction(h.get("bearing", ""))
            if v is not None:
                method, pos = "offset", (ax + v[0] * offset_m, ay + v[1] * offset_m)
                break

    if pins:
        key = anchor.name.lower()
        pin = pins.get((key, method)) or pins.get((key, "any"))
        if pin is not None:
            return (pin[0], pin[1], method + "-pinned")
    return (pos[0], pos[1], method)
