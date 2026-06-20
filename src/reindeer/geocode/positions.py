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

import json
import math

DEFAULT_OFFSET_M = 3000.0

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
                     offset_m: float = DEFAULT_OFFSET_M):
    """Return (east, north, method) for an observation, or None if no anchor resolves.

    method is one of: 'mot', 'offset', 'at-landmark'.
    """
    lms = landmark_phrases if isinstance(landmark_phrases, list) else json.loads(landmark_phrases or "[]")
    hints = direction_hints if isinstance(direction_hints, list) else json.loads(direction_hints or "[]")

    anchor = next((gaz[lm] for lm in lms if lm in gaz), None)
    if anchor is None:
        return None
    ax, ay = anchor.east, anchor.north

    # 1) "mot Y" — head towards another resolved landmark: take the midpoint
    for h in hints:
        if _fold(h.get("relation", "")) == "mot":
            tgt = gaz.get(h.get("landmark"))
            if tgt is not None and (tgt.east, tgt.north) != (ax, ay):
                return (ax + 0.5 * (tgt.east - ax), ay + 0.5 * (tgt.north - ay), "mot")

    # 2) compass offset from a "<dir> for" relation or a "<dir>over" bearing
    for h in hints:
        v = _direction(h.get("relation", "")) or _direction(h.get("bearing", ""))
        if v is not None:
            return (ax + v[0] * offset_m, ay + v[1] * offset_m, "offset")

    # 3) no usable direction -> at the landmark
    return (ax, ay, "at-landmark")
