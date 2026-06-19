"""Phase 2: resolve landmark names to EPSG:25832 coordinates via Kartverket SSR.

Strategy (decided 2026-06-12): the SSR name search returns nationwide hits in no
geographic order, so we fetch *all* hits for a name and keep only those within a
radius of the Lordalen anchor, then pick the nearest. Names with no in-area hit are
flagged for manual review (local names SSR may not carry). Output point + an
uncertainty estimate derived from the feature type (a valley is fuzzier than a peak).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path

import requests

API = "https://api.kartverket.no/stedsnavn/v1/navn"
USER_AGENT = "reindeer-heatmap/0.1 (personal hunting research)"
OUT_EPSG = 25832

# Lordalen valley representation point (SSR), our regional anchor in EPSG:25832.
LORDALEN_ANCHOR = (474445.22856, 6884302.14348)
DEFAULT_RADIUS_KM = 30.0
DEFAULT_DELAY = 0.6  # be gentle with the API

_ROOT = Path(__file__).resolve().parents[3]
SSR_CACHE = _ROOT / "data" / "raw" / "ssr"

# Positional uncertainty (metres) by feature-type keyword. A single point stands in
# for an extended feature, so valleys/plateaus are far fuzzier than summits or lakes.
_UNCERTAINTY = [
    (("kommune",), 15000),
    (("dal", "dalen", "dalf"), 3000),
    (("flye", "vidde", "fjellomr", "hei"), 2500),
    (("fjell", "topp", "berg", "nut", "nose", "horrung", "varde", "kollen", "nibba"), 1200),
    (("vatn", "vatnet", "innsj", "tjern", "tjørn", "vann"), 800),
    (("elv", "bekk", "å", "os"), 1500),
    (("nes", "odde", "neset"), 600),
    (("seter", "støl", "gard", "bruk", "bu", "hytte"), 400),
]
_DEFAULT_UNCERTAINTY = 1500


def _uncertainty_m(feature_type: str) -> int:
    ft = (feature_type or "").lower()
    for keys, val in _UNCERTAINTY:
        if any(k in ft for k in keys):
            return val
    return _DEFAULT_UNCERTAINTY


def _dist_km(pt: tuple[float, float], anchor=LORDALEN_ANCHOR) -> float:
    return math.dist(pt, anchor) / 1000.0


# SSR direction/size qualifiers we strip before comparing names.
_QUALIFIER = re.compile(
    r"^(indre|ytre|nedre|øvre|øvste|nørdre|nordre|søre|søndre|vestre|austre|"
    r"store?|vesle|lille|litle|nord|sør|aust|vest)\s+", re.I)


def _norm(s: str) -> str:
    """Fold Norwegian dialect spelling so 'Nysætri' ~ 'Nysetra'."""
    s = _QUALIFIER.sub("", (s or "").lower().strip())
    for a, b in (("å", "a"), ("ø", "o"), ("æ", "ae"), ("é", "e"), ("aa", "a")):
        s = s.replace(a, b)
    # collapse interchangeable dialect endings (tjønne/tjern, -i/-a/-e, double cons.)
    s = re.sub(r"(tjønn|tjøn|tjern|tjønne)", "tjon", s)
    return s


def _name_similar(query: str, matched: str) -> bool:
    """A fuzzy hit is trustworthy only if its name actually resembles the query:
    same first letters AND a high folded-string ratio. Blocks 'Sotfly'->'Nordli'
    and 'Liafjellet'->'Tverrfjellet' while keeping 'Heggebottflye'->'Heggebottflyi'."""
    q, m = _norm(query), _norm(matched)
    if not q or not m:
        return False
    if len(m) < len(q) - 3:   # matched a short substring, e.g. 'Skjåkgrunn'->'Skjåk'
        return False
    ratio = SequenceMatcher(None, q, m).ratio()
    return ratio >= 0.80 or (q[:3] == m[:3] and ratio >= 0.62)


@dataclass
class GeocodeResult:
    query: str
    status: str               # "ok" | "ambiguous" | "not_found"
    matched_name: str | None = None
    feature_type: str | None = None
    kommune: str | None = None
    east: float | None = None
    north: float | None = None
    dist_km: float | None = None
    uncertainty_m: int | None = None
    n_in_area: int = 0        # candidates within the radius


def _fetch_all(sok: str, session: requests.Session, delay: float,
               fuzzy: bool = False) -> list[dict]:
    """Fetch every SSR hit for a search string (paginated), with on-disk caching."""
    SSR_CACHE.mkdir(parents=True, exist_ok=True)
    tag = "fuzzy_" if fuzzy else ""
    cache = SSR_CACHE / (tag + sok.replace("*", "_star").replace("/", "_") + ".json")
    if cache.exists() and cache.stat().st_size > 0:
        return json.loads(cache.read_text(encoding="utf-8"))
    hits: list[dict] = []
    side = 1
    while True:
        params = {"sok": sok, "utkoordsys": OUT_EPSG,
                  "treffPerSide": 500, "side": side,
                  "fuzzy": "true" if fuzzy else "false"}
        r = session.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        hits.extend(data.get("navn", []))
        meta = data.get("metadata", {})
        if meta.get("viserTil", 0) >= meta.get("totaltAntallTreff", 0):
            break
        side += 1
        time.sleep(delay)
    cache.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    time.sleep(delay)
    return hits


def _nearest_in_area(hits: list[dict], radius_km: float) -> tuple[dict | None, int]:
    in_area = []
    for h in hits:
        p = h.get("representasjonspunkt")
        if not p:
            continue
        pt = (p["øst"], p["nord"])
        d = _dist_km(pt)
        if d <= radius_km:
            in_area.append((d, h, pt))
    if not in_area:
        return None, 0
    in_area.sort(key=lambda x: x[0])
    return in_area[0], len(in_area)


def geocode(name: str, session: requests.Session | None = None,
            radius_km: float = DEFAULT_RADIUS_KM,
            delay: float = DEFAULT_DELAY) -> GeocodeResult:
    """Resolve one landmark name to a single best in-area point."""
    session = session or _session()
    # Tier 1 exact, tier 2 wildcard-prefix, tier 3 fuzzy. Fuzzy matches approximate
    # spellings (SSR "Heggebottflyi" vs local "Heggebottflye") but can over-match, so
    # we tag those for review rather than trusting them.
    tier = "exact"
    hits = _fetch_all(name, session, delay)
    best, n = _nearest_in_area(hits, radius_km)
    if best is None:
        tier = "wildcard"
        hits = _fetch_all(name + "*", session, delay)
        best, n = _nearest_in_area(hits, radius_km)
    if best is None:
        tier = "fuzzy"
        hits = _fetch_all(name, session, delay, fuzzy=True)
        best, n = _nearest_in_area(hits, radius_km)
    if best is None:
        return GeocodeResult(query=name, status="not_found")
    d, h, pt = best
    kommune = ",".join(k["kommunenavn"] for k in h.get("kommuner", []))
    ft = h.get("navneobjekttype")
    if tier == "fuzzy":
        # keep the weak candidate but mark it for review unless the name really matches
        status = "fuzzy" if _name_similar(name, h.get("skrivemåte", "")) else "needs_review"
    else:
        status = "ok" if n == 1 else "ambiguous"
    return GeocodeResult(
        query=name,
        status=status,
        matched_name=h.get("skrivemåte"),
        feature_type=ft,
        kommune=kommune,
        east=round(pt[0], 2),
        north=round(pt[1], 2),
        dist_km=round(d, 2),
        uncertainty_m=_uncertainty_m(ft),
        n_in_area=n,
    )


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def result_to_row(res: GeocodeResult) -> dict:
    return asdict(res)
