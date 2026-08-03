"""Phase 3: forage-quality layer onto the 250 m grid, from NIBIO AR50 land cover.

Source (CLAUDE.md): NIBIO AR50 (arealressurskart 1:50 000), WFS coverage `ms:AR50`
on https://wfs.nibio.no/cgi-bin/ar50_2. The service serves GML (no GeoJSON), so we
request WFS 1.0.0 (bbox in easting,northing; simple GML2 <gml:coordinates>) directly
in EPSG:25832 and parse it. Each polygon carries an `arealtype` land-cover class; we
map that class to a 0..1 forage value for reindeer and assign each grid cell the value
of the polygon it falls in (point-in-polygon via a shapely STRtree).

v1 uses arealtype only. The finer `vegetasjonsdekke` field (bare rock vs vegetated
within open alpine ground) is a documented refinement — see IDEAS.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import requests
import shapely
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import Polygon

# The AR50 WFS returns GML geometry in WGS84 lon/lat (it honors srsName for the bbox
# filter but not for output), so we do the point-in-polygon join in 4326.
_to_ar50 = Transformer.from_crs(25832, 4326, always_xy=True)

WFS = "https://wfs.nibio.no/cgi-bin/ar50_2"
TYPENAME = "ms:AR50"
USER_AGENT = "reindeer-heatmap/0.1 (personal hunting research)"
GML = "{http://www.opengis.net/gml}"
MS = "{http://mapserver.gis.umn.edu/mapserver}"

_ROOT = Path(__file__).resolve().parents[3]
AR50_DIR = _ROOT / "data" / "raw" / "ar50"
AR50_GML = AR50_DIR / "ar50_area.gml"

# AR50 study-area bbox in EPSG:25832 (union of the two fields + margin).
BBOX_25832 = (447000, 6874000, 517000, 6917000)

# arealtype land-cover class -> forage value (0..1) for reindeer [expert-set, tunable].
FORAGE_BY_AREALTYPE = {
    50: 0.85,  # open alpine ground (snaumark) - prime summer range: lichen heath, grass/sedge
    60: 0.65,  # mire/bog (myr) - sedges, green forage
    30: 0.45,  # forest (skog) - some forage, less preferred
    20: 0.30,  # agriculture (jordbruk) - grass but low/disturbed
    10: 0.05,  # built-up (bebygd)
    70: 0.00,  # snow/ice/glacier (snø/isbre) - no forage (cooling value handled elsewhere)
    81: 0.00,  # freshwater
    82: 0.00,  # sea
    99: 0.50,  # not mapped - neutral
}
NEUTRAL_FORAGE = 0.50   # cells in no polygon / unknown class


def download_ar50(out: Path = AR50_GML, timeout: int = 300) -> Path:
    """Download the AR50 GML for the study bbox (cached)."""
    AR50_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 10000:
        return out
    minx, miny, maxx, maxy = BBOX_25832
    params = {
        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
        "typeName": TYPENAME, "srsName": "EPSG:25832",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:25832",
        "maxFeatures": 200000,
    }
    r = requests.get(WFS, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def _ring(coords_el) -> list | None:
    if coords_el is None or not coords_el.text:
        return None
    pts = []
    for tok in coords_el.text.split():
        x, y = tok.split(",")[:2]
        pts.append((float(x), float(y)))
    return pts


def load_ar50_polygons(path: Path = AR50_GML) -> tuple[list, np.ndarray]:
    """Parse the AR50 GML -> (list of shapely polygons, array of arealtype codes)."""
    tree = ET.parse(path)
    polys, types = [], []
    for feat in tree.iter(f"{MS}AR50"):
        at_el = feat.find(f"{MS}arealtype")
        if at_el is None or not at_el.text:
            continue
        at = int(at_el.text)
        for poly_el in feat.iter(f"{GML}Polygon"):
            shell = _ring(poly_el.find(
                f"{GML}outerBoundaryIs/{GML}LinearRing/{GML}coordinates"))
            if not shell or len(shell) < 4:
                continue
            holes = [h for h in (
                _ring(c) for c in poly_el.findall(
                    f"{GML}innerBoundaryIs/{GML}LinearRing/{GML}coordinates"))
                if h and len(h) >= 4]
            try:
                g = Polygon(shell, holes)
                if not g.is_valid:
                    g = g.buffer(0)
                if not g.is_empty:
                    polys.append(g)
                    types.append(at)
            except Exception:
                continue
    return polys, np.asarray(types, dtype=int)


def forage_to_grid(east, north, polys, types,
                   mapping=FORAGE_BY_AREALTYPE) -> tuple[np.ndarray, np.ndarray]:
    """For each (east, north) cell, return (arealtype code, forage value).

    Cells falling in no polygon get arealtype -1 and NEUTRAL_FORAGE.
    """
    lon, lat = _to_ar50.transform(np.asarray(east, float), np.asarray(north, float))
    tree = STRtree(polys)
    pts = shapely.points(lon, lat)
    pt_idx, poly_idx = tree.query(pts, predicate="within")  # point within polygon
    at = np.full(len(pts), -1, dtype=int)
    at[pt_idx] = types[poly_idx]
    forage = np.array([mapping.get(int(a), NEUTRAL_FORAGE) for a in at], dtype=float)
    return at, forage
