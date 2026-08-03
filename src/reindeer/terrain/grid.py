"""Phase 3: build the 250 m scoring grid in EPSG:25832, clipped to the field polygons.

The reported heatmap covers the **Lordalen Statsallmenning** polygon; the **Dalsida
Statsallmenning** polygon to the north is included as an extended buffer for
cross-field movement context (tagged `in_dalsida`, not part of the primary reported
surface). See docs/phase3_starting_point.md.

Geometry source: data/reference/lesja_lordalen_dalsida_area.kml (Lesja Fjellstyre
Google My Maps export, WGS84 lon/lat). Reprojected to EPSG:25832 on load.

Rules (decided 2026-06-12):
- A cell belongs to a polygon when its **centroid** lies inside that polygon.
- The grid origin is snapped to a multiple of the cell size so the lattice is
  reproducible and will align with future raster layers (DEM, AR5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

OUT_EPSG = 25832
CELL_SIZE_M = 250

_ROOT = Path(__file__).resolve().parents[3]
KML_PATH = _ROOT / "data" / "reference" / "lesja_lordalen_dalsida_area.kml"

_KML_NS = "{http://www.opengis.net/kml/2.2}"
LORDALEN = "Lordalen Statsallmenning"
DALSIDA = "Dalsida Statsallmenning"

# WGS84 (KML) -> ETRS89 / UTM 32N. always_xy keeps lon,lat / east,north order.
_to_utm = Transformer.from_crs(4326, OUT_EPSG, always_xy=True)


def _parse_polygon(coords_text: str) -> Polygon:
    pts = []
    for tok in coords_text.split():
        lon, lat, *_ = tok.split(",")
        east, north = _to_utm.transform(float(lon), float(lat))
        pts.append((east, north))
    return Polygon(pts)


def load_field_polygons(kml_path: Path = KML_PATH) -> dict[str, Polygon]:
    """Return {field_name: shapely Polygon in EPSG:25832} for the two statsallmenning."""
    tree = ET.parse(kml_path)
    out: dict[str, Polygon] = {}
    for pm in tree.iter(f"{_KML_NS}Placemark"):
        name_el = pm.find(f"{_KML_NS}name")
        if name_el is None or name_el.text not in (LORDALEN, DALSIDA):
            continue
        coords_el = pm.find(
            f"{_KML_NS}Polygon/{_KML_NS}outerBoundaryIs/"
            f"{_KML_NS}LinearRing/{_KML_NS}coordinates")
        if coords_el is None or not coords_el.text:
            continue
        out[name_el.text] = _parse_polygon(coords_el.text)
    missing = {LORDALEN, DALSIDA} - out.keys()
    if missing:
        raise ValueError(f"KML is missing expected polygons: {sorted(missing)}")
    return out


@dataclass
class Cell:
    cell_id: int
    col: int
    row: int
    east: float          # centroid easting (EPSG:25832)
    north: float         # centroid northing
    in_lordalen: bool    # inside the primary reported field
    in_dalsida: bool     # inside the extended-pattern buffer


def build_grid(cell_size: int = CELL_SIZE_M,
               kml_path: Path = KML_PATH) -> list[Cell]:
    """Build the clipped 250 m grid. Cells outside both polygons are dropped."""
    polys = load_field_polygons(kml_path)
    lord, dals = polys[LORDALEN], polys[DALSIDA]
    union = lord.union(dals)
    minx, miny, maxx, maxy = union.bounds

    # snap origin down to a clean multiple of cell_size (reproducible lattice)
    x0 = math.floor(minx / cell_size) * cell_size
    y0 = math.floor(miny / cell_size) * cell_size
    ncols = math.ceil((maxx - x0) / cell_size)
    nrows = math.ceil((maxy - y0) / cell_size)

    p_lord, p_dals = prep(lord), prep(dals)
    cells: list[Cell] = []
    cid = 0
    for r in range(nrows):
        cy = y0 + (r + 0.5) * cell_size
        for c in range(ncols):
            cx = x0 + (c + 0.5) * cell_size
            pt = Point(cx, cy)
            in_l = p_lord.contains(pt)
            in_d = p_dals.contains(pt)
            if not (in_l or in_d):
                continue
            cells.append(Cell(cid, c, r, cx, cy, in_l, in_d))
            cid += 1
    return cells
