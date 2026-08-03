"""Phase 3: distance-to-disturbance layer onto the 250 m grid.

Disturbance = human access/presence that reindeer avoid: roads, tracks, marked
trails, parking, cabins, camping. Sources:
  - OSM (Overpass) roads / paths / parking / huts -> data/raw/osm/disturbance.json
  - the cached Lesja Fjellstyre KML cabins + camping (data/reference/...kml)

Per cell we store the distance (m, EPSG:25832) to the NEAREST disturbance feature.
The scorer turns that distance into a penalty via a tunable decay (DISTURB_DECAY_M),
encoding the hunter's "they come lower only if hunters allow".

v1 treats all feature types equally (distance to nearest of any). Weighting by road
class / trail importance is a later refinement (see IDEAS).
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import shapely
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString, Point

_ROOT = Path(__file__).resolve().parents[3]
OSM_JSON = _ROOT / "data" / "raw" / "osm" / "disturbance.json"
KML = _ROOT / "data" / "reference" / "lesja_lordalen_dalsida_area.kml"
_KML_NS = "{http://www.opengis.net/kml/2.2}"

_to32 = Transformer.from_crs(4326, 25832, always_xy=True)


def load_osm_disturbance(path: Path = OSM_JSON) -> list:
    """OSM ways -> LineStrings, nodes -> Points, all reprojected to EPSG:25832."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    geoms = []
    for e in d.get("elements", []):
        if e.get("type") == "node" and "lat" in e:
            geoms.append(Point(_to32.transform(e["lon"], e["lat"])))
        elif e.get("type") == "way" and e.get("geometry"):
            coords = [_to32.transform(p["lon"], p["lat"]) for p in e["geometry"]]
            if len(coords) >= 2:
                geoms.append(LineString(coords))
            elif coords:
                geoms.append(Point(coords[0]))
    return geoms


def load_kml_disturbance(path: Path = KML,
                         folders=("Hytter", "Camping")) -> list:
    """Cabin + camping points from the fjellstyre KML, reprojected to EPSG:25832."""
    tree = ET.parse(path)
    geoms = []
    for folder in tree.iter(f"{_KML_NS}Folder"):
        nm = folder.find(f"{_KML_NS}name")
        if nm is None or nm.text not in folders:
            continue
        for c in folder.findall(f".//{_KML_NS}Point/{_KML_NS}coordinates"):
            if c.text:
                lon, lat, *_ = c.text.strip().split(",")
                geoms.append(Point(_to32.transform(float(lon), float(lat))))
    return geoms


def nearest_distance(east, north, geoms) -> np.ndarray:
    """Distance (m) from each (east, north) cell centroid to the nearest geom."""
    tree = STRtree(geoms)
    pts = shapely.points(np.asarray(east, float), np.asarray(north, float))
    idx, dist = tree.query_nearest(pts, all_matches=False, return_distance=True)
    out = np.full(len(pts), np.nan)
    out[idx[0]] = dist
    return out
