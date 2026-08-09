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


# Feature categories for MAP DISPLAY (viz/render.py): the same cached data, but
# kept apart so roads, tracks, trails and cabins can be drawn distinctly. Purely
# for orientation on the rendered map; the disturbance *layer* above is unchanged.
_ROAD_MAJOR = {"trunk", "primary", "secondary", "tertiary"}
_ROAD_MINOR = {"unclassified", "residential", "service", "raceway"}
_TRACK = {"track"}
_PATH = {"path", "footway", "bridleway", "cycleway", "steps"}
_HUT_TAGS = {"alpine_hut", "wilderness_hut", "chalet", "camp_site", "cabin", "hut"}


def load_overlay_features(osm_path: Path = OSM_JSON, kml_path: Path = KML) -> dict:
    """Categorised human features for drawing on the map (EPSG:25832).

    Returns {'road_major': [LineString], 'road_minor': [...], 'track': [...],
             'path': [...], 'cabin': [(Point, name|None)], 'parking': [(Point,
             name|None)]}. Points carry their name (OSM `name` tag / KML placemark
    name) so the renderer can label the important trailheads and cabins.
    Building/parking outlines collapse to centroid points (they are markers on a
    field-scale map). Missing source files simply yield empty lists.
    """
    out = {"road_major": [], "road_minor": [], "track": [], "path": [],
           "cabin": [], "parking": []}
    p = Path(osm_path)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for e in d.get("elements", []):
            t = e.get("tags", {})
            hw = t.get("highway")
            name = t.get("name")
            hutish = (t.get("tourism") in _HUT_TAGS or t.get("building") in _HUT_TAGS
                      or t.get("amenity") == "shelter")
            parking = t.get("amenity") == "parking"
            if e.get("type") == "node" and "lat" in e:
                pt = Point(_to32.transform(e["lon"], e["lat"]))
                if hutish:
                    out["cabin"].append((pt, name))
                elif parking:
                    out["parking"].append((pt, name))
            elif e.get("type") == "way" and e.get("geometry"):
                coords = [_to32.transform(q["lon"], q["lat"]) for q in e["geometry"]]
                if len(coords) < 2:
                    continue
                line = LineString(coords)
                if hw in _ROAD_MAJOR:
                    out["road_major"].append(line)
                elif hw in _ROAD_MINOR:
                    out["road_minor"].append(line)
                elif hw in _TRACK:
                    out["track"].append(line)
                elif hw in _PATH:
                    out["path"].append(line)
                elif hutish:
                    out["cabin"].append((line.centroid, name))
                elif parking:
                    out["parking"].append((line.centroid, name))
    kp = Path(kml_path)
    if kp.exists():
        tree = ET.parse(kp)
        for folder in tree.iter(f"{_KML_NS}Folder"):
            nm = folder.find(f"{_KML_NS}name")
            if nm is None or nm.text not in ("Hytter", "Camping"):
                continue
            for pm in folder.findall(f".//{_KML_NS}Placemark"):
                c = pm.find(f".//{_KML_NS}Point/{_KML_NS}coordinates")
                if c is None or not c.text:
                    continue
                lon, lat, *_ = c.text.strip().split(",")
                pname = pm.find(f"{_KML_NS}name")
                out["cabin"].append((Point(_to32.transform(float(lon), float(lat))),
                                     pname.text if pname is not None else None))
    return out


def nearest_distance(east, north, geoms) -> np.ndarray:
    """Distance (m) from each (east, north) cell centroid to the nearest geom."""
    tree = STRtree(geoms)
    pts = shapely.points(np.asarray(east, float), np.asarray(north, float))
    idx, dist = tree.query_nearest(pts, all_matches=False, return_distance=True)
    out = np.full(len(pts), np.nan)
    out[idx[0]] = dist
    return out
