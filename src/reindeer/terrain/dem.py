"""Phase 3: fetch the Kartverket national DTM for the study area via WCS.

Source (CLAUDE.md data table): Kartverket / høydedata.no. We use the WCS coverage
`nhm_dtm_topo_25833` on the Geonorge endpoint below. The service serves only
EPSG:25833 (UTM 33N) or 4326 — NOT our project CRS 25832 — so we request the raster
in 25833 and reproject cell centroids 25832->25833 when sampling (see terrain.py).
The server resamples to the requested resolution, so a coarse 50 m grid is one small,
polite GeoTIFF request that is ample for a 250 m output grid.
"""
from __future__ import annotations

from pathlib import Path

import requests
from pyproj import Transformer

from .grid import load_field_polygons, LORDALEN, DALSIDA

WCS = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
COVERAGE = "nhm_dtm_topo_25833"
DEM_EPSG = 25833
USER_AGENT = "reindeer-heatmap/0.1 (personal hunting research)"

_ROOT = Path(__file__).resolve().parents[3]
DEM_DIR = _ROOT / "data" / "raw" / "dem"

# project grid CRS -> DTM CRS
_to_dem = Transformer.from_crs(25832, DEM_EPSG, always_xy=True)


def request_bbox_25833(buffer_m: float = 2000.0) -> tuple[float, float, float, float]:
    """Bounding box (in EPSG:25833) that fully covers the union of both field
    polygons, plus a margin. Computed by transforming every polygon vertex to 25833
    (so the 32->33 zone rotation is captured), then taking min/max + buffer."""
    polys = load_field_polygons()
    xs, ys = [], []
    for name in (LORDALEN, DALSIDA):
        for x, y in polys[name].exterior.coords:
            X, Y = _to_dem.transform(x, y)
            xs.append(X)
            ys.append(Y)
    return (min(xs) - buffer_m, min(ys) - buffer_m,
            max(xs) + buffer_m, max(ys) + buffer_m)


def download_dtm(res_m: int = 50, out: Path | None = None,
                 timeout: int = 240) -> Path:
    """Download the DTM GeoTIFF for the study area at `res_m` resolution (cached)."""
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    out = out or DEM_DIR / f"dtm_{res_m}m_25833.tif"
    if out.exists() and out.stat().st_size > 1000:
        return out

    minx, miny, maxx, maxy = request_bbox_25833()
    width = round((maxx - minx) / res_m)
    height = round((maxy - miny) / res_m)
    params = {
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": COVERAGE, "CRS": f"EPSG:{DEM_EPSG}",
        "BBOX": f"{minx:.1f},{miny:.1f},{maxx:.1f},{maxy:.1f}",
        "WIDTH": width, "HEIGHT": height, "FORMAT": "GeoTIFF",
    }
    print(f"WCS GetCoverage {COVERAGE} @ {res_m} m -> {width}x{height} px")
    print(f"  bbox 25833: {params['BBOX']}")
    r = requests.get(WCS, params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    if "xml" in ct or r.content[:5] == b"<?xml":
        raise RuntimeError(f"WCS returned an exception:\n{r.text[:600]}")
    out.write_bytes(r.content)
    return out
