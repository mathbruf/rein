"""Phase 3: derive per-cell terrain attributes from the DTM onto the 250 m grid.

From a 50 m DTM (EPSG:25833) we derive, then sample onto each 250 m grid cell:
  elevation_m   mean elevation in the cell          -> "high & cool/breezy" signal
  slope_deg     mean slope                           -> very steep = travel-limiting
  ruggedness_m  std of elevation within the cell     -> broken vs smooth ground
  tpi_m         topographic position: cell elevation minus mean elevation in a
                ~1 km window -> ridge/summit/exposed (+) vs valley/hollow (-).
                This is the core wind-exposure / insect-escape signal.

These are STATIC layers. Weather is applied later by the scoring engine; nothing
here depends on the forecast. The DTM is EPSG:25833, so grid centroids (25832) are
reprojected to 25833 before sampling.
"""
from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

_to_dem = Transformer.from_crs(25832, 25833, always_xy=True)


def load_dtm(path) -> tuple[np.ndarray, np.ndarray, object, float]:
    """Return (Z float64 with NaN where invalid, valid mask, affine transform, res_m)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True)
        valid = ~np.ma.getmaskarray(arr)
        Z = arr.filled(np.nan).astype(np.float64)
        res = float((abs(ds.res[0]) + abs(ds.res[1])) / 2)
        return Z, valid, ds.transform, res


def slope_deg(Z: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(Z, res, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _box_mean(Z: np.ndarray, valid: np.ndarray, win: int) -> np.ndarray:
    """Mean of valid pixels in a (win x win) window, via an integral image.
    Border windows are shrunk to the available pixels (divide by actual count)."""
    Zf = np.where(valid, Z, 0.0)
    H, W = Z.shape
    S = np.zeros((H + 1, W + 1))
    C = np.zeros((H + 1, W + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(Zf, 0), 1)
    C[1:, 1:] = np.cumsum(np.cumsum(valid.astype(np.float64), 0), 1)
    r = win // 2
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    r0 = np.clip(rows - r, 0, H)
    r1 = np.clip(rows + r + 1, 0, H)
    c0 = np.clip(cols - r, 0, W)
    c1 = np.clip(cols + r + 1, 0, W)
    ssum = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
    scnt = C[r1, c1] - C[r0, c1] - C[r1, c0] + C[r0, c0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scnt > 0, ssum / scnt, np.nan)


def tpi(Z: np.ndarray, valid: np.ndarray, res: float, window_m: float = 1000.0) -> np.ndarray:
    win = max(3, round(window_m / res) | 1)  # odd window
    return Z - _box_mean(Z, valid, win)


def sample_to_grid(east: np.ndarray, north: np.ndarray,
                   Z: np.ndarray, slope: np.ndarray, tpi_r: np.ndarray,
                   transform, block_px: int = 5) -> dict[str, np.ndarray]:
    """Sample terrain onto cells. east/north are EPSG:25832 cell centroids.

    Per cell: mean elevation/slope and std-elevation (ruggedness) over the
    block_px x block_px DTM window covering the 250 m cell; tpi at the centroid.
    """
    X, Y = _to_dem.transform(east, north)
    rows, cols = rowcol(transform, X, Y)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    H, Wd = Z.shape
    half = block_px // 2
    n = east.size
    elev = np.full(n, np.nan)
    slp = np.full(n, np.nan)
    rug = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        if not (0 <= r < H and 0 <= c < Wd):
            continue
        r0, r1 = max(0, r - half), min(H, r + half + 1)
        c0, c1 = max(0, c - half), min(Wd, c + half + 1)
        blk = Z[r0:r1, c0:c1]
        if np.all(np.isnan(blk)):
            continue
        elev[i] = np.nanmean(blk)
        rug[i] = np.nanstd(blk)
        slp[i] = np.nanmean(slope[r0:r1, c0:c1])
        tp[i] = tpi_r[r, c]
    return {"elevation_m": elev, "slope_deg": slp,
            "ruggedness_m": rug, "tpi_m": tp}
