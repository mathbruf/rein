"""Phase 4: render a scored 250 m grid as a heatmap PNG.

Takes a scored table (east, north, score in EPSG:25832 on the 250 m lattice),
rasterises it to a 2D array, and draws a 0..1 heatmap with the field boundary
overlaid. Red = avoid / unlikely, green = strongly favored (RdYlGn), so a hunter
can read it at a glance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from reindeer.terrain.grid import CELL_SIZE_M, load_field_polygons, LORDALEN

CMAP = "RdYlGn"   # low score -> red (avoid), high -> green (favored)


def _rasterise(east, north, score, cell=CELL_SIZE_M):
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    xmin, ymin = east.min(), north.min()
    cols = np.round((east - xmin) / cell).astype(int)
    rows = np.round((north - ymin) / cell).astype(int)
    arr = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    arr[rows, cols] = np.asarray(score, float)
    # extent = outer pixel edges (centroids +/- half a cell)
    extent = [xmin - cell / 2, xmin + (cols.max() + 0.5) * cell,
              ymin - cell / 2, ymin + (rows.max() + 0.5) * cell]
    return arr, extent


def render_heatmap(east, north, score, out_png: Path,
                   title: str | None = None, top=None, dpi: int = 130) -> Path:
    """Render score over the grid to out_png. `top` optional (east, north) of marks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr, extent = _rasterise(east, north, score)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(arr, origin="lower", extent=extent, cmap=CMAP,
                   vmin=0, vmax=1, interpolation="nearest")

    # field boundary for context
    try:
        poly = load_field_polygons()[LORDALEN]
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="black", lw=1.2, alpha=0.7)
    except Exception:
        pass

    if top is not None:
        tx, ty = top
        ax.scatter(tx, ty, s=28, facecolors="none", edgecolors="black", lw=1.0)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (EPSG:25832)")
    ax.set_ylabel("Northing (EPSG:25832)")
    if title:
        ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("presence score (0 avoid - 1 favored)")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
