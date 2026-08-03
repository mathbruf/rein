"""Phase 4: render a scored 250 m grid as a human-readable heatmap.

The map is built for a hunter reading it the night before, so it leads with meaning,
not raw numbers:
  - a shaded-relief (hillshade) terrain background from the 50 m DTM, so ridges,
    valleys and passes are recognisable at a glance;
  - the 0..1 presence score as a single-hue green "glow" (a magnitude wears a
    sequential ramp, not red-green): low ground stays bare hillshade and colour
    builds only where the model favours the ground, pulling the eye to where to go.
    The wash is lightly smoothed for display so it reads as coherent zones rather
    than a per-250 m-cell mosaic (the scores/CSV themselves are unchanged);
  - a handful of numbered "go here" zones, clustered so they are distinct areas
    (not adjacent cells), each anchored to its nearest named landmark;
  - a side panel with the date, the day's weather in plain language, which driver is
    active (insect-escape vs shelter), and the ranked zone list with a short reason;
  - a plain-language legend, a scale bar and a north arrow.

Everything is EPSG:25832. `render_heatmap()` stays backward compatible with the old
`(east, north, score, out_png, title, top)` call; the new visuals switch on when a
DTM path and/or `zones` are supplied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from reindeer.terrain.grid import CELL_SIZE_M, load_field_polygons, LORDALEN, DALSIDA
from reindeer.terrain.terrain import _box_mean

# The presence score is a *magnitude* (0..1 chance), so it wears a single-hue
# sequential ramp (light -> dark green), not a red-green diverging scheme: red-green
# is the classic colourblind trap and, washed over a grey hillshade, reads as noise.
# Instead we leave low ground as bare terrain and let colour build only where the
# model favours the ground -- a "glow" that pulls the eye to where to GO.
CMAP = "Greens"
_DEFAULT_DTM = Path("data/raw/dem/dtm_50m_25833.tif")

# --- display tuning (image only; never touches the scores or the CSV) -----------
DISPLAY_SMOOTH_WIN = 5     # box-smooth the wash over ~5 cells (~1.25 km) so it reads
                           #   as coherent zones, not per-250 m-cell speckle
GLOW_LO = 0.40             # below this normalised score the ground stays bare terrain
GLOW_MAX_ALPHA = 0.85      # opacity of the strongest-favoured ground
GLOW_GAMMA = 1.3           # >1 concentrates the glow onto the very best ground


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


def _hillshade_array(z, res, az_deg=315.0, alt_deg=45.0):
    """Standard shaded relief from an elevation array (north-up), 0..1."""
    az = np.radians(360.0 - az_deg + 90.0)
    alt = np.radians(alt_deg)
    dy, dx = np.gradient(z, res, res)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    hs = (np.sin(alt) * np.sin(slope)
          + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(hs, 0.0, 1.0)


def _hillshade_background(extent, dtm_path=_DEFAULT_DTM, res=80.0):
    """Reproject the 50 m DTM to a hillshade over `extent` (EPSG:25832).

    Returns (hillshade_for_origin_lower, extent) or None if the DTM is unavailable.
    """
    dtm_path = Path(dtm_path)
    if not dtm_path.exists():
        return None
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
    except Exception:
        return None
    xmin, xmax, ymin, ymax = extent
    W = max(1, int(round((xmax - xmin) / res)))
    H = max(1, int(round((ymax - ymin) / res)))
    dst = np.full((H, W), np.nan, dtype="float32")
    dst_transform = from_origin(xmin, ymax, res, res)  # row 0 = north (top)
    try:
        with rasterio.open(dtm_path) as src:
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs="EPSG:25832",
                      src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
    except Exception:
        return None
    if not np.isfinite(dst).any():
        return None
    hs = _hillshade_array(np.nan_to_num(dst, nan=float(np.nanmin(dst))), res)
    return np.flipud(hs)  # flip so row 0 = south, for imshow origin="lower"


def cluster_top_zones(east, north, score, n=6, min_sep_m=2500.0):
    """Greedy, well-separated top zones: take the highest cell, suppress everything
    within min_sep_m, repeat. Turns a cloud of adjacent hot cells into distinct
    'go here' areas. Returns a list of (east, north, score) best-first."""
    e = np.asarray(east, float)
    n_ = np.asarray(north, float)
    s = np.asarray(score, float)
    order = np.argsort(-s)
    picked = []
    for i in order:
        if not np.isfinite(s[i]):
            continue
        if all((e[i] - pe) ** 2 + (n_[i] - pn) ** 2 >= min_sep_m ** 2
               for pe, pn, _ in picked):
            picked.append((float(e[i]), float(n_[i]), float(s[i])))
        if len(picked) >= n:
            break
    return picked


def _scale_bar(ax, extent, frac=0.22):
    """A rounded scale bar in the lower-left of the map axis."""
    xmin, xmax, ymin, ymax = extent
    span = xmax - xmin
    raw = span * frac
    # snap to a tidy 1/2/5 * 10^k metre length
    p = 10 ** np.floor(np.log10(raw))
    length = min([1, 2, 5, 10], key=lambda m: abs(m * p - raw)) * p
    x0 = xmin + span * 0.05
    y0 = ymin + (ymax - ymin) * 0.05
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=3, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + (ymax - ymin) * 0.012,
            f"{length/1000:g} km", ha="center", va="bottom", fontsize=8)


def _north_arrow(ax, extent):
    xmin, xmax, ymin, ymax = extent
    x = xmax - (xmax - xmin) * 0.06
    y0 = ymax - (ymax - ymin) * 0.16
    y1 = ymax - (ymax - ymin) * 0.06
    ax.annotate("N", xy=(x, y1), xytext=(x, y0), ha="center", va="center",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))


def render_heatmap(east, north, score, out_png: Path,
                   title: str | None = None, top=None, dpi: int = 140,
                   dtm_path=_DEFAULT_DTM, zones=None,
                   weather_text: str | None = None,
                   regime_text: str | None = None,
                   subtitle: str | None = None) -> Path:
    """Render score over the grid to out_png.

    Backward compatible: `top=(east_array, north_array)` still marks points.
    New (preferred): `zones=[(east, north, label), ...]` draws numbered pins and a
    ranked side-panel list; `weather_text`/`regime_text`/`subtitle` fill the panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colormaps, colors as mcolors
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter

    arr, extent = _rasterise(east, north, score)

    has_panel = bool(zones) or bool(weather_text) or bool(regime_text)
    fig = plt.figure(figsize=(14.0, 7.0) if has_panel else (11.0, 7.0),
                     layout="constrained")
    if has_panel:
        gs = GridSpec(1, 2, width_ratios=[3.25, 1.0], figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        panel = fig.add_subplot(gs[0, 1])
        panel.axis("off")
    else:
        ax = fig.add_subplot(1, 1, 1)
        panel = None

    # --- shaded-relief terrain background -------------------------------------
    hs = _hillshade_background(extent, dtm_path)
    if hs is not None:
        ax.imshow(hs, origin="lower", extent=extent, cmap="gray",
                  vmin=0.0, vmax=1.0, alpha=1.0, interpolation="bilinear", zorder=0)

    # --- probability wash: a green "glow" only on favoured ground -------------
    # Lightly smooth the surface for display so it reads as coherent zones, not a
    # per-cell mosaic (the underlying scores/CSV are untouched); colour it with a
    # single-hue green ramp and gate the opacity so low ground shows bare hillshade
    # and only genuinely favoured ground is tinted.
    valid = np.isfinite(arr)
    disp = _box_mean(np.nan_to_num(arr, nan=0.0), valid, DISPLAY_SMOOTH_WIN)
    disp = np.where(valid, disp, 0.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = colormaps[CMAP]
    rgba = cmap(norm(disp))
    glow = np.clip((disp - GLOW_LO) / (1.0 - GLOW_LO), 0.0, 1.0)
    a = GLOW_MAX_ALPHA * glow ** GLOW_GAMMA
    a[~valid] = 0.0                     # transparent outside the field
    rgba[..., 3] = a
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="bilinear",
              zorder=1)

    # --- field boundary -------------------------------------------------------
    try:
        polys = load_field_polygons()
        px, py = polys[LORDALEN].exterior.xy
        ax.plot(px, py, color="#111111", lw=1.8, alpha=0.85, zorder=3,
                label="Lordalen field")
        if DALSIDA in polys:
            dx, dy = polys[DALSIDA].exterior.xy
            ax.plot(dx, dy, color="#333333", lw=0.9, ls="--", alpha=0.5, zorder=3)
    except Exception:
        pass

    # --- numbered top zones ---------------------------------------------------
    zone_rows = []
    if zones:
        for i, z in enumerate(zones, 1):
            ze, zn = z[0], z[1]
            label = z[2] if len(z) > 2 else ""
            ax.scatter([ze], [zn], s=230, marker="o", facecolors="white",
                       edgecolors="black", linewidths=1.6, zorder=4)
            ax.text(ze, zn, str(i), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="black", zorder=5)
            zone_rows.append((i, label))
    elif top is not None:
        tx, ty = top
        ax.scatter(tx, ty, s=30, facecolors="none", edgecolors="black", lw=1.0,
                   zorder=4)

    _scale_bar(ax, extent)
    _north_arrow(ax, extent)

    ax.set_aspect("equal")
    # small margin so zone markers on the field edge aren't clipped by the frame
    padx = (extent[1] - extent[0]) * 0.02
    pady = (extent[3] - extent[2]) * 0.02
    ax.set_xlim(extent[0] - padx, extent[1] + padx)
    ax.set_ylim(extent[2] - pady, extent[3] + pady)
    # tick labels in km for readability
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
    ax.set_xlabel("Easting (km, UTM 32N / EPSG:25832)")
    ax.set_ylabel("Northing (km)")
    ax.tick_params(labelsize=8)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    if subtitle:
        ax.set_title(subtitle, fontsize=9, loc="right", color="#444444")

    # --- plain-language colour legend (horizontal, under the map) -------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.09)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["unlikely", "low", "moderate", "likely", "strong"])
    cb.set_label("chance reindeer favour this ground tomorrow "
                 "(bare terrain = unlikely)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # --- side panel -----------------------------------------------------------
    if panel is not None:
        import textwrap

        def _wrap(text, width):
            """Word-wrap each explicit line; return (text, n_lines)."""
            out = []
            for line in str(text).split("\n"):
                out.extend(textwrap.wrap(line, width) or [""])
            return "\n".join(out), len(out)

        y = 0.98
        panel.text(0.0, y, "Where to look", fontsize=13, fontweight="bold",
                   va="top", transform=panel.transAxes)
        y -= 0.06
        for txt, col in ((weather_text, "#1a1a1a"), (regime_text, "#8a4500")):
            if txt:
                wrapped, nlines = _wrap(txt, 32)
                panel.text(0.0, y, wrapped, fontsize=9.5, va="top", color=col,
                           linespacing=1.35, transform=panel.transAxes)
                y -= 0.030 * nlines + 0.022
        if zone_rows:
            y -= 0.015
            panel.text(0.0, y, "Ranked zones (best first):", fontsize=10,
                       fontweight="bold", va="top", transform=panel.transAxes)
            y -= 0.05
            # Fit-to-height WITHOUT overlap: advance y by the font's *natural* line
            # height (in axis fraction), and if the whole list won't fit above the
            # footer, shrink the font (down to a floor) instead of cramming lines.
            FOOTER_TOP = 0.10
            GAP_LINES = 0.5                     # blank space between entries, in lines
            wrapped_rows = [(num, *_wrap(label, 40)) for num, label in zone_rows]
            total_lines = sum(nl for _, _, nl in wrapped_rows)
            n_entries = len(wrapped_rows)
            panel_h_in = panel.get_position().height * fig.get_size_inches()[1]

            def line_frac(fs):                  # one text line as a fraction of panel height
                return (fs * 1.3 / 72.0) / max(panel_h_in, 1e-6)

            avail = max(0.05, y - FOOTER_TOP)
            zfs = 9.0
            while zfs > 6.5 and (total_lines + GAP_LINES * n_entries) * line_frac(zfs) > avail:
                zfs -= 0.5
            lf = line_frac(zfs)
            for num, wrapped, nlines in wrapped_rows:
                panel.text(0.0, y, f"{num}.", fontsize=zfs, fontweight="bold",
                           va="top", transform=panel.transAxes)
                panel.text(0.06, y, wrapped, fontsize=zfs, va="top",
                           linespacing=1.3, transform=panel.transAxes)
                y -= lf * nlines + GAP_LINES * lf
        panel.text(0.0, 0.02,
                   "Search-narrowing tool, not a GPS oracle: an honest probability\n"
                   "surface to complement glassing and fieldcraft, not replace them.",
                   fontsize=7.2, va="bottom", color="#666666", style="italic",
                   transform=panel.transAxes)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
