"""Phase 6: percentage hit-rate analysis of the shipped scorer, as a chart.

Computes, for the held-out sightings, the per-report date-matched percentile of the
observed location within that day's field, under BOTH backgrounds:
  - whole-field (every cell equally "available") — the effort-confounded view;
  - effort-matched (cells weighted to where reports actually occur, IDEA 009) — fair.

Renders two panels:
  A. cumulative gain curve — share of sightings captured vs the top fraction of
     ranked ground (chance = diagonal); area under = the AUC.
  B. hit-% at the headline thresholds (favoured half / top third / top 20% / top
     10%) vs chance.

Usage (repo root, venv active):
    python scripts/hit_analysis.py [out.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import validate as VAL  # noqa: E402
from cv_validate import per_config_percentiles  # noqa: E402
from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402
from reindeer.geocode.positions import load_manual_pins  # noqa: E402
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA  # noqa: E402
from reindeer.model.validation import effort_weights  # noqa: E402
from reindeer.weather.historical import fetch_archive, weather_by_date  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.prepared import prep  # noqa: E402

_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)

# validated colourblind-safe palette (dataviz skill): blue vs orange, neutral chance.
C_FAIR = "#2a78d6"      # effort-matched (fair) — headline
C_RAW = "#eb6834"       # whole-field (effort-confounded)
INK = "#0b0b0b"
INK2 = "#52514e"
GRIDC = "#e6e5e1"
SURF = "#fcfcfb"


def gain_curve(pct, qs):
    """Share of reports whose observed cell is in the top-q of its date's field."""
    return np.array([(pct >= 1.0 - q).mean() for q in qs])


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "data" / "processed" / "maps" / "hit_analysis.png"

    grid = VAL.load_grid()
    gaz = load_gazetteer()
    pins = load_manual_pins()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    resolved = VAL.resolve_observations(grid, gaz, inside, 3000.0, pins=pins)
    cx, cy = grid["east"].mean(), grid["north"].mean()
    lon, lat = _to_wgs.transform(cx, cy)
    wx = weather_by_date(fetch_archive(lat, lon, min(resolved["date"]), max(resolved["date"])))
    smooth = VAL.make_smoother(grid)
    radius = VAL.HEADLINE_RADIUS_M

    cell_dist = grid["dist_disturb_m"].to_numpy()
    obs_dist = cell_dist[resolved["cell_idx"].to_numpy()]
    weights = effort_weights(cell_dist, obs_dist)

    _, p_raw = per_config_percentiles({}, grid, resolved, wx, smooth, radius)
    _, p_fair = per_config_percentiles({}, grid, resolved, wx, smooth, radius, weights=weights)
    n = len(p_raw)

    # --- headline thresholds (share of reports in the model's top X) --------------
    thr = [("Favoured\nhalf", 0.50), ("Top\nthird", 1 / 3),
           ("Top\n20%", 0.20), ("Top\n10%", 0.10)]
    labels = [t[0] for t in thr]
    chance = np.array([t[1] for t in thr])
    hit_raw = np.array([(p_raw >= 1 - q).mean() for _, q in thr])
    hit_fair = np.array([(p_fair >= 1 - q).mean() for _, q in thr])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(13.5, 6.2), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    axA = fig.add_subplot(gs[0, 0]); axA.set_facecolor(SURF)
    axB = fig.add_subplot(gs[0, 1]); axB.set_facecolor(SURF)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.20, wspace=0.24)

    fig.text(0.065, 0.955, "Reindeer scorer — held-out hit-rate analysis",
             fontsize=17, fontweight="bold", ha="left", color=INK)
    fig.text(0.065, 0.905,
             f"{n} held-out sightings · date-matched · 2.5 km zone.  “Effort-matched” "
             "reweights the background to where\nreports actually occur, correcting the "
             "reporting bias toward accessible ground (IDEA 009).",
             fontsize=10, ha="left", va="top", color=INK2, linespacing=1.4)
    fig.text(0.065, 0.085,
             "Read: whole-field hugs the chance diagonal (the effort confound masks the "
             "signal); the effort-matched curve stands clearly\nabove chance — the scorer "
             "does narrow the search once reporting bias is removed. Small sample (n=37).",
             fontsize=9, ha="left", va="top", color=INK2, linespacing=1.4)

    # --- Panel A: cumulative gain curve ------------------------------------------
    qs = np.linspace(0, 1, 101)
    axA.plot([0, 1], [0, 1], ls="--", lw=1.6, color=INK2, zorder=1)
    axA.text(0.62, 0.55, "chance", rotation=34, color=INK2, fontsize=10,
             ha="center", va="bottom")
    axA.plot(qs, gain_curve(p_fair, qs), lw=2.8, color=C_FAIR, zorder=3,
             solid_capstyle="round", label=f"effort-matched (AUC {p_fair.mean():.2f})")
    axA.plot(qs, gain_curve(p_raw, qs), lw=2.6, color=C_RAW, zorder=2,
             solid_capstyle="round", label=f"whole-field (AUC {p_raw.mean():.2f})")
    axA.legend(loc="upper left", frameon=False, fontsize=10.5,
               handlelength=1.4, borderaxespad=0.6)
    axA.set_xlim(0, 1); axA.set_ylim(0, 1.02)
    axA.set_xlabel("top fraction of ranked ground searched")
    axA.set_ylabel("share of sightings found")
    axA.xaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    axA.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    axA.grid(True, color=GRIDC, lw=0.8)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.set_title("Cumulative gain — the higher above the diagonal, the better",
                  fontsize=11, color=INK2, loc="left", pad=8)

    # --- Panel B: hit-% at headline thresholds -----------------------------------
    x = np.arange(len(labels)); w = 0.38
    axB.bar(x - w / 2, hit_raw * 100, w, color=C_RAW, label="whole-field (confounded)",
            zorder=2)
    axB.bar(x + w / 2, hit_fair * 100, w, color=C_FAIR, label="effort-matched (fair)",
            zorder=2)
    # chance reference per group
    for xi, ch in zip(x, chance):
        axB.plot([xi - w, xi + w], [ch * 100, ch * 100], ls="--", lw=1.6,
                 color=INK2, zorder=3)
    axB.text(len(labels) - 0.5, chance[-1] * 100 + 1.5, "chance", color=INK2,
             fontsize=9.5, ha="right", va="bottom")
    for xi, v in zip(x - w / 2, hit_raw * 100):
        axB.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", color=C_RAW,
                 fontsize=9.5, fontweight="bold")
    for xi, v in zip(x + w / 2, hit_fair * 100):
        axB.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", color=C_FAIR,
                 fontsize=9.5, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylabel("% of sightings in that band")
    axB.set_ylim(0, max((hit_fair * 100).max(), (hit_raw * 100).max(), 60) + 10)
    axB.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axB.grid(True, axis="y", color=GRIDC, lw=0.8)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.legend(frameon=False, fontsize=9.5, loc="upper right")
    axB.set_title("Share of sightings the model puts on its favoured ground",
                  fontsize=11, color=INK2, loc="left", pad=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"whole-field  AUC {p_raw.mean():.3f} | half {(p_raw>=0.5).mean()*100:.0f}% "
          f"top20 {(p_raw>=0.8).mean()*100:.0f}% top10 {(p_raw>=0.9).mean()*100:.0f}%")
    print(f"effort-match AUC {p_fair.mean():.3f} | half {(p_fair>=0.5).mean()*100:.0f}% "
          f"top20 {(p_fair>=0.8).mean()*100:.0f}% top10 {(p_fair>=0.9).mean()*100:.0f}%")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
