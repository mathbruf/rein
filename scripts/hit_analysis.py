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
from reindeer.paths import outdir  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.prepared import prep  # noqa: E402

_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)

# validated colourblind-safe palette (dataviz skill): blue vs orange vs teal, neutral chance.
from _chartstyle import (C_RAW, C_FAIR, C_KIND, INK, INK2, GRIDC, SURF,  # noqa: E402
                         RC, brand_footer)
from reindeer.model import cv as CV  # noqa: E402


def gain_curve(pct, qs):
    """Share of reports whose observed cell is in the top-q of its date's field."""
    return np.array([(pct >= 1.0 - q).mean() for q in qs])


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else outdir("analysis") / "hit_analysis.png"

    grid = VAL.load_grid()
    gaz = load_gazetteer()
    pins = load_manual_pins()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    resolved = VAL.resolve_observations(grid, gaz, inside, 3000.0, pins=pins)
    print(f"building real per-cell weather fields for {resolved['date'].nunique()} dates ...")
    fields = VAL.build_fields(grid, resolved["date"], source="archive")
    smooth = VAL.make_smoother(grid)
    radius = VAL.HEADLINE_RADIUS_M

    cell_dist = grid["dist_disturb_m"].to_numpy()
    obs_dist = cell_dist[resolved["cell_idx"].to_numpy()]
    weights = effort_weights(cell_dist, obs_dist)

    used0, p_raw = per_config_percentiles({}, grid, resolved, fields, smooth, radius)
    _, p_fair = per_config_percentiles({}, grid, resolved, fields, smooth, radius, weights=weights)
    n = len(p_raw)
    # position-confident tier (reporter-error correction): drop the name-only
    # at-landmark placements that locate the feature, not the herd.
    conf = used0["method"].map(VAL.position_confident).to_numpy()
    p_kind = p_fair[conf]

    # --- headline thresholds (share of reports in the model's top X) --------------
    thr = [("Favoured\nhalf", 0.50), ("Top\nthird", 1 / 3),
           ("Top\n20%", 0.20), ("Top\n10%", 0.10)]
    labels = [t[0] for t in thr]
    chance = np.array([t[1] for t in thr])
    hit_raw = np.array([(p_raw >= 1 - q).mean() for _, q in thr])
    hit_fair = np.array([(p_fair >= 1 - q).mean() for _, q in thr])
    hit_kind = np.array([(p_kind >= 1 - q).mean() for _, q in thr])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC)
    st = CV.repeated_kfold_stats(p_kind, k=5)
    fig = plt.figure(figsize=(13.5, 6.6), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    axA = fig.add_subplot(gs[0, 0]); axA.set_facecolor(SURF)
    axB = fig.add_subplot(gs[0, 1]); axB.set_facecolor(SURF)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.775, bottom=0.235, wspace=0.24)

    fig.text(0.065, 0.958, "Reindeer scorer — held-out hit-rate analysis",
             fontsize=17, fontweight="bold", ha="left", color=INK)
    fig.text(0.065, 0.912,
             f"{n} held-out field reports · each scored inside its own day's real weather "
             "field · 2.5 km zone.  “Effort-matched” reweights the background to where "
             "reports actually\noccur; “confident positions” further drops name-only "
             "landmark placements that locate the feature, not the herd. Model weights "
             "were never fitted to these reports.",
             fontsize=10, ha="left", va="top", color=INK2, linespacing=1.4)
    fig.text(0.065, 0.125,
             "Read: whole-field hugs the chance diagonal (effort confound); correcting for "
             "reporting bias and for reporter position error lifts the\ncurve well above "
             f"chance — the scorer does narrow the search where the reports can be trusted. "
             f"Small sample (n={n}; {int(conf.sum())} confident).\nCross-validated gate: "
             f"CV AUC {st['mean']:.3f} ± {st['std']:.3f}, "
             f"{st['fold_beats_chance']*100:.0f}% of folds beat chance.",
             fontsize=9, ha="left", va="top", color=INK2, linespacing=1.4)
    brand_footer(fig)

    # --- Panel A: cumulative gain curve ------------------------------------------
    qs = np.linspace(0, 1, 101)
    axA.plot([0, 1], [0, 1], ls="--", lw=1.6, color=INK2, zorder=1)
    axA.text(0.62, 0.55, "chance", rotation=34, color=INK2, fontsize=10,
             ha="center", va="bottom")
    axA.plot(qs, gain_curve(p_kind, qs), lw=3.0, color=C_KIND, zorder=4,
             solid_capstyle="round",
             label=f"effort-matched + confident positions (AUC {p_kind.mean():.2f})")
    axA.plot(qs, gain_curve(p_fair, qs), lw=2.8, color=C_FAIR, zorder=3,
             solid_capstyle="round", label=f"effort-matched, all reports (AUC {p_fair.mean():.2f})")
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
    x = np.arange(len(labels)); w = 0.26
    axB.bar(x - w, hit_raw * 100, w, color=C_RAW, label="whole-field (confounded)",
            zorder=2)
    axB.bar(x, hit_fair * 100, w, color=C_FAIR, label="effort-matched, all reports",
            zorder=2)
    axB.bar(x + w, hit_kind * 100, w, color=C_KIND,
            label="effort-matched + confident positions", zorder=2)
    # chance reference per group
    for xi, ch in zip(x, chance):
        axB.plot([xi - 1.5 * w, xi + 1.5 * w], [ch * 100, ch * 100], ls="--", lw=1.6,
                 color=INK2, zorder=3)
    axB.text(len(labels) - 0.5, chance[-1] * 100 + 1.5, "chance", color=INK2,
             fontsize=9.5, ha="right", va="bottom")
    for xs, vals, col in ((x - w, hit_raw, C_RAW), (x, hit_fair, C_FAIR),
                          (x + w, hit_kind, C_KIND)):
        for xi, v in zip(xs, vals * 100):
            axB.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", color=col,
                     fontsize=9, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylabel("% of sightings in that band")
    axB.set_ylim(0, max((hit_kind * 100).max(), (hit_fair * 100).max(),
                        (hit_raw * 100).max(), 60) + 10)
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
    print(f"kind (conf.) AUC {p_kind.mean():.3f} | half {(p_kind>=0.5).mean()*100:.0f}% "
          f"top20 {(p_kind>=0.8).mean()*100:.0f}% top10 {(p_kind>=0.9).mean()*100:.0f}% "
          f"(n={len(p_kind)})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
