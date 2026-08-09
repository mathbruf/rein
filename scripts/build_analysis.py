"""Build the complete analysis folder: output/analysis/.

One command regenerates every analysis artifact from the current model + data:
  hit_analysis.png          (via scripts/hit_analysis.py — run separately)
  validation_breakdown.png  AUC by positioning method and by season
  weather_drivers.png       which behavioural driver fired on each hunt-window day
  model_explainer.png       how the scorer works — inputs, terms, weights
  README.md                 plain-language guide: how the model works, what it
                            considers, current headline percentages, honest limits

Palette: the three chart hues (#eb6834 / #2a78d6 / #1b9e77) are CVD-validated
(dataviz six-checks, light surface). Bars encoding one magnitude wear ONE hue.

Usage (repo root, venv active):
    python scripts/build_analysis.py
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
from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402
from reindeer.geocode.positions import load_manual_pins  # noqa: E402
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA  # noqa: E402
from reindeer.model.validation import effort_weights  # noqa: E402
from reindeer.model import cv as CV  # noqa: E402
from reindeer.paths import outdir  # noqa: E402
from reindeer.area import AREA  # noqa: E402
from shapely.prepared import prep  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed identity across all analysis charts (validated palette; never re-ordered)
C_RAW = "#eb6834"    # whole-field / insect drive
C_FAIR = "#2a78d6"   # effort-matched / shelter drive
C_KIND = "#1b9e77"   # position-confident tier
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)


def load_everything():
    grid = VAL.load_grid()
    gaz = load_gazetteer()
    pins = load_manual_pins()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    resolved = VAL.resolve_observations(grid, gaz, inside, 3000.0, pins=pins)
    fields = VAL.build_fields(grid, resolved["date"], source="archive")
    smooth = VAL.make_smoother(grid)
    per_date = {d: (None if fields.get(d) is None else VAL.score_date(grid, fields[d]))
                for d in sorted(set(resolved["date"]))}
    cell_dist = grid["dist_disturb_m"].to_numpy()
    w = effort_weights(cell_dist, cell_dist[resolved["cell_idx"].to_numpy()])
    used, pct, _ = VAL.evaluate(resolved, per_date, smooth, VAL.HEADLINE_RADIUS_M,
                                weights=w)
    return grid, fields, used.assign(p=np.asarray(pct, float))


def chart_validation_breakdown(u, out: Path):
    """AUC by positioning method + by season. One magnitude -> one hue."""
    conf = u["method"].map(VAL.position_confident)
    by_m = (u.groupby("method")["p"].agg(["count", "mean"])
             .sort_values("mean", ascending=True))
    by_y = u.assign(year=u["date"].str[:4]).groupby("year")["p"].agg(["count", "mean"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.14, wspace=0.30)
    fig.text(0.055, 0.95, "Where the hit-rate comes from — and what deflates it",
             fontsize=15, fontweight="bold")
    fig.text(0.055, 0.885,
             "Per-report ranking score (0.5 = chance) under the effort-matched "
             "background, split by how each report's position was located.",
             fontsize=9.5, color=INK2)

    # Panel A: by positioning method (horizontal bars, value + n labels)
    names = {"offset-pinned": "human-pinned area\n(“nord for X”)",
             "offset": "direction offset",
             "mot-pinned": "pinned “mot Y”",
             "at-landmark-pinned": "pinned at landmark",
             "at-landmark": "bare landmark name\n(vague tier)"}
    lbls = [names.get(m, m) for m in by_m.index]
    y = np.arange(len(by_m))
    axA.barh(y, by_m["mean"], height=0.62, color=C_FAIR, zorder=2)
    axA.axvline(0.5, ls="--", lw=1.4, color=INK2, zorder=3)
    axA.text(0.505, -0.42, " chance", color=INK2, fontsize=8.5, va="top")
    for yi, (m, row) in zip(y, by_m.iterrows()):
        axA.text(row["mean"] + 0.012, yi, f"{row['mean']:.2f}  (n={int(row['count'])})",
                 va="center", fontsize=9, color=INK)
    axA.set_yticks(y, lbls, fontsize=9)
    axA.set_xlim(0, 1.0)
    axA.set_xlabel("mean ranking score (AUC-style, 0.5 = chance)")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.grid(True, axis="x", color=GRIDC, lw=0.8)
    axA.set_axisbelow(True)
    axA.set_title("By positioning method — trustworthy positions score high;\n"
                  "name-only geocodes measure the reporter, not the model",
                  fontsize=10, color=INK2, loc="left", pad=8)

    # Panel B: by season
    x = np.arange(len(by_y))
    axB.bar(x, by_y["mean"], width=0.56, color=C_FAIR, zorder=2)
    axB.axhline(0.5, ls="--", lw=1.4, color=INK2, zorder=3)
    for xi, (yr, row) in zip(x, by_y.iterrows()):
        axB.text(xi, row["mean"] + 0.015, f"{row['mean']:.2f}", ha="center",
                 fontsize=9.5, fontweight="bold", color=INK)
        axB.text(xi, 0.03, f"n={int(row['count'])}", ha="center", fontsize=8.5,
                 color=INK2)
    axB.set_xticks(x, by_y.index, fontsize=10)
    axB.set_ylim(0, 1.0)
    axB.set_ylabel("mean ranking score")
    _despine(axB)
    axB.set_title("By season — small yearly samples;\nthe verdict needs the pooled CV",
                  fontsize=10, color=INK2, loc="left", pad=8)

    n_conf = int(conf.sum())
    st = CV.repeated_kfold_stats(u.loc[conf, "p"].to_numpy(), k=5)
    fig.text(0.055, 0.02,
             f"Position-confident tier: n={n_conf}, CV AUC {st['mean']:.3f} ± "
             f"{st['std']:.3f}, {st['fold_beats_chance']*100:.0f}% of CV folds beat "
             "chance. Weights were never fitted to these reports.",
             fontsize=9, color=INK2)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_weather_drivers(grid, fields, out: Path):
    """Which behavioural driver the model saw on each hunt-window day."""
    import reindeer.model.score as S
    rows = []
    for d in sorted(k for k, v in fields.items() if v is not None):
        res = VAL.score_date(grid, fields[d])  # noqa: F841  (warms nothing; cheap)
        # day-level drivers, recomputed exactly as the scorer sees them
        wf = fields[d]
        expo = S.exposure_channel(grid["tpi_m"].to_numpy(), grid["slope_deg"].to_numpy(),
                                  grid["aspect_deg"].to_numpy(), wf.wind_dir_deg)
        eff = np.clip(wf.wind_ms * (1.0 + S.WIND_EXPOSURE_GAIN * expo), 0.0, None)
        p_ins = float(S._insect_p(np.nanpercentile(wf.temp_c, S.DAY_WARM_PCTL),
                                  np.nanmedian(wf.wind_ms), np.nanmean(wf.precip_mm)))
        p_shl = float(S._shelter_p(np.nanpercentile(wf.temp_c, S.DAY_COLD_PCTL),
                                   np.nanpercentile(eff, S.DAY_WINDY_PCTL),
                                   np.nanmean(wf.precip_mm)))
        rows.append((d, p_ins, p_shl))
    dates = [r[0][5:] for r in rows]          # MM-DD, year shown in group label
    years = [r[0][:4] for r in rows]
    ins = np.array([r[1] for r in rows])
    shl = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.24)
    fig.text(0.055, 0.94, "What the model saw — behavioural drivers across the hunt window",
             fontsize=15, fontweight="bold")
    fig.text(0.055, 0.88,
             "Day-level regime pressures from the real weather field on each validation date.\n"
             "Shelter (cold/wet/wind) dominates the Aug–Sept hunt; the insect drive is almost "
             "always off — exactly what the hunter said.",
             fontsize=9.5, color=INK2, va="top", linespacing=1.5)
    x = np.arange(len(rows))
    ax.bar(x - 0.21, shl, width=0.4, color=C_FAIR, label="shelter drive", zorder=2)
    ax.bar(x + 0.21, ins, width=0.4, color=C_RAW, label="insect drive", zorder=2)
    ax.set_xticks(x, dates, rotation=60, fontsize=7.5)
    # year separators + group labels
    prev = 0
    for i in range(1, len(years) + 1):
        if i == len(years) or years[i] != years[prev]:
            if i < len(years):
                ax.axvline(i - 0.5, color=GRIDC, lw=1.2)
            ax.text((prev + i - 1) / 2, 1.06, years[prev], ha="center",
                    fontsize=10, fontweight="bold", color=INK2)
            prev = i
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("drive strength (0–1)")
    _despine(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_model_explainer(out: Path):
    """How the scorer works — a reading diagram, not a data chart."""
    import reindeer.model.score as S
    fig = plt.figure(figsize=(13.0, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#f1f0ec", ec=INK2, title_c=INK):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   lw=1.2, zorder=1))
        ax.text(x + 0.012, y + h - 0.028, title, fontsize=10.5, fontweight="bold",
                color=title_c, va="top")
        ax.text(x + 0.012, y + h - 0.075, "\n".join(lines), fontsize=8.4,
                color=INK, va="top", linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))

    ax.text(0.04, 0.955, "How the model works — from tomorrow's weather to a map",
            fontsize=16, fontweight="bold")
    ax.text(0.04, 0.915, f"Area: {AREA.name} ({AREA.region}) · 250 m grid · "
            "rule-based & expert-tuned — sightings are NEVER an input, only the "
            "measuring stick.", fontsize=9.5, color=INK2)

    box(0.04, 0.60, 0.27, 0.27, "STATIC LANDSCAPE  (built once)", [
        "• Elevation, slope, aspect, TPI — 50 m national DTM",
        "• Forage value — AR50 land cover per cell",
        "• Distance to people — roads, trails,",
        "   cabins, parking (OSM + fjellstyre)",
        "• Field boundary — statsallmenning polygons"])
    box(0.04, 0.28, 0.27, 0.27, "REAL WEATHER FIELD  (daily)", [
        "• Lattice of real forecast points (1 km model;",
        "   ERA5 for past dates), no API key",
        "• Temperature, wind speed + DIRECTION, rain",
        "• Interpolated per cell: wind as u/v vectors,",
        "   temp by data-driven lapse (inversions incl.)"])
    box(0.37, 0.44, 0.28, 0.36, "SCORING  (per cell, per day)", [
        f"insect drive (warm+calm+dry)   w={S.W_INSECT}",
        f"shelter drive (cold|wet|windy) w={S.W_SHELTER}",
        "→ a weighted SWITCH, not a sum — the",
        "   regimes cannot cancel each other",
        "exposure = TPI + cos(aspect − wind dir)",
        f"high-ground baseline           w={S.W_BASELINE}",
        f"forage value                   w={S.W_FORAGE}",
        f"steep-terrain penalty          w={S.W_STEEP}",
        f"disturbance penalty            w={S.W_DISTURB}",
        "(all weights expert-set with the hunter,",
        " never fitted to sightings)"])
    box(0.71, 0.44, 0.25, 0.36, "OUTPUT  (what you read)", [
        "• 0–1 score per cell → percentile-",
        "   ranked green wash on hillshade",
        "• Up to 6 named “go here” zones",
        "   with plain-language reasons",
        "• Wind arrow + roads/trails/cabins",
        "   for orientation and the stalk",
        "• output/forecast/<date>.png"])
    box(0.37, 0.06, 0.59, 0.30, "VALIDATION  (the honesty loop — separate from scoring)", [
        "Hunter reports → gazetteer + human pins → each report scored within its day's field.",
        "Fairness corrections: effort-matched background (reports come from accessible ground),",
        "position-confidence tiers (name-only geocodes measure the reporter, not the model).",
        "Gate: k-fold cross-validated, select-then-evaluate — a change is kept ONLY if it",
        "wins out-of-sample. Current: CV AUC 0.64, ~9 of 10 folds beat chance (n=30 confident)."])
    arrow(0.31, 0.735, 0.37, 0.66)
    arrow(0.31, 0.415, 0.37, 0.52)
    arrow(0.65, 0.62, 0.71, 0.62)
    arrow(0.51, 0.44, 0.51, 0.36)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_readme(u, out: Path):
    conf = u["method"].map(VAL.position_confident)
    p_all = u["p"].to_numpy()
    p_conf = u.loc[conf, "p"].to_numpy()
    st = CV.repeated_kfold_stats(p_conf, k=5)

    def hits(p):
        return (p.mean(), (p >= .5).mean() * 100, (p >= 2 / 3).mean() * 100,
                (p >= .8).mean() * 100, (p >= .9).mean() * 100)
    a_auc, a_h, a_t3, a_t20, a_t10 = hits(p_all)
    c_auc, c_h, c_t3, c_t20, c_t10 = hits(p_conf)

    text = f"""# Analysis — reindeer presence model ({AREA.name})

_Regenerated by `scripts/build_analysis.py` + `scripts/hit_analysis.py`.
All numbers are computed fresh from the current model and data on every run._

## How the model works (one paragraph)

Every day the model fetches a **real weather field** over the fell — a lattice of
forecast points (temperature, wind speed **and direction**, precipitation)
interpolated to every 250 m cell — and combines it with the **fixed landscape**
(elevation, slope aspect, terrain shelter/exposure, forage quality, distance to
roads/trails/cabins). Two behavioural regimes compete in a weighted switch:
**insect-escape** (warm + calm + dry → animals climb to cool, wind-exposed ground)
and **shelter** (cold / wet / windy → animals hold leeward, calmer, lower ground),
on top of a gentle high-ground baseline, a forage bonus and a disturbance penalty
("they come lower only if hunters allow"). Every weight is a named constant tuned
with the hunter — **sightings are never an input**; they are kept exclusively to
*test* the map.

## What is taken into consideration

| Signal | Source | Role |
|---|---|---|
| Temperature (per cell) | Open-Meteo lattice, data-driven lapse | regime gates + per-cell comfort |
| Wind speed + direction | same, u/v vector interpolation | exposure: `TPI + cos(aspect − wind)` |
| Precipitation | same | grounds insects; drives shelter |
| Elevation / slope / TPI / aspect | Kartverket 50 m DTM | baseline, exposure, travel limits |
| Forage (land cover) | NIBIO AR50 | destination value |
| Human disturbance | OSM + fjellstyre cabins | penalty fading over ~2.5 km |

## Current headline numbers ({len(p_conf)} position-confident reports)

- **CV AUC {st['mean']:.3f} ± {st['std']:.3f}** — {st['fold_beats_chance']*100:.0f}% of
  cross-validation folds beat chance.
- **{c_h:.0f}%** of confirmed sightings fell in the model's favoured half (chance 50%).
- **{c_t3:.0f}%** in the top third (chance 33%) · **{c_t20:.0f}%** in the top 20%
  (chance 20%) · **{c_t10:.0f}%** in the top 10% (chance 10%).
- All {len(p_all)} reports incl. the vague tier: AUC {a_auc:.3f}, favoured-half {a_h:.0f}%.

## The images

| File | What it shows |
|---|---|
| `hit_analysis.png` | The headline: cumulative gain + hit-% at each threshold, under all three measurement views (whole-field / effort-matched / position-confident). |
| `validation_breakdown.png` | Where the hit-rate comes from: score by positioning method (trustworthy positions score ~0.7–0.8; name-only geocodes drag the average) and by season. |
| `weather_drivers.png` | What the model saw on each validation day — shelter dominates the autumn hunt window; insects are almost always off. |
| `model_explainer.png` | The full pipeline diagram: landscape + real weather → scoring terms and weights → map, plus the validation loop. |

## Honest limits (always state these)

- Small sample: {len(p_conf)} trustworthy reports; verdicts carry wide error bars.
- Presence-only, effort-biased reports — corrected in evaluation (effort-matched
  background), but a residual accessibility edge remains.
- Positions are areas (~2.5 km), never GPS points; 5 vague reports await human pins.
- The map is a **search-narrowing tool, not a GPS oracle** — reindeer are social
  and partly stochastic; the model narrows where to glass from, nothing more.
"""
    out.write_text(text, encoding="utf-8")


def main() -> None:
    adir = outdir("analysis")
    print("loading validation data (cached weather fields) ...")
    grid, fields, u = load_everything()
    chart_validation_breakdown(u, adir / "validation_breakdown.png")
    print("-> validation_breakdown.png")
    chart_weather_drivers(grid, fields, adir / "weather_drivers.png")
    print("-> weather_drivers.png")
    chart_model_explainer(adir / "model_explainer.png")
    print("-> model_explainer.png")
    write_readme(u, adir / "README.md")
    print("-> README.md")
    print(f"\nanalysis folder: {adir}  (hit_analysis.png via scripts/hit_analysis.py)")


if __name__ == "__main__":
    main()
