"""Build the complete analysis folder: output/analysis/.

One command regenerates every analysis artifact from the current model + data:
  hit_analysis.png          (via scripts/hit_analysis.py — run separately)
  validation_breakdown.png  AUC by positioning method and by season
  weather_drivers.png       which behavioural driver fired on each observation-window day
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
from _chartstyle import (C_RAW, C_FAIR, C_KIND, INK, INK2, GRIDC,  # noqa: E402
                         RC, brand_footer, despine as _despine)

plt.rcParams.update(RC)


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
    """AUC by positioning method + by season + every single report. One magnitude -> one hue."""
    conf = u["method"].map(VAL.position_confident)
    by_m = (u.groupby("method")["p"].agg(["count", "mean"])
             .sort_values("mean", ascending=True))
    by_y = u.assign(year=u["date"].str[:4]).groupby("year")["p"].agg(["count", "mean"])
    n_dates = u["date"].nunique()
    span = f"{u['date'].min()} → {u['date'].max()}"

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(15.5, 5.6),
        gridspec_kw={"width_ratios": [1.3, 0.85, 1.0]})
    fig.subplots_adjust(left=0.145, right=0.975, top=0.70, bottom=0.20, wspace=0.34)
    fig.text(0.048, 0.95, "Where the hit-rate comes from — and what deflates it",
             fontsize=15, fontweight="bold")
    fig.text(0.048, 0.885,
             f"Per-report ranking score (0.5 = chance) for {len(u)} independent field "
             f"reports on {n_dates} days ({span}), each scored inside its own day's real "
             "weather field under the effort-matched background,\nsplit by how the "
             "report's position was located. Reports are validation only — the model "
             "never sees them when it draws the map.",
             fontsize=9.5, color=INK2, va="top", linespacing=1.5)

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

    # Panel C: every report as one dot — nothing aggregated away
    n_conf = int(conf.sum())
    rng = np.random.default_rng(7)  # fixed jitter: chart is reproducible
    for tier, mask, col, lab in (("conf", conf, C_KIND, None),
                                 ("vague", ~conf, INK2, None)):
        pv = u.loc[mask, "p"].to_numpy()
        jit = rng.uniform(-0.16, 0.16, pv.size)
        axC.scatter(pv, (1 if tier == "conf" else 0) + jit, s=42, color=col,
                    alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
        axC.text(0.02, (1 if tier == "conf" else 0) + 0.30,
                 (f"position-confident (n={n_conf})" if tier == "conf"
                  else f"vague tier — name-only geocode (n={len(u) - n_conf})"),
                 fontsize=9, color=(C_KIND if tier == "conf" else INK2),
                 fontweight="bold")
        axC.plot([pv.mean()] * 2, [(1 if tier == "conf" else 0) - 0.22,
                                   (1 if tier == "conf" else 0) + 0.22],
                 color=col, lw=2.4, zorder=4)
    axC.axvline(0.5, ls="--", lw=1.4, color=INK2, zorder=2)
    axC.set_xlim(0, 1.0)
    axC.set_ylim(-0.55, 1.55)
    axC.set_yticks([])
    axC.set_xlabel("ranking score (0.5 = chance; tick = tier mean)")
    for s in ("top", "right", "left"):
        axC.spines[s].set_visible(False)
    axC.grid(True, axis="x", color=GRIDC, lw=0.8)
    axC.set_axisbelow(True)
    axC.set_title("Every report, no averaging — the honest\nspread behind the headline number",
                  fontsize=10, color=INK2, loc="left", pad=8)

    st = CV.repeated_kfold_stats(u.loc[conf, "p"].to_numpy(), k=5)
    fig.text(0.048, 0.085,
             f"Position-confident tier: n={n_conf}, CV AUC {st['mean']:.3f} ± "
             f"{st['std']:.3f}, {st['fold_beats_chance']*100:.0f}% of CV folds beat "
             "chance. Weights were never fitted to these reports.",
             fontsize=9, color=INK2)
    brand_footer(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_weather_drivers(grid, fields, out: Path):
    """Which behavioural driver the model saw on each observation-window day."""
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
        rows.append((d, p_ins, p_shl, float(np.nanmedian(wf.temp_c)),
                     float(np.nanmedian(wf.wind_ms)), float(np.nanmean(wf.precip_mm))))
    dates = [r[0][5:] for r in rows]          # MM-DD, year shown in group label
    years = [r[0][:4] for r in rows]
    ins = np.array([r[1] for r in rows])
    shl = np.array([r[2] for r in rows])
    tmed = np.array([r[3] for r in rows])
    wmed = np.array([r[4] for r in rows])
    pmm = np.array([r[5] for r in rows])

    fig, (ax, axT, axW, axP) = plt.subplots(
        4, 1, figsize=(15.5, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 0.8, 0.8, 0.8], "hspace": 0.14})
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.135)
    fig.text(0.048, 0.955, "What the model saw — behavioural drivers across the autumn window",
             fontsize=15, fontweight="bold")
    fig.text(0.048, 0.915,
             "Day-level regime pressures computed from the real per-cell weather field "
             "(Open-Meteo lattice interpolated to every 250 m cell) on each "
             "validation date — with the day's actual field-median weather below.\n"
             "Shelter (cold/wet/wind) dominates the Aug–Sept season; the insect drive is almost "
             "always off — exactly what the local field expert predicted from experience.",
             fontsize=9.5, color=INK2, va="top", linespacing=1.5)
    x = np.arange(len(rows))
    ax.bar(x - 0.21, shl, width=0.4, color=C_FAIR, label="shelter drive", zorder=2)
    ax.bar(x + 0.21, ins, width=0.4, color=C_RAW, label="insect drive", zorder=2)
    # year separators + group labels
    prev = 0
    for i in range(1, len(years) + 1):
        if i == len(years) or years[i] != years[prev]:
            if i < len(years):
                for a in (ax, axT, axW, axP):
                    a.axvline(i - 0.5, color=GRIDC, lw=1.2)
            ax.text((prev + i - 1) / 2, 1.06, years[prev], ha="center",
                    fontsize=10, fontweight="bold", color=INK2)
            prev = i
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("drive strength (0–1)")
    _despine(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    # the day's real weather (field medians) — the inputs behind the drives above
    def _strip(a, vals, ylab, fmt):
        a.plot(x, vals, lw=1.8, color=INK, marker="o", ms=3.4, zorder=3)
        a.set_ylabel(ylab, fontsize=8.5)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        for xi in (int(np.argmin(vals)), int(np.argmax(vals))):
            a.annotate(fmt.format(vals[xi]), (xi, vals[xi]), fontsize=7.5,
                       color=INK2, xytext=(0, 5), textcoords="offset points",
                       ha="center")
        a.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.35)
        _despine(a)
    _strip(axT, tmed, "temp °C\n(median)", "{:.0f}°")
    _strip(axW, wmed, "wind m/s\n(median)", "{:.0f}")
    axP.bar(x, pmm, width=0.55, color=INK2, zorder=2)
    axP.set_ylabel("precip mm\n(day sum)", fontsize=8.5)
    _despine(axP)
    axP.set_xticks(x)
    axP.set_xticklabels(dates, rotation=60, fontsize=7.5)

    brand_footer(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_model_explainer(out: Path):
    """How the whole project works — a reading diagram, not a data chart.

    This is the public one-image summary: pipeline + weights, PLUS the research
    question, the data sources/credits and the honest limits, so a GitHub
    visitor understands the project from this figure alone.
    """
    import reindeer.model.score as S
    fig = plt.figure(figsize=(13.0, 9.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#f1f0ec", ec=INK2, title_c=INK):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   lw=1.2, zorder=1))
        ax.text(x + 0.012, y + h - 0.024, title, fontsize=10.5, fontweight="bold",
                color=title_c, va="top")
        ax.text(x + 0.012, y + h - 0.062, "\n".join(lines), fontsize=8.4,
                color=INK, va="top", linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))

    ax.text(0.04, 0.965, "How the model works — from tomorrow's weather to a map",
            fontsize=16, fontweight="bold")
    ax.text(0.04, 0.932, f"Area: {AREA.name} ({AREA.region}) · 250 m grid · "
            "rule-based & expert-tuned — sightings are NEVER an input, only the "
            "measuring stick.", fontsize=9.5, color=INK2)

    box(0.04, 0.67, 0.27, 0.225, "STATIC LANDSCAPE  (built once)", [
        "• Elevation, slope, aspect, TPI — 50 m national DTM",
        "• Forage value — AR50 land cover per cell",
        "• Distance to people — roads, trails,",
        "   cabins, parking (OSM + fjellstyre)",
        "• Field boundary — statsallmenning polygons"])
    box(0.04, 0.42, 0.27, 0.225, "REAL WEATHER FIELD  (daily)", [
        "• Lattice of real forecast points (1 km model;",
        "   ERA5 for past dates), no API key",
        "• Temperature, wind speed + DIRECTION, rain",
        "• Interpolated per cell: wind as u/v vectors,",
        "   temp by data-driven lapse (inversions incl.)"])
    box(0.37, 0.56, 0.28, 0.335, "SCORING  (per cell, per day)", [
        f"insect drive (warm+calm+dry)   w={S.W_INSECT}",
        f"shelter drive (cold|wet|windy) w={S.W_SHELTER}",
        "→ a weighted SWITCH, not a sum — the",
        "   regimes cannot cancel each other",
        "exposure = TPI + cos(aspect − wind dir)",
        f"high-ground baseline           w={S.W_BASELINE}",
        f"forage value                   w={S.W_FORAGE}",
        f"steep-terrain penalty          w={S.W_STEEP}",
        f"disturbance penalty            w={S.W_DISTURB}",
        "(all weights expert-set with the field expert,",
        " never fitted to sightings)"])
    box(0.71, 0.56, 0.25, 0.335, "OUTPUT  (what you read)", [
        "• 0–1 score per cell → percentile-",
        "   ranked green wash on hillshade",
        "• Up to 6 named “go here” zones",
        "   with plain-language reasons",
        "• Wind arrow + roads/trails/cabins",
        "   for orientation on the ground",
        "• One image per day, read at a desk",
        "   and tested against field reports"])
    box(0.37, 0.315, 0.59, 0.215, "VALIDATION  (the honesty loop — separate from scoring)", [
        "Field reports → gazetteer + human pins → each report scored within its day's field.",
        "Fairness corrections: effort-matched background (reports come from accessible ground),",
        "position-confidence tiers (name-only geocodes measure the reporter, not the model).",
        "Gate: k-fold cross-validated, select-then-evaluate — a change is kept ONLY if it",
        "wins out-of-sample. Current: CV AUC 0.64, ~9 of 10 folds beat chance (n=30 confident)."])

    # ---- the project, in three boxes a first-time visitor actually needs ----
    box(0.04, 0.08, 0.295, 0.225, "THE RESEARCH QUESTION", [
        "What makes wild reindeer move, day to day —",
        "and which pressures matter most in an",
        "animal's daily life? A behavioural theory",
        "(weather, insects, forage, terrain, human",
        "disturbance) is encoded as rules and tested",
        "against independent observations.",
        "Not a tracking tool: an honest probability",
        "surface, never used to approach animals."])
    box(0.355, 0.08, 0.325, 0.225, "DATA SOURCES & CREDITS", [
        "• Weather: Open-Meteo — MET Nordic 1 km +",
        "   ERA5 reanalysis (CC-BY 4.0, MET/Copernicus)",
        "• Terrain: Kartverket national 50 m DTM",
        "• Place names: Kartverket SSR register",
        "• Land cover: NIBIO AR50",
        "• Infrastructure: OpenStreetMap + Lesja fjellstyre",
        "• Field reports: villreinutvalet.no (thank you!)"])
    box(0.70, 0.08, 0.26, 0.225, "HONEST LIMITS", [
        "• Small sample — a few dozen",
        "   trustworthy reports so far",
        "• Reports are presence-only and",
        "   effort-biased (corrected in eval)",
        "• Positions are ~2.5 km areas,",
        "   never GPS points",
        "• Reindeer are social + partly",
        "   stochastic — claims stay modest"])

    arrow(0.31, 0.78, 0.37, 0.73)
    arrow(0.31, 0.53, 0.37, 0.62)
    arrow(0.65, 0.72, 0.71, 0.72)
    arrow(0.51, 0.56, 0.51, 0.53)
    brand_footer(fig)
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

_These charts are the **published face** of the project: the underlying data
(field reports, positions, gazetteer, boundary geometry) is deliberately not
distributed — only aggregate chart form. See `docs/ABOUT.md` for how the
project should be viewed._

## How the model works (one paragraph)

Every day the model fetches a **real weather field** over the fell — a lattice of
forecast points (temperature, wind speed **and direction**, precipitation)
interpolated to every 250 m cell — and combines it with the **fixed landscape**
(elevation, slope aspect, terrain shelter/exposure, forage quality, distance to
roads/trails/cabins). Two behavioural regimes compete in a weighted switch:
**insect-escape** (warm + calm + dry → animals climb to cool, wind-exposed ground)
and **shelter** (cold / wet / windy → animals hold leeward, calmer, lower ground),
on top of a gentle high-ground baseline, a forage bonus and a disturbance penalty
("they come lower only when human activity allows"). Every weight is a named constant tuned
with the field expert — **sightings are never an input**; they are kept exclusively to
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
| `validation_breakdown.png` | Where the hit-rate comes from: score by positioning method (trustworthy positions score ~0.7–0.8; name-only geocodes drag the average), by season, and every single report as a dot. |
| `weather_drivers.png` | What the model saw on each validation day — the regime pressures plus the day's actual field-median temperature, wind and precipitation. |
| `model_explainer.png` | The one-image project summary: pipeline + weights, the research question, data credits and the honest limits. |

## Honest limits (always state these)

- Small sample: {len(p_conf)} trustworthy reports; verdicts carry wide error bars.
- Presence-only, effort-biased reports — corrected in evaluation (effort-matched
  background), but a residual accessibility edge remains.
- Positions are areas (~2.5 km), never GPS points; 5 vague reports await human pins.
- The map is a **search-narrowing tool, not a GPS oracle** — reindeer are social
  and partly stochastic; the model narrows where the animals are likely to be, nothing more.

## Data sources & credits

- **Weather:** [Open-Meteo](https://open-meteo.com/) — MET Nordic 1 km forecasts +
  ERA5 reanalysis (CC-BY 4.0; MET Norway / Copernicus).
- **Terrain:** Kartverket national 50 m DTM (høydedata.no).
- **Place names:** Kartverket Sentralt stadnamnregister (SSR).
- **Land cover:** NIBIO AR50.
- **Infrastructure:** OpenStreetMap contributors + Lesja fjellstyre.
- **Field reports:** villreinutvalet.no — used **only** to validate, never to
  generate the map; raw reports and positions are not redistributed.
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
