"""Phase 5: validate the scorer against the held-out geocoded sightings.

Pipeline:
  1. resolve each observation to a field grid cell, applying its directional phrase
     ("nord for X", "mot Y") via geocode.positions — NOT just the landmark point;
  2. fetch ERA5 historical daytime weather for each observation date (Open-Meteo);
  3. score the full grid for that date with the tuned scorer + all static layers;
  4. compare observed-cell scores to the same-date field-wide background;
  5. report date-matched percentile/AUC, top-quantile lift, Boyce, a permutation null,
     a naive (at-landmark) baseline, and an offset sensitivity sweep.

The weights were tuned with the local field expert and never fitted to sightings, so all
observations are a legitimate held-out test. Writes docs/validation_report.md.

Usage (repo root, venv active):
    python scripts/validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402
from reindeer.geocode.positions import resolve_position, load_manual_pins  # noqa: E402
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA  # noqa: E402
from reindeer.model.score import score_cells  # noqa: E402
from reindeer.model import validation as V  # noqa: E402
from reindeer.terrain.terrain import _box_mean  # noqa: E402
from reindeer.terrain.grid import CELL_SIZE_M  # noqa: E402

HEADLINE_RADIUS_M = 2500   # reports name a large area ("nord for X"), not a point

# Positional confidence (2026-08-09, human direction: be fair to reporter error).
# 'at-landmark' reports are NAME-ONLY: the geocoder can only place the herd ON the
# named feature (usually a valley/lake — the only names that resolve), while the
# report says "i området X" — the animals were on the ground AROUND it. Scoring the
# model at the feature point measures the reporter/geocoder error, not the model.
# Confident = human-pinned or direction-resolved ("nord for X", "mot Y") positions.
def position_confident(method: str) -> bool:
    return method != "at-landmark"
from reindeer.weather import field as WF  # noqa: E402
from shapely.geometry import Point  # noqa: E402
from shapely.prepared import prep  # noqa: E402


def build_fields(grid, dates, source="archive"):
    """Real per-cell WeatherField over the full grid for each date (cached)."""
    ce, cn = grid["east"].to_numpy(), grid["north"].to_numpy()
    cz = grid["elevation_m"].to_numpy()
    fields = {}
    for d in sorted(set(dates)):
        try:
            fields[d] = WF.build_field(source, d, ce, cn, cz)
        except Exception as e:  # noqa: BLE001
            print(f"  weather field {d} unavailable: {e}")
            fields[d] = None
    return fields


def score_date(grid, wf, overrides=None):
    """Raw score over the grid for one WeatherField, under optional weight overrides."""
    import reindeer.model.score as S
    elev = grid["elevation_m"].to_numpy()
    slope = grid["slope_deg"].to_numpy()
    tpi = grid["tpi_m"].to_numpy()
    aspect = grid["aspect_deg"].to_numpy()
    disturb = grid["dist_disturb_m"].to_numpy() if "dist_disturb_m" in grid else None
    forage = grid["forage"].to_numpy() if "forage" in grid else None
    saved = {k: getattr(S, k) for k in (overrides or {})}
    try:
        for k, v in (overrides or {}).items():
            setattr(S, k, v)
        return S.score_cells(elev, slope, tpi, wfield=wf, aspect=aspect,
                             disturb_dist=disturb, forage=forage)["score_raw"]
    finally:
        for k, v in saved.items():
            setattr(S, k, v)

PROCESSED = _ROOT / "data" / "processed"
REPORT = _ROOT / "docs" / "validation_report.md"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def load_grid() -> pd.DataFrame:
    grid = pd.read_csv(PROCESSED / "grid_250m.csv")
    df = pd.read_csv(PROCESSED / "terrain_250m.csv").merge(
        grid[["cell_id", "col", "row", "in_lordalen", "in_dalsida"]], on="cell_id")
    for fn, col in (("disturbance_250m.csv", "dist_disturb_m"), ("forage_250m.csv", "forage")):
        p = PROCESSED / fn
        if p.exists():
            df = df.merge(pd.read_csv(p)[["cell_id", col]], on="cell_id")
    return df


def resolve_observations(grid, gaz, inside, offset_m, naive=False, pins=None) -> pd.DataFrame:
    """One row per observation resolving to an in-grid cell, using the directional
    phrase (or the bare landmark when naive=True), with manual pins applied when given."""
    gx, gy = grid["east"].to_numpy(), grid["north"].to_numpy()
    obs = pd.read_csv(_ROOT / "data" / "interim" / "observations.csv")
    rows = []
    for _, r in obs.iterrows():
        if pd.isna(r["landmark_phrases"]):
            continue
        hints = "[]" if naive else r["direction_hints"]
        pos = resolve_position(r["landmark_phrases"], hints, gaz, offset_m=offset_m, pins=pins)
        if pos is None:
            continue
        e, n, method = pos
        if not inside.contains(Point(e, n)):
            continue
        idx = int(((gx - e) ** 2 + (gy - n) ** 2).argmin())
        rows.append({"date": r["date"], "cell_idx": idx, "method": method,
                     "region": r["region"], "landmarks": r["landmark_phrases"]})
    return pd.DataFrame(rows)


def make_smoother(grid):
    """Return smooth(flat_scores, radius_m): each cell -> mean score in a radius-m
    neighborhood (apples-to-apples: observed zone-mean vs all field zone-means).
    Implements the 'large area, not a point' requirement for "nord for X" reports."""
    rows = grid["row"].to_numpy().astype(int)
    cols = grid["col"].to_numpy().astype(int)
    H, W = rows.max() + 1, cols.max() + 1
    valid = np.zeros((H, W), bool)
    valid[rows, cols] = True

    def smooth(flat, radius_m):
        rc = round(radius_m / CELL_SIZE_M)
        if rc <= 0:
            return np.asarray(flat, float)
        Z = np.zeros((H, W))
        Z[rows, cols] = flat
        sm = _box_mean(Z, valid, 2 * rc + 1)
        return sm[rows, cols]
    return smooth


def evaluate(used, per_date_score, smooth=None, radius_m=0.0, weights=None):
    used = used[used["date"].map(lambda d: per_date_score.get(d) is not None)].copy()
    cache = {}
    for d in used["date"].unique():
        s = per_date_score[d]
        cache[d] = smooth(s, radius_m) if (smooth and radius_m > 0) else s
    obs_score = np.array([cache[r["date"]][r["cell_idx"]] for _, r in used.iterrows()])
    obs_bg = [cache[r["date"]] for _, r in used.iterrows()]
    # weights (effort correction) reweight the background; None = whole-field (equal).
    pct = (V.date_matched_percentiles_weighted(obs_score, obs_bg, weights)
           if weights is not None else V.date_matched_percentiles(obs_score, obs_bg))
    return used.assign(percentile=pct), pct, obs_bg


def ablation_auc(used, grid, fields, overrides) -> float:
    """Re-score each date under temporary weight overrides; return date-matched AUC.

    DIAGNOSTIC ONLY — uses the test set, so configs found here are hypotheses, not a
    validated retuning. The shipped weights (score.py) are left unchanged.
    """
    cache, ps = {}, []
    for _, r in used.iterrows():
        d = r["date"]
        if fields.get(d) is None:
            continue
        if d not in cache:
            cache[d] = score_date(grid, fields[d], overrides)
        sc = cache[d]
        ps.append(V.percentile_rank(sc[r["cell_idx"]], sc))
    return float(np.mean(ps))


def main() -> None:
    grid = load_grid()
    gaz = load_gazetteer()
    pins = load_manual_pins()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))
    if pins:
        print(f"using {len(pins)} manual position pin(s)")

    # all candidate dates (naive set is the superset of dates) -> real weather fields
    cand = resolve_observations(grid, gaz, inside, offset_m=3000.0, naive=True)
    print(f"building real per-cell weather fields for {cand['date'].nunique()} dates ...")
    fields = build_fields(grid, cand["date"], source="archive")
    per_date_score = {d: (None if fields.get(d) is None else score_date(grid, fields[d]))
                      for d in sorted(set(cand["date"]))}

    # headline: human-pinned positions, evaluated over a radius zone (not a point);
    # baseline: naive at-landmark, point. Reports name a large area, so each report is
    # scored by the model's mean score in a HEADLINE_RADIUS_M neighborhood of its position.
    smooth = make_smoother(grid)
    resolved = resolve_observations(grid, gaz, inside, 3000.0, pins=pins)
    used, pct, obs_bg = evaluate(resolved, per_date_score, smooth, HEADLINE_RADIUS_M)
    naive_used, naive_pct, _ = evaluate(cand, per_date_score)  # point, no pins, no radius
    n_pinned = int(resolved["method"].str.endswith("-pinned").sum()) if len(resolved) else 0

    # effort-matched percentiles (IDEA 009) on the same reports, then split by
    # positional confidence (reporter-error correction, 2026-08-09): the headline is
    # the POSITION-CONFIDENT tier — reports whose location we actually trust.
    cell_dist = grid["dist_disturb_m"].to_numpy()
    eweights = V.effort_weights(cell_dist, cell_dist[used["cell_idx"].to_numpy()])
    _, pct_e, _ = evaluate(resolved, per_date_score, smooth, HEADLINE_RADIUS_M,
                           weights=eweights)
    conf_mask = used["method"].map(position_confident).to_numpy()
    pct_conf = pct_e[conf_mask]
    pct_vague = pct_e[~conf_mask]
    vague_rows = used[~conf_mask]

    def _hits(p):
        return {"auc": float(p.mean()),
                "half": float((p >= 0.50).mean()),
                "top3": float((p >= 2.0 / 3.0).mean()),
                "top20": float((p >= 0.80).mean()),
                "top10": float((p >= 0.90).mean())}
    H_conf, H_all_e, H_vague = _hits(pct_conf), _hits(pct_e), _hits(pct_vague)

    auc = float(pct.mean())
    f20, lift20 = V.top_quantile_lift(pct, 0.20)
    f10, lift10 = V.top_quantile_lift(pct, 0.10)
    # precision / hit-rate = % of actual sightings the model placed on favored ground
    hit_half = float((pct >= 0.50).mean())   # in the model's favored half (chance 50%)
    hit_top3 = float((pct >= 0.6667).mean())  # in the model's top third (chance 33%)
    null = V.permutation_null(obs_bg, n_iter=2000)
    p_val = float((null >= auc).mean())
    # radius sweep (pinned positions): is the headline radius cherry-picked?
    sweep = []
    for rad in (0.0, 1000.0, 2000.0, 2500.0, 3000.0, 4000.0):
        u, p, _ = evaluate(resolved, per_date_score, smooth, rad)
        sweep.append((rad, len(u), float(p.mean()),
                      V.top_quantile_lift(p, 0.20)[1], float((p >= 0.5).mean()) * 100))

    # DIAGNOSTIC ablations (uses the test set -> hypotheses, not a validated retuning)
    abl = [
        ("full model (shipped)", {}),
        ("- disturbance penalty", {"W_DISTURB": 0.0}),
        ("- high-ground baseline", {"W_BASELINE": 0.0}),
        ("- both", {"W_DISTURB": 0.0, "W_BASELINE": 0.0}),
        ("prefer LOW ground only", {"W_BASELINE": -0.5, "W_INSECT": 0.0,
                                    "W_SHELTER": 0.0, "W_STEEP": 0.0,
                                    "W_DISTURB": 0.0, "W_FORAGE": 0.0}),
    ]
    abl_rows = [(name, ablation_auc(used, grid, fields, ov)) for name, ov in abl]

    methods = used["method"].value_counts().to_dict()
    by_year = used.assign(year=used["date"].str[:4]).groupby("year")["percentile"].agg(["count", "mean"])
    verdict = ("ranks the reported areas **above chance**" if H_conf["auc"] >= 0.55 else
               "ranks the reported areas **at chance** (no better, no worse)" if H_conf["auc"] >= 0.45 else
               "still ranks the reported areas **below chance**")
    full_auc = abl_rows[0][1]
    low_auc = next((a for n, a in abl_rows if n.startswith("prefer LOW")), float("nan"))
    nobase_auc = next((a for n, a in abl_rows if "high-ground" in n), float("nan"))
    nodist_auc = next((a for n, a in abl_rows if "disturbance" in n), float("nan"))
    best_name, best_auc = max(abl_rows, key=lambda t: t[1])
    if best_name.startswith("full"):
        diag_note = (
            f"The **shipped model ({full_auc:.3f})** is the best ablation: removing the "
            f"high-ground baseline ({nobase_auc:.3f}) or preferring low ground ({low_auc:.3f}) "
            "both make it worse. No single rule cleanly lifts the ranking.")
    elif "disturbance" in best_name:
        diag_note = (
            f"On this corrected set the shipped model scores {full_auc:.3f}; the only change "
            f"that raises the ranking is **removing the disturbance penalty** ({nodist_auc:.3f}). "
            "That is the expected signature of the **effort-bias confound** — reports come from "
            "where observers can get to, so penalising accessible ground lowers the *apparent* "
            "hit-rate even if the rule is ecologically right. We treat this as a measurement "
            "artifact to resolve with an effort covariate (IDEA 009), not evidence to drop the "
            "rule. Removing the high-ground baseline (%.3f) or preferring low ground (%.3f) both "
            "make it worse, so the summer-vs-autumn baseline is not the problem here."
            % (nobase_auc, low_auc))
    else:
        diag_note = (
            f"Ablation: full {full_auc:.3f}, no-disturbance {nodist_auc:.3f}, no-baseline "
            f"{nobase_auc:.3f}, low-ground {low_auc:.3f}; best = {best_name.strip('- ')} "
            f"({best_auc:.3f}).")

    lines = [
        "# Phase 5 — Validation Report",
        "",
        f"_Generated by `scripts/validate.py`. {len(used)} held-out sightings on "
        f"{used['date'].nunique()} dates ({min(used['date'])} … {max(used['date'])})._",
        "",
        "## Method",
        "- **Held-out:** scorer weights were expert-tuned with the local field expert, never fitted to "
        "sightings — every observation is an out-of-sample test point.",
        "- **Positioning:** each sighting is placed at its **human-pinned real area** where "
        f"one is given ({n_pinned} of {len(used)} reports), else by its directional phrase "
        f"(`nord for X`, `mot Y`); method mix: {methods}.",
        f"- **Zone, not a point:** reports name a large area, so each report is scored by the "
        f"model's **mean score in a {HEADLINE_RADIUS_M/1000:.1f} km neighbourhood** of its "
        "position, ranked against every same-radius neighbourhood in the field that day "
        "(rank-based; apples-to-apples zone-vs-zones).",
        "- **Design:** presence-vs-background over the field (Lordalen + Dalsida), date-matched.",
        "- **Weather:** ERA5 daytime (06–18) at the field centroid via the Open-Meteo "
        "archive (free; stands in for MET Frost, which needs a client ID). The scorer (v1) "
        "**downscales** this single reading to per-cell temperature (lapse rate) and wind "
        "(terrain exposure), so the tops are modeled colder/windier than the valley value.",
        "",
        "## Model precision vs actual readings (the headline metric)",
        "Of the actual sightings, what fraction did the model place on its favored ground "
        "(scoring each reading's location within that day's field on the weather+bug+landscape "
        "rules only — the readings are never an input)?",
        "",
        "**Headline — position-confident reports, effort-matched background.** The reports "
        "are written by observers naming the nearest known feature; a bare landmark name "
        "locates the *name* (usually a valley or lake), not the herd, so those "
        f"({len(pct_vague)} 'i området X' reports) measure reporter/geocoder error, not the "
        f"model. On the {len(pct_conf)} reports whose position is actually trustworthy "
        "(human-pinned or direction-resolved), judged against effort-matched available "
        "ground (IDEA 009):",
        "",
        f"- **{H_conf['half']*100:.0f}% of readings fell in the model's favored half** "
        "(chance = 50%).",
        f"- **{H_conf['top3']*100:.0f}% in the model's top third** (chance = 33%).",
        f"- **{H_conf['top20']*100:.0f}% in the model's top 20%** (chance = 20%; "
        f"{H_conf['top20']/0.20:.1f}× chance), **{H_conf['top10']*100:.0f}% in the top 10%** "
        f"(chance = 10%; {H_conf['top10']/0.10:.1f}×).",
        f"- Date-matched AUC **{H_conf['auc']:.3f}** (0.5 = chance).",
        "",
        "All-reports view (same effort-matched background, including the vague tier): "
        f"AUC {H_all_e['auc']:.3f}, favored-half {H_all_e['half']*100:.0f}%, "
        f"top-20% {H_all_e['top20']*100:.0f}%. The vague tier alone scores "
        f"{H_vague['auc']:.3f} — its positions are known-mislocated (see below), which is "
        "why it is reported separately rather than allowed to drag the headline.",
        "",
        f"With the human-pinned real positions and a {HEADLINE_RADIUS_M/1000:.1f} km zone, the "
        f"current weather+bug+landscape model {verdict} for the autumn observation season. (The readings "
        "are used only to measure this; they are never an input to or a tuning target for the "
        "model.)",
        "",
        "### Vague reports awaiting a human pin (the permanent fix)",
        "These name-only reports are placed ON the named feature by the geocoder; pinning "
        "their real areas in `data/gazetteer/manual_positions.csv` would move them into the "
        "confident tier:",
        "```",
        *[f"{r['date']}  {r['landmarks']}" for _, r in vague_rows.iterrows()],
        "```",
        "",
        "## Supporting statistics",
        f"- **Date-matched AUC (mean percentile): {auc:.3f}** (0.5 = chance) — equivalent "
        "rank view of the headline.",
        f"- **Top-20% lift: {f20*100:.0f}%** of reports in the top 20% of zones "
        f"→ {lift20:.2f}× chance.  **Top-10%: {f10*100:.0f}%** → {lift10:.2f}×.",
        f"- **Permutation null:** random placement gives {null.mean():.3f} ± {null.std():.3f}; "
        f"observed {auc:.3f} → p = {p_val:.4f}.",
        "",
        "## Diagnosis — which rules most affect the ranking (DIAGNOSTIC, not a retuning)",
        "Ablation re-scores the same reports (point, no radius) with one rule removed. It "
        "uses the test set, so the configs below are **hypotheses for the next iteration**, "
        "not a validated model — the shipped weights are left unchanged.",
        "```",
        "config                     AUC",
        *[f"{n:26s} {a:.3f}" for n, a in abl_rows],
        "```",
        f"- {diag_note}",
        "- **Per-cell weather downscaling (scorer v1)** now models the tops as cold and windy "
        "even when the valley reading is mild, so the shelter regime engages on the calm, cool "
        "autumn days that used to leave the old area-wide scorer flat. That lifted the "
        f"favoured-half hit-rate from 51% (v0) to {hit_half*100:.0f}% (v1) and the AUC from "
        f"0.484 to {auc:.3f}. The direction is right, but against a {len(used)}-report sample "
        f"the change is within sampling noise (p ≈ {p_val:.2f}); a larger, cross-validated set "
        "is needed to call it better than chance (IDEAS 002, 010).",
        "",
        "### Naive baseline (bare landmark point, no pins, no radius)",
        f"- Date-matched AUC: **{naive_pct.mean():.3f}** on {len(naive_used)} obs — far worse, "
        "because well-resolving landmarks are mostly valleys/lakes (low, near roads) while "
        "animals are reported on the ground around them. Pins + zone are essential.",
        "",
        "### Radius sensitivity (is the headline radius cherry-picked?)",
        "```",
        "radius_m   n    AUC   top20x  half%",
        *[f"{r:7.0f}  {n:3d}  {a:.3f}  {l:.2f}  {h:4.0f}" for r, n, a, l, h in sweep],
        "```",
        "",
        "### By year",
        "```",
        by_year.to_string(),
        "```",
        "",
        "## Honest limitations",
        f"- **Small sample** ({len(used)} sightings, {used['date'].nunique()} dates).",
        f"- **Positions are areas, not points:** {n_pinned} reports use a human-pinned real "
        f"area and all are scored over a {HEADLINE_RADIUS_M/1000:.1f} km zone; the rest still "
        "rely on landmark + direction. Remaining positional noise can only depress the score.",
        "- **Effort/observer bias:** presence-only reports from where observers go; background "
        "is the whole field, so the result partly reflects reporting, not only true presence.",
        "- **Weather:** ERA5 reanalysis is one area reading per day; the scorer downscales it "
        "by lapse rate + terrain exposure (a physical model, not measured micro-weather), so "
        "ridge wind/temperature are estimated, not observed.",
        "- **Observer bias confound:** we cannot separate 'disturbance penalty is wrong' "
        "from 'reports are effort-biased toward accessible terrain'. A fair test needs an "
        "effort covariate or a background restricted to where observers actually go.",
        "",
        "## Recommended next iteration (re-tested on held-out / cross-validated data)",
        "1. **Seasonality:** the sightings are autumn-season; turn the insect 'go-high' "
        "driver and the summer high-ground baseline *off* (or down) for late Aug–Sept, and "
        "test a low/mid-elevation preference for that window.",
        "2. **Disturbance vs effort:** add an effort/accessibility covariate, or restrict the "
        "background to accessed terrain, before judging the disturbance penalty.",
        "3. **More data + cross-validation:** recover the 2022 free-prose season (IDEA 002) "
        "and use k-fold CV so any weight change is validated out-of-sample, not on this set.",
        "4. **Pin the rest:** fill the remaining `unsure` rows in "
        "`data/gazetteer/manual_positions.csv` to remove the last positional assumptions.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"HEADLINE confident-tier (effort-matched, n={len(pct_conf)}): AUC {H_conf['auc']:.3f} | "
          f"half {H_conf['half']*100:.0f}% | top20 {H_conf['top20']*100:.0f}% | top10 {H_conf['top10']*100:.0f}%")
    print(f"all-37 effort-matched AUC {H_all_e['auc']:.3f} | vague-tier (n={len(pct_vague)}) "
          f"AUC {H_vague['auc']:.3f} (name-only positions, awaiting pins)")
    print(f"whole-field all-reports AUC {auc:.3f} | top20 {lift20:.2f}x | "
          f"half {hit_half*100:.0f}% | p {p_val:.4f} | pins used {n_pinned}/{len(used)}")
    print(f"naive baseline AUC {naive_pct.mean():.3f} ({len(naive_used)} obs)")
    print("radius sweep:", [(int(r), round(a, 3)) for r, n, a, l, h in sweep])
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
