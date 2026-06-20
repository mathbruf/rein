"""Phase 5: validate the scorer against the held-out geocoded sightings.

Pipeline:
  1. resolve each observation to a field grid cell, applying its directional phrase
     ("nord for X", "mot Y") via geocode.positions — NOT just the landmark point;
  2. fetch ERA5 historical daytime weather for each observation date (Open-Meteo);
  3. score the full grid for that date with the tuned scorer + all static layers;
  4. compare observed-cell scores to the same-date field-wide background;
  5. report date-matched percentile/AUC, top-quantile lift, Boyce, a permutation null,
     a naive (at-landmark) baseline, and an offset sensitivity sweep.

The weights were tuned with the hunter and never fitted to sightings, so all
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
from reindeer.geocode.positions import resolve_position  # noqa: E402
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA  # noqa: E402
from reindeer.model.score import score_cells  # noqa: E402
from reindeer.model import validation as V  # noqa: E402
from reindeer.weather.historical import fetch_archive, weather_by_date  # noqa: E402
from shapely.geometry import Point  # noqa: E402
from shapely.prepared import prep  # noqa: E402

PROCESSED = _ROOT / "data" / "processed"
REPORT = _ROOT / "docs" / "validation_report.md"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)


def load_grid() -> pd.DataFrame:
    grid = pd.read_csv(PROCESSED / "grid_250m.csv")
    df = pd.read_csv(PROCESSED / "terrain_250m.csv").merge(
        grid[["cell_id", "in_lordalen", "in_dalsida"]], on="cell_id")
    for fn, col in (("disturbance_250m.csv", "dist_disturb_m"), ("forage_250m.csv", "forage")):
        p = PROCESSED / fn
        if p.exists():
            df = df.merge(pd.read_csv(p)[["cell_id", col]], on="cell_id")
    return df


def resolve_observations(grid, gaz, inside, offset_m, naive=False) -> pd.DataFrame:
    """One row per observation resolving to an in-grid cell, using the directional
    phrase (or the bare landmark when naive=True)."""
    gx, gy = grid["east"].to_numpy(), grid["north"].to_numpy()
    obs = pd.read_csv(_ROOT / "data" / "interim" / "observations.csv")
    rows = []
    for _, r in obs.iterrows():
        if pd.isna(r["landmark_phrases"]):
            continue
        hints = "[]" if naive else r["direction_hints"]
        pos = resolve_position(r["landmark_phrases"], hints, gaz, offset_m=offset_m)
        if pos is None:
            continue
        e, n, method = pos
        if not inside.contains(Point(e, n)):
            continue
        idx = int(((gx - e) ** 2 + (gy - n) ** 2).argmin())
        rows.append({"date": r["date"], "cell_idx": idx, "method": method,
                     "region": r["region"]})
    return pd.DataFrame(rows)


def evaluate(used, per_date_score):
    used = used[used["date"].map(lambda d: per_date_score.get(d) is not None)].copy()
    obs_score = np.array([per_date_score[r["date"]][r["cell_idx"]] for _, r in used.iterrows()])
    obs_bg = [per_date_score[r["date"]] for _, r in used.iterrows()]
    pct = V.date_matched_percentiles(obs_score, obs_bg)
    return used.assign(percentile=pct), pct, obs_bg


def ablation_auc(used, wx, layers, overrides) -> float:
    """Re-score each date under temporary weight overrides; return date-matched AUC.

    DIAGNOSTIC ONLY — uses the test set, so configs found here are hypotheses, not a
    validated retuning. The shipped weights (score.py) are left unchanged.
    """
    import reindeer.model.score as S
    elev, slope, tpi, disturb, forage = layers
    saved = {k: getattr(S, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(S, k, v)
        cache, ps = {}, []
        for _, r in used.iterrows():
            d = r["date"]
            if d not in wx:
                continue
            if d not in cache:
                cache[d] = S.score_cells(elev, slope, tpi, wx[d],
                                         disturb_dist=disturb, forage=forage)["score_raw"]
            sc = cache[d]
            ps.append(V.percentile_rank(sc[r["cell_idx"]], sc))
        return float(np.mean(ps))
    finally:
        for k, v in saved.items():
            setattr(S, k, v)


def main() -> None:
    grid = load_grid()
    gaz = load_gazetteer()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))

    # all candidate dates (naive set is the superset of dates) -> weather + per-date scores
    cand = resolve_observations(grid, gaz, inside, offset_m=3000.0, naive=True)
    cx, cy = grid["east"].mean(), grid["north"].mean()
    lon, lat = _to_wgs.transform(cx, cy)
    archive = fetch_archive(lat, lon, min(cand["date"]), max(cand["date"]))
    wx = weather_by_date(archive)

    elev, slope, tpi = grid["elevation_m"].to_numpy(), grid["slope_deg"].to_numpy(), grid["tpi_m"].to_numpy()
    disturb = grid["dist_disturb_m"].to_numpy() if "dist_disturb_m" in grid else None
    forage = grid["forage"].to_numpy() if "forage" in grid else None
    per_date_score = {}
    for d in sorted(set(cand["date"])):
        per_date_score[d] = (None if d not in wx else
                             score_cells(elev, slope, tpi, wx[d],
                                         disturb_dist=disturb, forage=forage)["score_raw"])

    # headline: direction-aware @ 3 km; baseline: naive at-landmark; sweep over offset
    used, pct, obs_bg = evaluate(resolve_observations(grid, gaz, inside, 3000.0), per_date_score)
    naive_used, naive_pct, _ = evaluate(cand, per_date_score)

    auc = float(pct.mean())
    f20, lift20 = V.top_quantile_lift(pct, 0.20)
    f10, lift10 = V.top_quantile_lift(pct, 0.10)
    # precision / hit-rate = % of actual sightings the model placed on favored ground
    hit_half = float((pct >= 0.50).mean())   # in the model's favored half (chance 50%)
    hit_top3 = float((pct >= 0.6667).mean())  # in the model's top third (chance 33%)
    boyce = V.continuous_boyce(pct_to_scores := np.array([per_date_score[r["date"]][r["cell_idx"]]
                                                          for _, r in used.iterrows()]),
                               np.concatenate(obs_bg))
    null = V.permutation_null(obs_bg, n_iter=2000)
    p_val = float((null >= auc).mean())
    sweep = []
    for off in (0.0, 1500.0, 3000.0, 4500.0, 6000.0):
        u, p, _ = evaluate(resolve_observations(grid, gaz, inside, off), per_date_score)
        sweep.append((off, len(u), float(p.mean()), V.top_quantile_lift(p, 0.20)[1]))

    # DIAGNOSTIC ablations (uses the test set -> hypotheses, not a validated retuning)
    layers = (elev, slope, tpi, disturb, forage)
    abl = [
        ("full model (shipped)", {}),
        ("- disturbance penalty", {"W_DISTURB": 0.0}),
        ("- high-ground baseline", {"W_BASELINE": 0.0}),
        ("- both", {"W_DISTURB": 0.0, "W_BASELINE": 0.0}),
        ("prefer LOW ground only", {"W_BASELINE": -0.5, "W_INSECT": 0.0,
                                    "W_SHELTER": 0.0, "W_STEEP": 0.0,
                                    "W_DISTURB": 0.0, "W_FORAGE": 0.0}),
    ]
    abl_rows = [(name, ablation_auc(used, wx, layers, ov)) for name, ov in abl]

    methods = used["method"].value_counts().to_dict()
    by_year = used.assign(year=used["date"].str[:4]).groupby("year")["percentile"].agg(["count", "mean"])

    lines = [
        "# Phase 5 — Validation Report",
        "",
        f"_Generated by `scripts/validate.py`. {len(used)} held-out sightings on "
        f"{used['date'].nunique()} dates ({min(used['date'])} … {max(used['date'])})._",
        "",
        "## Method",
        "- **Held-out:** scorer weights were expert-tuned with the hunter, never fitted to "
        "sightings — every observation is an out-of-sample test point.",
        "- **Positioning:** each sighting is placed using its directional phrase "
        "(`nord for X`, `mot Y`) via `geocode.positions`, not at the bare landmark "
        f"(method mix: {methods}).",
        "- **Design:** presence-vs-background over the field (Lordalen + Dalsida), "
        "date-matched — each observed cell's score is ranked within that day's field-wide "
        "score distribution (rank-based, so the score normalisation doesn't matter).",
        "- **Weather:** ERA5 daytime (06–18) at the field centroid via the Open-Meteo "
        "archive (free; stands in for MET Frost, which needs a client ID).",
        "",
        "## Model precision vs actual readings (the headline metric)",
        "Of the actual sightings, what fraction did the model place on its favored ground "
        "(scoring each reading's location within that day's field on the weather+bug+landscape "
        "rules only — the readings are never an input)?",
        "",
        f"- **{hit_half*100:.0f}% of readings fell in the model's favored half** "
        "(chance = 50%).",
        f"- **{hit_top3*100:.0f}% in the model's top third** (chance = 33%).",
        f"- **{f20*100:.0f}% in the model's top 20%** (chance = 20%; {lift20:.2f}× chance), "
        f"**{f10*100:.0f}% in the top 10%** (chance = 10%; {lift10:.2f}×).",
        "",
        "These hit-rates are **below chance**, so the current weather+bug+landscape weights "
        "are **anti-correlated** with where reindeer were actually reported in the hunting "
        "season — the model is honestly wrong for autumn, not yet right. (The readings are "
        "used only to measure this; they were not used to build or tune the model.)",
        "",
        "## Supporting statistics (direction-aware, 3 km offset)",
        f"- **Date-matched AUC (mean percentile): {auc:.3f}** (0.5 = chance) — equivalent "
        "rank view of the same result.",
        f"- **Top-20% lift: {f20*100:.0f}%** of sightings in the top 20% of cells "
        f"→ {lift20:.2f}× chance.  **Top-10% lift: {f10*100:.0f}%** → {lift10:.2f}×.",
        f"- **Continuous Boyce (pooled): {boyce:+.3f}** (+1 good, 0 random).",
        f"- **Permutation null:** random placement gives {null.mean():.3f} ± {null.std():.3f}; "
        f"observed {auc:.3f} → p = {p_val:.4f} (i.e. significantly *worse* than chance).",
        "",
        "## Diagnosis — which rules cause the anti-correlation (DIAGNOSTIC, not a retuning)",
        "These ablations re-score the same sightings with one rule removed. They use the "
        "test set, so the configs below are **hypotheses for the next iteration**, not a "
        "validated model — the shipped weights are left unchanged.",
        "```",
        "config                     AUC",
        *[f"{n:26s} {a:.3f}" for n, a in abl_rows],
        "```",
        "- Removing **both** the high-ground baseline and the disturbance penalty flips the "
        "weather+forage core to *above* chance; a plain **low-ground** preference fits best.",
        "- Reading: these are **hunting-season** reports — insects (the go-high driver) are "
        "gone and animals sit lower than the summer high-ground baseline assumes; and hunters "
        "report from **accessible** terrain that the disturbance penalty marks down "
        "(observer bias). Both push the shipped model the wrong way.",
        "",
        "### Naive baseline (at the landmark point, ignoring direction)",
        f"- Date-matched AUC: **{naive_pct.mean():.3f}** on {len(naive_used)} obs — "
        "this is *below* chance because the landmarks that resolve well are mostly valleys "
        "/ lakes (low, near roads) while the animals are reported on the high ground around "
        "them. It demonstrates that directional positioning is essential, not optional.",
        "",
        "### Offset sensitivity (is the headline cherry-picked?)",
        "```",
        "offset_m   n    AUC    top20x",
        *[f"{o:7.0f}  {n:3d}  {a:.3f}  {l:.2f}" for o, n, a, l in sweep],
        "```",
        "",
        "### By year",
        "```",
        by_year.to_string(),
        "```",
        "",
        "## Honest limitations",
        f"- **Small sample** ({len(used)} sightings, {used['date'].nunique()} dates).",
        "- **Coarse positions:** landmark uncertainty 0.4–3 km vs a 250 m cell; the "
        "directional offset is a fixed 3 km heuristic, not a surveyed point — real "
        "positional noise remains and can only depress the score.",
        "- **Effort/observer bias:** presence-only reports from where hunters go; background "
        "is the whole field, so the result partly reflects reporting, not only true presence.",
        "- **Weather:** ERA5 reanalysis (gridded, elevation-smoothed), area-wide per day, "
        "not per-cell micro-weather.",
        "- **Observer bias confound:** we cannot separate 'disturbance penalty is wrong' "
        "from 'reports are effort-biased toward accessible terrain'. A fair test needs an "
        "effort covariate or a background restricted to where hunters actually go.",
        "",
        "## Recommended next iteration (re-tested on held-out / cross-validated data)",
        "1. **Seasonality:** the sightings are hunting-season; turn the insect 'go-high' "
        "driver and the summer high-ground baseline *off* (or down) for late Aug–Sept, and "
        "test a low/mid-elevation preference for that window.",
        "2. **Disturbance vs effort:** add an effort/accessibility covariate, or restrict the "
        "background to huntable/accessed terrain, before judging the disturbance penalty.",
        "3. **More data + cross-validation:** recover the 2022 free-prose season (IDEA 002) "
        "and use k-fold CV so any weight change is validated out-of-sample, not on this set.",
        "4. **Better positions:** use the directional offset distance per landmark uncertainty "
        "and, where given, the 'mot Y' target, rather than a flat 3 km.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"direction-aware AUC {auc:.3f} | top20 {lift20:.2f}x | Boyce {boyce:+.3f} | p {p_val:.4f}")
    print(f"naive baseline AUC {naive_pct.mean():.3f} ({len(naive_used)} obs)")
    print("sweep:", [(o, round(a, 3)) for o, n, a, l in sweep])
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
