"""Phase 6: cross-validated evaluation of the scorer (IDEA 010).

Reuses the data wiring in scripts/validate.py (grid + static layers, ERA5 weather,
pinned positions, the 2.5 km zone smoother) and adds a k-fold CV layer on top:

  1. CV baseline — repeated k-fold mean +/- spread of the shipped model's headline
     (zone-based, date-matched) AUC, so we know how stable the ~chance verdict is.
  2. Select-then-evaluate — an honest procedure for judging validation-motivated
     changes: candidate model variants are selected on the TRAIN folds and scored on
     the held-out TEST fold (model/cv.cv_select_evaluate). The variants below are
     the *diagnostic* ablations only; this run demonstrates the gate and gives the
     baseline. Real Phase-6 fixes are added as new candidate variants and kept only
     if they raise the select-then-evaluate test AUC over the shipped baseline.

Writes docs/cv_report.md.

Usage (repo root, venv active):
    python scripts/cv_validate.py
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

import validate as VAL  # reuse load_grid/resolve_observations/make_smoother/evaluate  # noqa: E402
import reindeer.model.score as S  # noqa: E402
from reindeer.model import cv as CV  # noqa: E402
from reindeer.geocode.gazetteer import load_gazetteer  # noqa: E402
from reindeer.geocode.positions import load_manual_pins  # noqa: E402
from reindeer.terrain.grid import load_field_polygons, LORDALEN, DALSIDA  # noqa: E402
from reindeer.model.score import score_cells  # noqa: E402
from reindeer.weather.historical import fetch_archive, weather_by_date  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.prepared import prep  # noqa: E402

REPORT = _ROOT / "docs" / "cv_report.md"
_to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)

# Candidate variants for the select-then-evaluate demo. Config 0 MUST be the shipped
# model. The rest are the diagnostic ablations (hypotheses only). Phase-6 fixes get
# appended here as they are implemented, and are kept only if they lift the CV test
# AUC of the selection procedure above the shipped baseline.
CANDIDATES = [
    ("shipped", {}),
    ("- disturbance penalty", {"W_DISTURB": 0.0}),
    ("- high-ground baseline", {"W_BASELINE": 0.0}),
]


def per_config_percentiles(overrides, grid, resolved, wx, smooth, radius, weights=None):
    """Per-report zone percentile vector under a set of temporary weight overrides.

    weights=None → whole-field background; an array → effort-matched background
    (IDEA 009 correction, evaluation only).
    """
    elev = grid["elevation_m"].to_numpy()
    slope = grid["slope_deg"].to_numpy()
    tpi = grid["tpi_m"].to_numpy()
    disturb = grid["dist_disturb_m"].to_numpy() if "dist_disturb_m" in grid else None
    forage = grid["forage"].to_numpy() if "forage" in grid else None
    saved = {k: getattr(S, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(S, k, v)
        per_date = {}
        for d in sorted(set(resolved["date"])):
            per_date[d] = (None if d not in wx else
                           score_cells(elev, slope, tpi, wx[d],
                                       disturb_dist=disturb, forage=forage)["score_raw"])
    finally:
        for k, v in saved.items():
            setattr(S, k, v)
    used, pct, _ = VAL.evaluate(resolved, per_date, smooth, radius, weights=weights)
    return used, np.asarray(pct, float)


def main() -> None:
    grid = VAL.load_grid()
    gaz = load_gazetteer()
    pins = load_manual_pins()
    polys = load_field_polygons()
    inside = prep(polys[LORDALEN].union(polys[DALSIDA]))

    resolved = VAL.resolve_observations(grid, gaz, inside, 3000.0, pins=pins)
    cx, cy = grid["east"].mean(), grid["north"].mean()
    lon, lat = _to_wgs.transform(cx, cy)
    archive = fetch_archive(lat, lon, min(resolved["date"]), max(resolved["date"]))
    wx = weather_by_date(archive)
    smooth = VAL.make_smoother(grid)
    radius = VAL.HEADLINE_RADIUS_M

    # effort-matched background weights (IDEA 009): weight each field cell by the
    # empirical density of the reports' own distance-to-disturbance, so "available"
    # ground matches where reports actually occur. Evaluation only; model unchanged.
    cell_dist = grid["dist_disturb_m"].to_numpy() if "dist_disturb_m" in grid else None
    weights = None
    if cell_dist is not None:
        obs_dist = cell_dist[resolved["cell_idx"].to_numpy()]
        from reindeer.model.validation import effort_weights
        weights = effort_weights(cell_dist, obs_dist)

    # per-report percentile matrices over the candidate variants (rows aligned by
    # report), under BOTH the whole-field and the effort-matched backgrounds.
    names = [nm for nm, _ in CANDIDATES]
    P = np.vstack([per_config_percentiles(ov, grid, resolved, wx, smooth, radius)[1]
                   for _, ov in CANDIDATES])
    Pe = (np.vstack([per_config_percentiles(ov, grid, resolved, wx, smooth, radius,
                                            weights=weights)[1]
                     for _, ov in CANDIDATES]) if weights is not None else P)
    n = P.shape[1]

    k = 5
    base = CV.repeated_kfold_stats(P[0], k=k)
    sel = CV.cv_select_evaluate(P, k=k)
    base_e = CV.repeated_kfold_stats(Pe[0], k=k)
    sel_e = CV.cv_select_evaluate(Pe, k=k)

    lines = [
        "# Phase 6 — Cross-validation report",
        "",
        f"_Generated by `scripts/cv_validate.py`. {n} held-out reports, {k}-fold CV "
        "(repeated), zone-based date-matched percentile (the headline metric)._",
        "",
        "## CV baseline — shipped model",
        f"- **{k}-fold CV AUC: {base['mean']:.3f} ± {base['std']:.3f}** "
        f"(test-fold mean ± between-fold std over repeated splits).",
        f"- Fraction of test folds beating chance (>0.5): **{base['fold_beats_chance']*100:.0f}%**.",
        "- Reading: the single-number headline (~0.50) is confirmed as **~chance**, and "
        "the wide between-fold spread shows the verdict is not yet robust on this sample "
        "— exactly why Phase-6 changes must clear CV, not a single split.",
        "",
        "## Select-then-evaluate (honest test of validation-motivated changes)",
        "Per split: pick the candidate with the best **train**-fold AUC, score it on the "
        "held-out **test** fold. If letting the validation set choose among variants "
        "generalised, the selection AUC would beat the shipped baseline; if it is just "
        "in-sample optimism, it will not.",
        "",
        f"- Shipped baseline, CV test AUC: **{sel['baseline_mean']:.3f} ± {sel['baseline_std']:.3f}**",
        f"- Selection procedure, CV test AUC: **{sel['select_mean']:.3f} ± {sel['select_std']:.3f}**",
        "",
        "Candidate variants and how often each was selected on the train folds:",
        "```",
        "config                     selected%",
        *[f"{nm:26s} {fr*100:5.0f}" for nm, fr in zip(names, sel["select_frac"])],
        "```",
        f"- Delta (selection − baseline): **{sel['select_mean']-sel['baseline_mean']:+.3f}** "
        "AUC. A positive, stable delta is the bar a real fix must clear.",
        "",
        "**Interpretation (important):** the variant selected here is *removing the "
        "disturbance penalty*, and it generalises across folds — but that is the expected "
        "signature of the **effort-bias confound (IDEA 009)**, not proof the rule is wrong. "
        "Presence reports come from where hunters can walk in, so any rule that down-weights "
        "accessible ground will lower the apparent hit-rate against effort-biased reports "
        "regardless of true reindeer behaviour. This Δ is therefore **not** licence to drop "
        "the rule; it is evidence to add an effort/accessibility covariate or an "
        "access-restricted background before the disturbance penalty can be judged fairly. "
        "A genuine ecological fix must raise the CV test AUC **without** simply exploiting "
        "this bias — that is the real Phase-6 bar.",
        "",
        "## Effort-matched background (IDEA 009 correction) — the primary gate",
        "The background is reweighted so \"available\" ground matches where reports "
        "actually occur (each field cell weighted by the empirical density of the "
        "reports' own distance-to-disturbance). This is an **evaluation-only** change; "
        "the scorer is untouched. It removes the accessibility confound so fixes can be "
        "judged on ecology, not on how they correlate with access.",
        "",
        f"- **Effort-matched CV baseline (shipped): {base_e['mean']:.3f} ± {base_e['std']:.3f}** "
        f"(vs {base['mean']:.3f} whole-field); folds beating chance "
        f"{base_e['fold_beats_chance']*100:.0f}%.",
        f"- Effort-matched select-then-evaluate: selection **{sel_e['select_mean']:.3f}** vs "
        f"shipped baseline **{sel_e['baseline_mean']:.3f}** "
        f"(Δ {sel_e['select_mean']-sel_e['baseline_mean']:+.3f}).",
        "```",
        "config                     selected%",
        *[f"{nm:26s} {fr*100:5.0f}" for nm, fr in zip(names, sel_e["select_frac"])],
        "```",
        f"- Whether removing the disturbance penalty still wins tells us if the confound "
        f"is neutralised: its selection share drops from "
        f"{sel['select_frac'][1]*100:.0f}% (whole-field) to "
        f"{sel_e['select_frac'][1]*100:.0f}% (effort-matched), and the selection Δ from "
        f"{sel['select_mean']-sel['baseline_mean']:+.3f} to "
        f"{sel_e['select_mean']-sel_e['baseline_mean']:+.3f}. From here, structural fixes "
        "(012–018) are judged against the **effort-matched** baseline.",
        "",
        "## How Phase-6 fixes use this harness",
        "Each structural fix (IDEAS 012–018) is added to `CANDIDATES` in "
        "`scripts/cv_validate.py` as a new variant (or becomes the new shipped default "
        "only after it clears the gate). A fix is **kept** iff it raises the "
        "select-then-evaluate CV test AUC over the shipped baseline with the delta "
        "larger than its std; otherwise it is reverted and the negative result logged. "
        "No constant is ever tuned against the full 37-report set.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"[whole-field ] CV baseline {base['mean']:.3f} ± {base['std']:.3f} "
          f"| folds>chance {base['fold_beats_chance']*100:.0f}% | n={n}")
    print(f"[whole-field ] select {sel['select_mean']:.3f} vs baseline "
          f"{sel['baseline_mean']:.3f} (Δ {sel['select_mean']-sel['baseline_mean']:+.3f}) "
          f"| disturb-off selected {sel['select_frac'][1]*100:.0f}%")
    print(f"[effort-match] CV baseline {base_e['mean']:.3f} ± {base_e['std']:.3f} "
          f"| folds>chance {base_e['fold_beats_chance']*100:.0f}%")
    print(f"[effort-match] select {sel_e['select_mean']:.3f} vs baseline "
          f"{sel_e['baseline_mean']:.3f} (Δ {sel_e['select_mean']-sel_e['baseline_mean']:+.3f}) "
          f"| disturb-off selected {sel_e['select_frac'][1]*100:.0f}%")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
