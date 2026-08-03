"""Phase 6: k-fold cross-validation harness for the scorer (IDEA 010).

Why this exists
---------------
The shipped scorer has **no parameters fitted to the sightings**, so a single
date-matched AUC over all 37 reports is already "out-of-sample". CV adds two things
the single number can't give, both needed before we trust any Phase-6 change:

1. **A stability estimate.** Repeated k-fold resampling of the per-report percentiles
   gives the mean AND spread of the headline metric, so we can say whether the
   at-/above-chance verdict is robust or an artifact of a few reports.

2. **An honest select-then-evaluate procedure.** When a change is *motivated by the
   validation set* (e.g. a threshold sweep, or "which structural variant helps"),
   picking the best variant on the same data it is judged on is the classic
   optimism trap. `cv_select_evaluate` selects the variant on the TRAIN folds and
   scores it on the held-out TEST fold, so the reported number reflects how the
   *selection procedure* generalises — not the best in-sample fit.

Everything here is pure numpy over a matrix of pre-computed per-report percentiles;
the expensive grid scoring lives in the caller (scripts/cv_validate.py).
"""
from __future__ import annotations

import numpy as np


def _folds(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """A random k-way partition of range(n) (fold sizes differ by at most 1)."""
    return np.array_split(rng.permutation(n), k)


def repeated_kfold_stats(percentiles, k: int = 5, repeats: int = 40,
                         seed: int = 0) -> dict:
    """Mean and spread of the test-fold mean percentile over repeated k-fold splits.

    For a fixed model each report's percentile is fold-independent, so this
    quantifies the *sampling* uncertainty of the headline AUC (how stable the
    verdict is), not a fit. Returns the pooled test-fold mean, the between-fold std,
    and the fraction of test folds that beat chance.
    """
    p = np.asarray(percentiles, float)
    n = len(p)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "fold_beats_chance": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    fold_means = []
    for _ in range(repeats):
        for f in _folds(n, k, rng):
            if len(f):
                fold_means.append(float(p[f].mean()))
    fm = np.asarray(fold_means)
    return {"mean": float(fm.mean()), "std": float(fm.std()),
            "fold_beats_chance": float((fm > 0.5).mean()), "n": n}


def cv_select_evaluate(pct_matrix, k: int = 5, repeats: int = 40,
                       seed: int = 0) -> dict:
    """Nested selection CV over candidate configs.

    pct_matrix: shape (n_configs, n_reports) — row c = per-report percentile of the
    observed cell under config c. Config 0 is treated as the shipped baseline.

    Per split: choose the config with the best *train*-fold mean percentile, then
    score that config on the *test* fold. Returns the CV test AUC of this selection
    procedure (mean + std), plus how often each config was selected and the shipped
    baseline's own CV test AUC for comparison.
    """
    P = np.asarray(pct_matrix, float)
    n_configs, n = P.shape
    rng = np.random.default_rng(seed)
    sel_test, base_test = [], []
    sel_counts = np.zeros(n_configs)
    for _ in range(repeats):
        folds = _folds(n, k, rng)
        for i in range(k):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(k) if j != i])
            if not len(test) or not len(train):
                continue
            best = int(np.argmax(P[:, train].mean(axis=1)))
            sel_counts[best] += 1
            sel_test.append(float(P[best, test].mean()))
            base_test.append(float(P[0, test].mean()))
    sel_test = np.asarray(sel_test)
    base_test = np.asarray(base_test)
    total = sel_counts.sum() or 1.0
    return {"select_mean": float(sel_test.mean()), "select_std": float(sel_test.std()),
            "baseline_mean": float(base_test.mean()), "baseline_std": float(base_test.std()),
            "select_frac": (sel_counts / total).tolist()}
