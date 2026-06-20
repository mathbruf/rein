"""Phase 5: metrics for testing the scorer against held-out sightings.

The scorer's weights were expert-tuned with the hunter and were NEVER fitted to the
sightings, so every geocoded observation is a legitimate held-out test point. We use
a presence-vs-background design restricted to the field (the huntable terrain that is
also the "available" area), with date-matched comparisons so day-to-day differences
in the overall score scale don't bias the result.

Headline metric: date-matched percentile rank of each observed cell within that day's
field-wide score distribution (== a date-matched AUC). 0.5 = chance, 1.0 = the
observation always sits in the single highest-scored cell. Also: top-quantile lift,
a pooled continuous Boyce index, and a permutation null.
"""
from __future__ import annotations

import numpy as np


def percentile_rank(value: float, background: np.ndarray) -> float:
    """Fraction of background scored below `value` (ties count half). 0..1."""
    bg = np.asarray(background, float)
    return float((bg < value).mean() + 0.5 * (bg == value).mean())


def date_matched_percentiles(obs_score: np.ndarray, obs_bg_scores: list[np.ndarray]) -> np.ndarray:
    """Per-observation percentile of its score within its own date's background."""
    return np.array([percentile_rank(s, bg) for s, bg in zip(obs_score, obs_bg_scores)])


def top_quantile_lift(percentiles: np.ndarray, q: float) -> tuple[float, float]:
    """Fraction of observations in the top-q of their date's field, and the lift vs q."""
    frac = float((percentiles >= 1.0 - q).mean())
    return frac, frac / q


def continuous_boyce(pres_scores: np.ndarray, bg_scores: np.ndarray,
                     nbins: int = 10) -> float:
    """Continuous Boyce index: Spearman corr of predicted/expected vs score bin.

    pres_scores/bg_scores pooled across dates (caveat: mixes day scales). +1 good,
    0 ~ random, negative = worse than random.
    """
    pres = np.asarray(pres_scores, float)
    bg = np.asarray(bg_scores, float)
    lo, hi = bg.min(), bg.max()
    if hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, nbins + 1)
    mids, pe = [], []
    for i in range(nbins):
        a, b = edges[i], edges[i + 1]
        in_bin = (lambda x: (x >= a) & (x < b)) if i < nbins - 1 else (lambda x: (x >= a) & (x <= b))
        exp = in_bin(bg).mean()
        if exp == 0:
            continue
        pred = in_bin(pres).mean()
        mids.append((a + b) / 2)
        pe.append(pred / exp)
    if len(pe) < 3:
        return float("nan")
    return _spearman(np.array(mids), np.array(pe))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def permutation_null(obs_bg_scores: list[np.ndarray], n_iter: int = 2000,
                     seed: int = 0) -> np.ndarray:
    """Null distribution of the mean date-matched percentile when each observation is
    placed at a RANDOM field cell on its date. Returns n_iter mean-percentile draws."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    n = len(obs_bg_scores)
    for k in range(n_iter):
        ps = np.empty(n)
        for j, bg in enumerate(obs_bg_scores):
            v = bg[rng.integers(len(bg))]
            ps[j] = percentile_rank(v, bg)
        means[k] = ps.mean()
    return means
