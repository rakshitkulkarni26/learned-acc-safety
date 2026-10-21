"""Ranking metrics without extra dependencies (AUROC, AUPRC, bootstrap CI)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) formula, ties averaged."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)  # average ranks for ties
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve, step-wise)."""
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0:
        return float("nan")
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    yt, st = y[order], s[order]
    tp = np.cumsum(yt)
    # evaluate only at the last index of each group of tied scores (one threshold per value)
    last = np.r_[np.flatnonzero(np.diff(st) != 0), len(st) - 1]
    precision = tp[last] / (last + 1)
    recall = tp[last] / yt.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_ci(
    metric, y: np.ndarray, s: np.ndarray, groups: np.ndarray, n_boot: int = 200, seed: int = 0
) -> tuple[float, float]:
    """95 % CI by resampling whole groups (events), so correlated samples stay together."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = np.split(np.argsort(inv, kind="mergesort"), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[g] for g in pick])
        v = metric(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
