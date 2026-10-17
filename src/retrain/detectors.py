"""Drift signals. Each takes the reference batch(es) the model was trained on and a new batch, and
returns a score plus a fired/not-fired decision.

Input drift (needs no labels, available immediately):
- psi:  Population Stability Index per feature, worst feature reported. Industry rule of thumb:
        > 0.2 = significant shift.
- ks:   two-sample Kolmogorov-Smirnov test per feature, Bonferroni-corrected p < 0.01.
Output drift (no labels):
- pred_psi: PSI on the model's predictions.
Performance (needs labels, which arrive late in real life):
- error_ratio: rolling mean absolute error over the last `window` labelled batches divided by the
  error on held-out data at training time; fires above `ratio`.
"""
import numpy as np
from scipy.stats import ks_2samp


def psi(ref, cur, bins=10):
    """Bins from the reference quantiles; a small floor keeps empty bins finite."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_psi(Xref, Xcur, threshold=0.2):
    s = max(psi(Xref[:, j], Xcur[:, j]) for j in range(Xref.shape[1]))
    return s, s > threshold


def feature_ks(Xref, Xcur, alpha=0.01):
    p = min(ks_2samp(Xref[:, j], Xcur[:, j]).pvalue for j in range(Xref.shape[1])) * Xref.shape[1]
    return float(p), p < alpha


def prediction_psi(pref, pcur, threshold=0.2):
    s = psi(pref, pcur)
    return s, s > threshold


def error_ratio(recent_mae, baseline_mae, ratio=1.25):
    r = float(np.mean(recent_mae) / baseline_mae)
    return r, r > ratio
