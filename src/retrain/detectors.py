"""Drift signals. Each returns a *score*; a trigger fires when its score crosses a threshold.

Thresholds are calibrated (calibrate.py) so every trigger has the same false-alarm rate on streams
with no drift: otherwise "more sensitive" and "better" get confused. PSI's industry rule of thumb
(> 0.2) is kept as an extra, uncalibrated row for comparison.

Input drift (no labels needed, available immediately):
- psi:     Population Stability Index per feature, worst feature.
- ks:      Kolmogorov-Smirnov per feature; score = -log10(Bonferroni-corrected p).
- c2st:    classifier two-sample test: can a classifier tell reference rows from new rows?
           Score = cross-validated AUC (0.5 = indistinguishable). Sees joint changes, not just one
           feature at a time.
Output drift (no labels):
- pred_psi: PSI on the model's predictions.
Performance (needs labels, which arrive late):
- error:   mean absolute error of the last 3 labelled batches / error at training time.
- page_hinkley: cumulative evidence that the error has risen (Page-Hinkley statistic on batch error,
           in units of the training-time error). Reacts to small persistent rises the ratio misses.
"""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def psi(ref, cur, bins=10):
    """Bins from the reference quantiles; a small floor keeps empty bins finite."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_psi(Xref, Xcur, bins=10):
    return max(psi(Xref[:, j], Xcur[:, j], bins) for j in range(Xref.shape[1]))


def feature_ks(Xref, Xcur):
    p = min(ks_2samp(Xref[:, j], Xcur[:, j]).pvalue for j in range(Xref.shape[1])) * Xref.shape[1]
    return float(-np.log10(max(min(p, 1.0), 1e-300)))


def c2st(Xref, Xcur, seed=0, n=500):
    rng = np.random.default_rng(seed)
    a = Xref[rng.choice(len(Xref), min(n, len(Xref)), replace=False)]
    b = Xcur[rng.choice(len(Xcur), min(n, len(Xcur)), replace=False)]
    X = np.vstack([a, b])
    X = np.hstack([X, X ** 2])                        # lets a linear classifier see spread changes
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=2, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def error_ratio(recent_mae, baseline_mae):
    return float(np.mean(recent_mae) / baseline_mae)


def page_hinkley(mae_seq, baseline_mae, delta=0.05):
    """Current Page-Hinkley statistic: cumulative excess error now minus its lowest point so far,
    in baseline units. `mae_seq` = labelled batch errors since the last retrain."""
    if len(mae_seq) == 0:
        return 0.0
    c = np.cumsum(np.asarray(mae_seq) / baseline_mae - 1 - delta)
    return float(c[-1] - min(0.0, c.min()))
