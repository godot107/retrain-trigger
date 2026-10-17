"""Calibrate every trigger's threshold to the same false-alarm rate on drift-free streams.

Calibration streams use their own seeds (never reused for testing). The initial model is trained
and never retrained; each trigger's score is recorded every batch. Threshold = the score exceeded in
only `rate` of batches (default 1 in 300: about one false alarm per five 60-batch streams).

psi_error ("input drift flags, error confirms") gets a deliberately lower error bar, the 90th
percentile of the error score: it only matters after an input-drift flag, so it can afford to be
more sensitive.
"""
import json
from pathlib import Path

import numpy as np

from retrain import backtest, simulate

CAL_SEEDS = range(1000, 1050)
SCORED = ("psi", "ks", "c2st", "pred_psi", "error", "page_hinkley")


def scores_for(seed, label_delay=2):
    s = simulate.stream("none", seed)
    st = backtest.initial(s, seed)
    mae_hist, out = {}, {k: [] for k in SCORED}
    for b in range(backtest.TRAIN_BATCHES, len(s.X)):
        pred = st.model.predict(s.X[b])
        mae_hist[b] = float(np.mean(np.abs(pred - s.y[b])))
        for k in SCORED:
            out[k].append(backtest.score(k, st, s, b, pred, mae_hist, label_delay, seed))
    return out


def calibrate(rate=1 / 300, path=None, pool=None):
    mapper = pool.map if pool else map
    allsc = {k: [] for k in SCORED}
    for sc in mapper(scores_for, CAL_SEEDS):
        for k in SCORED:
            allsc[k] += sc[k]
    th = {k: float(np.quantile(allsc[k], 1 - rate)) for k in SCORED}
    th["psi_error"] = float(np.quantile(allsc["error"], 0.90))
    if path:
        Path(path).write_text(json.dumps({"rate_per_batch": rate, "thresholds": th,
                                          "n_scores": len(allsc["psi"])}, indent=2) + "\n")
    return th
