"""Walk-forward backtest of a retraining trigger on a stream.

Each batch (think: one week): the current model predicts; the true targets arrive `label_delay`
batches later. The trigger computes its drift score; if the score crosses its threshold (and the
cooldown since the last retrain has passed), the model is retrained on the most recent `window`
labelled batches using one of three actions (models.py). Accounting:
- error cost: the batch's mean absolute error, summed over batches;
- retrain cost: a price per retrain (data-scientist time + compute). Applied afterwards, so one run
  gives the cost for every retrain price (triggers don't know the price).

Triggers: never | schedule | psi | psi_rule (PSI > 0.2, uncalibrated) | ks | c2st | pred_psi |
          error | page_hinkley | psi_error (an input-drift flag that error confirms) | oracle
The oracle knows when harmful drift began: it retrains once, as soon as 3 post-change batches are
labelled, on post-change data only. A reference point, not a strict bound.
"""
from dataclasses import dataclass, field

import numpy as np

from retrain import detectors, models

TRAIN_BATCHES = 20
VAL_BATCHES = 2
TRIGGERS = ("never", "schedule", "psi", "psi_rule", "ks", "c2st", "pred_psi", "error", "page_hinkley",
            "psi_error", "error_cp", "page_hinkley_cp", "error_cpw", "page_hinkley_cpw", "oracle")
# *_cp: same alarm as the base trigger, but the retrain uses only data since the estimated change point
# (where the cumulative excess error was lowest), waiting until enough post-change data is labelled.
# *_cpw: keeps the full window (so a seasonal cycle survives) but weights post-change rows CP_WEIGHT x.
CP_WEIGHT = 4.0
LABEL_FREE = {"psi", "psi_rule", "ks", "c2st", "pred_psi"}


@dataclass
class Result:
    trigger: str
    action: str
    mae: list = field(default_factory=list)
    retrains: list = field(default_factory=list)
    scores: list = field(default_factory=list)

    def cost(self, retrain_cost):
        return float(np.sum(self.mae) + retrain_cost * len(self.retrains))


def _split(X, y, lo, hi, val=VAL_BATCHES):
    """Batches lo..hi-1 -> train on all but the last `val`, validate on those."""
    Xt, yt = np.vstack(X[lo:hi - val]), np.concatenate(y[lo:hi - val])
    Xv, yv = np.vstack(X[hi - val:hi]), np.concatenate(y[hi - val:hi])
    return Xt, yt, Xv, yv


class State:
    """What a trigger knows about the current model."""

    def __init__(self, model, params, ref_X, val_mae, since):
        self.model, self.params, self.ref_X, self.val_mae, self.since = model, params, ref_X, val_mae, since
        self.ref_pred = model.predict(ref_X)


def initial(s, seed=0):
    tb, vb = _sizes(s)
    Xt, yt, Xv, yv = _split(s.X, s.y, 0, tb, vb)
    params, model, val_mae = models.tune(Xt, yt, Xv, yv, seed)
    st = State(model, params, Xt, val_mae, tb)
    st.drift_cols = getattr(s, "drift_cols", None)
    if st.drift_cols is not None:
        st.ref_X = st.ref_X[:, st.drift_cols]
    return st


def _sizes(s):
    """Streams may override the defaults (the real-data streams use a year of weekly batches)."""
    return getattr(s, "train_batches", TRAIN_BATCHES), getattr(s, "val_batches", VAL_BATCHES)


def _cols(s, b):
    dc = getattr(s, "drift_cols", None)
    return s.X[b] if dc is None else s.X[b][:, dc]


def score(trigger, st, s, b, pred, mae_hist, label_delay, seed=0):
    """The trigger's drift score at batch b (higher = more drift). mae_hist[i] = error of batch i."""
    labelled = [i for i in range(st.since, b - label_delay + 1)]
    if trigger in ("psi", "psi_rule"):
        return detectors.feature_psi(st.ref_X, _cols(s, b))
    if trigger == "psi_seasonal":
        # Compare this week with the same weeks one season earlier (inputs are known without labels),
        # so ordinary seasonal swings don't count as drift. Needs a stream with `season` batches.
        # Last 4 batches vs the same 4 batches one season earlier, 5 bins: a single week holds only 7
        # distinct values of each day-level feature, too few to fill 10 bins (PSI then reads 4-7 on
        # ordinary weeks, all of it from empty bins).
        k = getattr(s, "season", None)
        if k is None or b - k - 3 < 0:
            return 0.0
        cur = np.vstack([_cols(s, i) for i in range(b - 3, b + 1)])
        ref = np.vstack([_cols(s, i) for i in range(b - k - 3, b - k + 1)])
        return detectors.feature_psi(ref, cur, bins=5)
    if trigger == "ks":
        return detectors.feature_ks(st.ref_X, _cols(s, b))
    if trigger == "c2st":
        return detectors.c2st(st.ref_X, _cols(s, b), seed=seed + b)
    if trigger == "pred_psi":
        return detectors.psi(st.ref_pred, pred)
    if trigger in ("error", "psi_error"):
        if len(labelled) < 3:
            return 0.0
        return detectors.error_ratio([mae_hist[i] for i in labelled[-3:]], st.val_mae)
    if trigger == "page_hinkley":
        return detectors.page_hinkley([mae_hist[i] for i in labelled], st.val_mae)
    return 0.0


def change_point(mae_hist, st, hi, baseline, delta=0.05):
    """Estimated first batch of the change: one after the lowest point of the cumulative excess error
    over the labelled batches since the last retrain (the Page-Hinkley change-point estimate)."""
    idx = list(range(st.since, hi))
    if not idx:
        return st.since
    c = np.cumsum([mae_hist[i] / baseline - 1 - delta for i in idx])
    k = int(np.argmin(np.r_[0.0, c]))           # 0 = before the first labelled batch
    return idx[0] + k


def run(s, trigger, thresholds, action="refit", label_delay=2, window=10, every=10, cooldown=6, seed=0,
        init=None):
    """thresholds: {trigger: threshold} from calibrate.py (psi_error uses both 'psi' and 'error')."""
    st = init if init is not None else initial(s, seed)
    if init is not None:   # a shared initial state is mutated by warm starts: copy the model
        import copy
        st = copy.deepcopy(init)
    res = Result(trigger, action)
    mae_hist = {}
    recent_psi = []
    tb, vb = _sizes(s)
    for b in range(tb, len(s.X)):
        pred = st.model.predict(s.X[b])
        mae_hist[b] = float(np.mean(np.abs(pred - s.y[b])))
        res.mae.append(mae_hist[b])
        labelled_upto = b - label_delay
        if trigger == "never":
            fired = False
        elif trigger == "schedule":
            fired = (b - st.since) >= every
        elif trigger == "oracle":
            fired = s.harmful and b == s.onset + label_delay + 3
        elif trigger == "psi_error":
            recent_psi.append(detectors.feature_psi(st.ref_X, _cols(s, b)) > thresholds["psi"])
            err = score("error", st, s, b, pred, mae_hist, label_delay, seed)
            fired = any(recent_psi[-5:]) and err > thresholds["psi_error"]
            res.scores.append(err)
        else:
            base = trigger.removesuffix("_cpw").removesuffix("_cp")
            sc = score(base, st, s, b, pred, mae_hist, label_delay, seed)
            res.scores.append(sc)
            thr = 0.2 if trigger == "psi_rule" else thresholds[base]
            fired = sc > thr
        if not fired or b - st.since < cooldown and trigger != "oracle":
            continue
        hi = labelled_upto + 1
        lo = s.onset if trigger == "oracle" else max(0, hi - window)
        weights = None
        if trigger.endswith("_cp"):
            lo = max(lo, change_point(mae_hist, st, hi, st.val_mae))
        if hi - lo < vb + 1:
            continue
        Xt, yt, Xv, yv = _split(s.X, s.y, lo, hi, vb)
        if trigger.endswith("_cpw"):
            cp = change_point(mae_hist, st, hi, st.val_mae)
            if hi - cp < vb + 1:          # wait until some post-change data is in the training part
                continue
            sizes = [len(s.y[i]) for i in range(lo, hi - vb)]
            weights = np.concatenate([np.full(n, CP_WEIGHT if i >= cp else 1.0)
                                      for i, n in zip(range(lo, hi - vb), sizes)])
        model, params, val_mae = models.retrain(action, st.model, st.params, Xt, yt, Xv, yv, seed,
                                                sample_weight=weights)
        dc = getattr(s, "drift_cols", None)
        st = State(model, params, np.vstack(s.X[lo:hi]), val_mae, b)
        if dc is not None:
            st.ref_X = st.ref_X[:, dc]
        recent_psi = []
        res.retrains.append(b)
    return res
