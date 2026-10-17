"""Walk-forward backtest of a retraining policy on a stream.

Each batch: the current model predicts, then (after `label_delay` batches) the true targets arrive.
A policy decides, batch by batch, whether to retrain; a retrain fits a fresh model on the most recent
`window` labelled batches. Accounting:
- error cost: mean absolute error per batch, summed;
- retrain cost: `retrain_cost` per retrain (data-scientist time + compute), in the same units.
The oracle knows when harmful drift started: it retrains once, as soon as 3 post-change batches are
labelled, on post-change data only. A reference point, not a strict bound (gradual drift needs more).

Policies: never | schedule (every `every` batches) | psi | ks | pred_psi | error (performance) | oracle
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from retrain import detectors

TRAIN_BATCHES = 20


def fit(X, y, seed=0):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
    return m.fit(X, y)


@dataclass
class Result:
    policy: str
    mae: list = field(default_factory=list)
    retrains: list = field(default_factory=list)   # batch indices where a retrain happened
    alarms: list = field(default_factory=list)     # batches where the policy's signal fired

    def cost(self, retrain_cost):
        return float(np.sum(self.mae) + retrain_cost * len(self.retrains))


def run(s, policy, label_delay=2, window=10, every=10, cooldown=10, seed=0):
    X, y = s.X, s.y
    Xtr, ytr = np.vstack(X[:TRAIN_BATCHES - 2]), np.concatenate(y[:TRAIN_BATCHES - 2])
    model = fit(Xtr, ytr, seed)
    ref_X = Xtr
    val_X, val_y = np.vstack(X[TRAIN_BATCHES - 2:TRAIN_BATCHES]), np.concatenate(y[TRAIN_BATCHES - 2:TRAIN_BATCHES])
    base_mae = float(np.mean(np.abs(model.predict(val_X) - val_y)))
    ref_pred = model.predict(ref_X)
    res = Result(policy)
    last_retrain = TRAIN_BATCHES
    for b in range(TRAIN_BATCHES, len(X)):
        pred = model.predict(X[b])
        res.mae.append(float(np.mean(np.abs(pred - y[b]))))
        labelled_upto = b - label_delay          # labels for batches <= this are known now
        fired = False
        if policy == "schedule":
            fired = (b - TRAIN_BATCHES) % every == every - 1
        elif policy == "psi":
            fired = detectors.feature_psi(ref_X, X[b])[1]
        elif policy == "ks":
            fired = detectors.feature_ks(ref_X, X[b])[1]
        elif policy == "pred_psi":
            fired = detectors.prediction_psi(ref_pred, pred)[1]
        elif policy == "error" and labelled_upto >= last_retrain + 2:
            recent = [res.mae[i - TRAIN_BATCHES] for i in range(max(last_retrain, labelled_upto - 2), labelled_upto + 1)]
            fired = detectors.error_ratio(recent, base_mae)[1]
        elif policy == "oracle":
            fired = s.harmful and b == s.onset + label_delay + 3   # 3 labelled post-change batches to train on
        if fired:
            res.alarms.append(b)
        if fired and b - last_retrain >= cooldown and labelled_upto - window + 1 >= 0:
            lo = max(0, labelled_upto - window + 1)
            if policy == "oracle":
                lo = s.onset   # the oracle knows when the change began: train on post-change data only
            Xr, yr = np.vstack(X[lo:labelled_upto + 1]), np.concatenate(y[lo:labelled_upto + 1])
            model = fit(Xr[:-2 * len(X[0])], yr[:-2 * len(X[0])], seed)
            vX, vy = Xr[-2 * len(X[0]):], yr[-2 * len(X[0]):]
            base_mae = float(np.mean(np.abs(model.predict(vX) - vy)))
            ref_X, ref_pred = Xr, model.predict(Xr)
            res.retrains.append(b)
            last_retrain = b
    return res


POLICIES = ("never", "schedule", "psi", "ks", "pred_psi", "error", "oracle")
