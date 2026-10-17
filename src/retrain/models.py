"""The model and the three ways to retrain it.

Model: gradient-boosted trees (scikit-learn HistGradientBoostingRegressor). Its hyperparameters
(learning rate, tree size, how many trees, regularisation) are tuned once at the start by a small
random search on a validation split, as a careful data scientist would.

Retrain actions:
- refit:  new model, same hyperparameters, trained on recent data.            ("new weights")
- retune: new random search on recent data, then fit.                         ("reset weights and hyperparameters")
- warm:   keep the old trees and add more, fitted to recent data.             ("fine-tune")
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

SEARCH_SPACE = {
    "learning_rate": [0.03, 0.1, 0.3],
    "max_leaf_nodes": [4, 8, 15, 31],
    "max_depth": [2, 3, None],
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0.0, 1.0],
    "max_iter": [100, 200, 400],
}
N_CANDIDATES = 12


def make(params, seed=0, **extra):
    return HistGradientBoostingRegressor(random_state=seed, **params, **extra)


def tune(X, y, Xval, yval, seed=0):
    """Random search; returns (best params, fitted model, validation MAE)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(N_CANDIDATES):
        params = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        m = make(params, seed).fit(X, y)
        mae = float(np.mean(np.abs(m.predict(Xval) - yval)))
        if best is None or mae < best[2]:
            best = (params, m, mae)
    return best


def retrain(action, old_model, params, X, y, Xval, yval, seed=0):
    """Returns (model, params, validation MAE) after the chosen action."""
    if action == "retune":
        params, m, mae = tune(X, y, Xval, yval, seed)
        return m, params, mae
    if action == "warm":
        m = old_model
        m.set_params(warm_start=True, max_iter=m.max_iter + 100)
        m.fit(X, y)
    else:  # refit
        m = make(params, seed).fit(X, y)
    return m, params, float(np.mean(np.abs(m.predict(Xval) - yval)))
