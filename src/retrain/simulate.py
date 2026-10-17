"""Synthetic data streams with drift injected at a known batch: the answer key for every trigger.

A stream is a list of batches (think: one per week). Each batch has 3 features and a target
y = f(x) + noise. Scenarios, all starting to change at batch ONSET:

- none:                nothing changes. Any alarm is a false alarm.
- covariate_harmless:  the inputs move, but to a region the model learned well. Inputs drift, the
                       model is fine: retraining is wasted money.
- covariate_harmful:   the inputs move into a region the model never saw, where f bends.
- concept_sudden:      f itself changes (the same inputs now mean a different target); the inputs
                       look exactly as before, so input-drift checks are blind to it.
- concept_gradual:     the same change phased in over 20 batches.
- concept_complex:     the relationship becomes much more intricate (strong interactions, faster
                       wiggles). Fresh data alone may not be enough: the model's tuned size and
                       learning settings, chosen for the simpler world, may need re-tuning.
"""
from dataclasses import dataclass

import numpy as np

SCENARIOS = ("none", "covariate_harmless", "covariate_harmful", "concept_sudden", "concept_gradual",
             "concept_complex")
ONSET = 40
N_BATCHES = 80
BATCH_SIZE = 500


@dataclass
class Stream:
    scenario: str
    X: list          # per batch: (BATCH_SIZE, 3)
    y: list          # per batch: (BATCH_SIZE,)
    onset: int
    harmful: bool    # does the drift actually hurt a model trained before ONSET?


def f(X, w):
    """Smooth in the training region (x0 within about +-2); bends sharply beyond x0 = 3."""
    return w[0] * X[:, 0] + w[1] * np.sin(X[:, 1]) + w[2] * X[:, 2] ** 2 + 4.0 * np.maximum(X[:, 0] - 3, 0) ** 2


def complex_term(X):
    return 3.0 * np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 2]) + 2.0 * X[:, 1] * X[:, 2]


def stream(scenario, seed=0):
    rng = np.random.default_rng(seed)
    w0, w1 = np.array([2.0, 1.5, 0.5]), np.array([-1.0, 1.5, 0.5])   # concept change flips x0's effect
    X, y = [], []
    for b in range(N_BATCHES):
        k = 0.0 if b < ONSET else 1.0
        if scenario == "concept_gradual":
            k = float(np.clip((b - ONSET) / 20, 0, 1))
        mean = np.zeros(3)
        cov = np.eye(3)
        if scenario == "covariate_harmless" and b >= ONSET:
            mean = np.array([0.0, 1.2, 0.0])     # x1 moves within the range seen in training; f is periodic in it
            cov = np.diag([1.0, 0.5, 1.0])
        if scenario == "covariate_harmful" and b >= ONSET:
            mean = np.array([2.5, 0.0, 0.0])     # x0 moves past 3, where f bends
        Xb = rng.multivariate_normal(mean, cov, BATCH_SIZE)
        w = (1 - k) * w0 + k * w1 if scenario in ("concept_sudden", "concept_gradual") else w0
        target = f(Xb, w)
        if scenario == "concept_complex" and b >= ONSET:
            target = target + complex_term(Xb)
        X.append(Xb)
        y.append(target + rng.normal(0, 0.5, BATCH_SIZE))
    return Stream(scenario, X, y, ONSET, harmful=scenario not in ("none", "covariate_harmless"))
