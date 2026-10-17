"""Known-answer checks for the detectors and the backtest."""
import numpy as np

from retrain import backtest, detectors, simulate

# Fixed thresholds for tests (results/thresholds.json holds the calibrated ones).
TH = {"psi": 0.06, "ks": 1.6, "c2st": 0.56, "pred_psi": 0.06, "error": 1.12, "page_hinkley": 0.15,
      "psi_error": 1.05}


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert detectors.psi(a, b) < 0.02
    assert detectors.psi(a, rng.normal(1.5, 1, 5000)) > 0.5


def test_c2st_sees_a_joint_change_that_single_features_miss():
    rng = np.random.default_rng(0)
    ref = rng.multivariate_normal([0, 0], [[1, 0.0], [0.0, 1]], 2000)
    cur = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 500)   # same marginals, new correlation
    assert detectors.feature_psi(ref, cur) < 0.1
    assert detectors.c2st(ref, cur) < 0.6     # a linear model on x, x^2 can't see correlation either...
    cur2 = rng.multivariate_normal([0, 0], [[3, 0], [0, 1]], 500)       # ...but does see a spread change
    assert detectors.c2st(ref, cur2) > 0.6


def test_page_hinkley_rises_with_persistent_excess_error_only():
    assert detectors.page_hinkley([1.0] * 20, 1.0) == 0.0
    assert detectors.page_hinkley([1.0] * 10 + [1.3] * 10, 1.0) > 2.0


def test_scenarios_do_what_they_say():
    none, harmless, concept = (simulate.stream(s) for s in ("none", "covariate_harmless", "concept_sudden"))
    on = simulate.ONSET
    assert detectors.feature_psi(concept.X[on - 1], concept.X[on + 1]) < 0.1   # concept: inputs unchanged
    assert detectors.feature_psi(harmless.X[on - 1], harmless.X[on + 1]) > 0.2
    assert not none.harmful and not harmless.harmful and concept.harmful


def test_never_retrains_and_error_trigger_ignores_harmless_drift():
    s = simulate.stream("covariate_harmless")
    init = backtest.initial(s)
    assert backtest.run(s, "never", TH, init=init).retrains == []
    assert backtest.run(s, "error", TH, init=init).retrains == []
    assert len(backtest.run(s, "psi_rule", TH, init=init).retrains) >= 1   # pays for a useless retrain


def test_error_trigger_catches_concept_drift_and_beats_never():
    s = simulate.stream("concept_sudden")
    init = backtest.initial(s)
    err, never = backtest.run(s, "error", TH, init=init), backtest.run(s, "never", TH, init=init)
    assert err.retrains and err.retrains[0] >= s.onset
    assert err.cost(5) < never.cost(5)


def test_shared_initial_model_is_not_mutated_by_warm_start():
    s = simulate.stream("concept_sudden")
    init = backtest.initial(s)
    n_before = init.model.n_iter_
    backtest.run(s, "oracle", TH, action="warm", init=init)
    assert init.model.n_iter_ == n_before
