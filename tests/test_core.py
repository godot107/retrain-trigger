"""Known-answer checks for the detectors and the backtest."""
import numpy as np

from retrain import backtest, detectors, simulate


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert detectors.psi(a, b) < 0.02
    assert detectors.psi(a, rng.normal(1.5, 1, 5000)) > 0.5


def test_scenarios_do_what_they_say():
    none, harmless, concept = (simulate.stream(s) for s in ("none", "covariate_harmless", "concept_sudden"))
    on = simulate.ONSET
    # concept drift: inputs unchanged, targets changed
    assert detectors.feature_psi(concept.X[on - 1], concept.X[on + 1])[0] < 0.1
    # harmless covariate drift: inputs changed
    assert detectors.feature_psi(harmless.X[on - 1], harmless.X[on + 1])[1]
    assert not none.harmful and not harmless.harmful and concept.harmful


def test_never_retrains_and_error_trigger_ignores_harmless_drift():
    s = simulate.stream("covariate_harmless")
    assert backtest.run(s, "never").retrains == []
    assert backtest.run(s, "error").retrains == []
    assert len(backtest.run(s, "psi").retrains) >= 1      # the input alarm pays for a useless retrain


def test_error_trigger_catches_concept_drift_and_beats_never():
    s = simulate.stream("concept_sudden")
    err, never = backtest.run(s, "error"), backtest.run(s, "never")
    assert err.retrains and err.retrains[0] >= s.onset    # no retrain before the change
    assert err.cost(5) < never.cost(5)
