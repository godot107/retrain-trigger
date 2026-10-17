"""The Lambda's pure-Python monitor must agree with the offline detectors."""
import time

import numpy as np

from retrain import detectors
from retrain.aws import handler, monitor_core as core
from retrain.aws.replay import LocalStore


def test_page_hinkley_and_error_ratio_match_offline_exactly():
    rng = np.random.default_rng(0)
    seq = list(rng.uniform(0.8, 1.6, 40))
    assert abs(core.page_hinkley(seq, 1.0) - detectors.page_hinkley(seq, 1.0)) < 1e-12
    assert abs(core.error_ratio(seq[-3:], 0.9) - detectors.error_ratio(seq[-3:], 0.9)) < 1e-12


def test_change_point_finds_the_rise():
    weeks = [f"w{i:02d}" for i in range(20)]
    seq = [1.0] * 12 + [1.5] * 8
    assert core.change_point(weeks, seq, 1.0) == "w12"


def test_auc_by_hand():
    assert core.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert core.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_c2st_close_to_sklearn_version_and_fast_enough():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, (600, 4))
    for shift in (0.0, 0.5):
        cur = rng.normal(shift, 1, (300, 4))
        t0 = time.perf_counter()
        a = core.c2st(ref.tolist(), cur.tolist(), n=300)
        assert time.perf_counter() - t0 < 30
        b = detectors.c2st(ref, cur, n=300)
        assert abs(a - b) < 0.06, (shift, a, b)


def test_handler_resets_state_after_a_retrain_and_pages(tmp_path):
    store, sent = LocalStore(tmp_path), []
    m = "models/X/"
    store.put(m + "config.json", {"baseline_mae": 1.0, "trigger": "page_hinkley", "threshold": 1.0,
                                  "trained": "2020-01-00"})
    for i in range(1, 9):
        w = f"2020-01-{i:02d}"
        store.put(m + f"weeks/{w}.json", {"pred": [0.0], "actual": [1.0 if i < 4 else 2.0]})
        r = handler.run(store, "X", publish=lambda s, b: sent.append(s))
    assert r["fired"] and sent and "page_hinkley" in sent[0]
    cfg = store.get(m + "config.json")
    cfg["trained"] = "2020-01-08"             # the data scientist retrained
    store.put(m + "config.json", cfg)
    store.put(m + "weeks/2020-01-09.json", {"pred": [0.0], "actual": [1.0]})
    r = handler.run(store, "X")
    assert r["labelled_weeks"] == 1 and not r["fired"]
