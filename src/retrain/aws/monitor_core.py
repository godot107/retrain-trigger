"""Retraining monitor logic in pure Python (no numpy), so the Lambda needs no layers.

Mirrors retrain.detectors / retrain.backtest exactly for the two performance triggers, and closely for the
classifier test (a small logistic regression trained by gradient descent instead of scikit-learn's):

- error_ratio:  mean absolute error of the last 3 labelled weeks / the model's validation error.
- page_hinkley: cumulative excess error since the model was trained, minus its lowest point so far,
                in units of the validation error; the change-point estimate is where that low point was.
- c2st:         can a classifier tell this week's inputs from the training sample? Cross-validated AUC.

State that must survive between weekly runs (the labelled errors since training) is a plain dict, stored
by the handler as JSON next to the data.
"""
import math
import random

PH_DELTA = 0.05


def mae(pred, actual):
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(actual)


def error_ratio(recent_mae, baseline_mae):
    return sum(recent_mae) / len(recent_mae) / baseline_mae


def page_hinkley(mae_seq, baseline_mae, delta=PH_DELTA):
    """Same definition as detectors.page_hinkley: c[-1] - min(0, min(c))."""
    if not mae_seq:
        return 0.0
    c, lowest = 0.0, 0.0
    for m in mae_seq:
        c += m / baseline_mae - 1 - delta
        lowest = min(lowest, c)
    return c - lowest


def change_point(weeks, mae_seq, baseline_mae, delta=PH_DELTA):
    """First week of the change: the week after the cumulative excess error was lowest."""
    c, lowest, k = 0.0, 0.0, 0
    for i, m in enumerate(mae_seq):
        c += m / baseline_mae - 1 - delta
        if c < lowest:
            lowest, k = c, i + 1
    return weeks[k] if k < len(weeks) else None


# ---------- classifier two-sample test ----------

def _standardise(rows):
    cols = list(zip(*rows))
    mu = [sum(c) / len(c) for c in cols]
    sd = [max(1e-9, math.sqrt(sum((x - m) ** 2 for x in c) / len(c))) for c, m in zip(cols, mu)]
    return [[(x - m) / s for x, m, s in zip(r, mu, sd)] for r in rows]


def _fit_logistic(X, y, epochs=200, lr=0.1, l2=1e-3):
    w, b = [0.0] * len(X[0]), 0.0
    n = len(X)
    for _ in range(epochs):
        gw, gb = [0.0] * len(w), 0.0
        for xi, yi in zip(X, y):
            z = b + sum(a * c for a, c in zip(w, xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - yi
            gb += e
            for j, v in enumerate(xi):
                gw[j] += e * v
        w = [wj - lr * (g / n + l2 * wj) for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def auc(scores, labels):
    """Probability a random positive outranks a random negative (ties count half)."""
    pairs = sorted(zip(scores, labels))
    rank_sum, i, n = 0.0, 0, len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    pos = sum(labels)
    neg = n - pos
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def c2st(ref_rows, cur_rows, seed=0, n=500):
    """Cross-validated (2-fold) AUC of a logistic regression on [x, x^2], like detectors.c2st."""
    rng = random.Random(seed)
    a = rng.sample(ref_rows, min(n, len(ref_rows)))
    b = rng.sample(cur_rows, min(n, len(cur_rows)))
    X = _standardise([r + [v * v for v in r] for r in a + b])
    y = [0] * len(a) + [1] * len(b)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    folds = [idx[0::2], idx[1::2]]
    scores = [0.0] * len(X)
    for k in (0, 1):
        train, test = folds[1 - k], folds[k]
        w, bias = _fit_logistic([X[i] for i in train], [y[i] for i in train])
        for i in test:
            scores[i] = bias + sum(p * q for p, q in zip(w, X[i]))
    return auc(scores, y)


def evaluate(state, week, pred, actual=None, labelled=None, inputs=None, reference=None, cfg=None):
    """One weekly step. `labelled` = (week, pred, actual) newly labelled, if any. Returns (state, report).
    cfg: trigger ("page_hinkley" | "error" | "c2st"), threshold, baseline_mae, min_weeks_between."""
    cfg = cfg or {}
    state = dict(state or {"weeks": [], "mae": [], "last_page": None})
    if labelled is not None:
        lw, lp, la = labelled
        if lw not in state["weeks"]:
            state["weeks"] = state["weeks"] + [lw]
            state["mae"] = state["mae"] + [mae(lp, la)]
    base, trig, thr = cfg["baseline_mae"], cfg["trigger"], cfg["threshold"]
    if trig == "page_hinkley":
        score = page_hinkley(state["mae"], base)
    elif trig == "error":
        score = error_ratio(state["mae"][-3:], base) if len(state["mae"]) >= 3 else 0.0
    else:
        score = c2st(reference, inputs, seed=hash(week) % 10_000) if inputs and reference else 0.5
    fired = score > thr
    report = {"week": week, "trigger": trig, "score": score, "threshold": thr, "fired": fired,
              "labelled_weeks": len(state["mae"])}
    if fired:
        report["change_point"] = change_point(state["weeks"], state["mae"], base)
        report["recommendation"] = ("retrain on recent data (a full seasonal cycle); "
                                    "re-tune hyperparameters if the relationship looks more complex")
    return state, report
