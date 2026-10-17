"""Real-data experiment: retraining triggers on day-ahead load forecasting, 12 grid operators.

Real data has no drift-free stretch to calibrate on and no known onset, so each trigger's threshold is
*chosen* on 2020-2021 (COVID, Winter Storm Uri) and *scored once* on 2022-2025 (Winter Storm Elliott,
ERCOT's load growth): the same train/test discipline applied to the trigger itself.

Cost units: one retrain costs as much as `k` weeks of the operator's normal forecast error (its
validation error at the initial training), k in RETRAIN_WEEKS. Weekly error = mean absolute error as
a fraction of the operator's 2019 mean load.

    python -m retrain.realdata          # ~10-20 min on 11 cores; writes results/real_runs.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retrain import backtest, eia  # noqa: E402

RES = Path(__file__).resolve().parents[2] / "results"
GRID = {
    "psi": [0.05, 0.1, 0.2, 0.4], "ks": [2, 5, 10, 20], "c2st": [0.55, 0.6, 0.7, 0.8],
    "pred_psi": [0.05, 0.1, 0.2, 0.4], "error": [1.1, 1.25, 1.5, 2.0],
    "page_hinkley": [0.5, 1, 2, 4], "error_cp": [1.1, 1.25, 1.5, 2.0], "page_hinkley_cp": [0.5, 1, 2, 4],
    "schedule": [13, 26, 52], "never": [None],
}
# Second pass (2026-10-17): the first grid was too narrow. Seasonal load features keep input-drift
# scores high all year, so even the least sensitive values fired every cooldown; the error triggers
# picked their grid maximum. EXTRA extends each grid; results are appended to the same file.
EXTRA = {
    "psi": [0.8, 1.5, 3.0], "ks": [50, 100, 300], "c2st": [0.9, 0.95, 0.99], "pred_psi": [0.8, 1.5, 3.0],
    "error": [2.5, 3.0], "page_hinkley": [8, 16], "error_cp": [2.5, 3.0], "page_hinkley_cp": [8, 16],
    "schedule": [104],
    "psi_seasonal": [0.5, 1.0, 2.0, 4.0, 8.0],   # 4-week vs same 4 weeks last year, 5 bins
}
# Third pass: change point with a full-year window, post-change rows weighted (backtest.CP_WEIGHT).
CPW = {"error_cpw": [1.1, 1.25, 1.5, 2.0, 2.5, 3.0], "page_hinkley_cpw": [0.5, 1, 2, 4, 8, 16]}
RETRAIN_WEEKS = (1, 4, 16)
SETTINGS = {"label_delay": 1, "window": 52, "cooldown": 4}
TUNE_YEARS, TEST_YEARS = (2020, 2021), (2022, 2023, 2024, 2025)


WX_GRID = {  # weather model: the triggers that mattered, full grids
    "never": [None], "schedule": [13, 26, 52, 104],
    "error": [1.1, 1.25, 1.5, 2.0, 2.5, 3.0], "page_hinkley": [0.5, 1, 2, 4, 8, 16],
    "error_cpw": [1.1, 1.25, 1.5, 2.0, 2.5, 3.0], "page_hinkley_cpw": [0.5, 1, 2, 4, 8, 16],
    "c2st": [0.6, 0.7, 0.8, 0.9, 0.95, 0.99], "psi_seasonal": [0.5, 1.0, 2.0, 4.0, 8.0],
}


def run_ba(ba, grid=None, weather=False):
    grid = grid or GRID
    raw = eia.load_raw()
    s = eia.stream(ba, raw, weather=weather)
    init = backtest.initial(s)
    weeks = [w.isoformat() for w in s.weeks[s.train_batches:]]
    out = []
    for trig, values in grid.items():
        for v in values:
            th = {k: 1e9 for k in ("psi", "ks", "c2st", "pred_psi", "error", "page_hinkley", "psi_error",
                                   "psi_seasonal")}
            extra = {}
            if trig == "schedule":
                extra["every"] = v
            elif trig != "never":
                th[trig.removesuffix("_cpw").removesuffix("_cp")] = v
            r = backtest.run(s, "schedule" if trig == "schedule" else trig, th, init=init, **SETTINGS, **extra)
            out.append({"ba": ba, "trigger": trig, "threshold": v, "weeks": weeks, "mae": r.mae,
                        "retrains": [s.weeks[b].isoformat() for b in r.retrains],
                        "normal_error": init.val_mae, "params": init.params})
    return out


def _extra(ba):
    return run_ba(ba, EXTRA)


def _wx(ba):
    return run_ba(ba, WX_GRID, weather=True)


def _cpw(ba):
    return run_ba(ba, CPW)


def _seasonal(ba):
    return run_ba(ba, {"psi_seasonal": EXTRA["psi_seasonal"]})


def main(argv=None):
    """No args: the first grid (overwrites). `--extra`: the extended grid, appended."""
    argv = sys.argv[1:] if argv is None else argv
    RES.mkdir(exist_ok=True)
    if argv[:1] == ["--weather"]:     # separate results file
        with ProcessPoolExecutor(max_workers=min(len(eia.BAS), max(1, (os.cpu_count() or 2) - 1))) as pool:
            rows = [r for rs in pool.map(_wx, eia.BAS) for r in rs]
        (RES / "real_runs_weather.json").write_text(json.dumps(rows))
        print(len(rows), "weather runs")
        return
    fn = {"--extra": _extra, "--seasonal": _seasonal, "--cpw": _cpw}.get(argv[0] if argv else "", run_ba)
    with ProcessPoolExecutor(max_workers=min(len(eia.BAS), max(1, (os.cpu_count() or 2) - 1))) as pool:
        rows = [r for rs in pool.map(fn, eia.BAS) for r in rs]
    if fn is not run_ba:
        rows = json.loads((RES / "real_runs.json").read_text()) + rows
    (RES / "real_runs.json").write_text(json.dumps(rows))
    print(len(rows), "runs")


def table(rows=None):
    """Per (ba, trigger, threshold): error and retrains in the tuning and test periods."""
    rows = rows or json.loads((RES / "real_runs.json").read_text())
    out = []
    for r in rows:
        years = np.array([int(w[:4]) for w in r["weeks"]])
        ry = np.array([int(w[:4]) for w in r["retrains"]]) if r["retrains"] else np.array([], int)
        mae = np.asarray(r["mae"])
        for period, yrs in (("tune", TUNE_YEARS), ("test", TEST_YEARS)):
            sel = np.isin(years, yrs)
            out.append({"ba": r["ba"], "trigger": r["trigger"], "threshold": r["threshold"], "period": period,
                        "error_sum": float(mae[sel].sum()), "error_mean": float(mae[sel].mean()),
                        "retrains": int(np.isin(ry, yrs).sum()), "normal_error": r["normal_error"]})
    return pd.DataFrame(out)


def choose_and_score(t, k):
    """For each trigger: the threshold with the lowest total tuning-period cost (all operators),
    then that threshold's test-period cost. Cost per operator = error + k x normal_error x retrains."""
    t = t.assign(cost=t.error_sum + k * t.normal_error * t.retrains)
    tot = t.groupby(["trigger", "threshold", "period"], dropna=False).cost.sum().unstack("period").reset_index()
    best = tot.loc[tot.groupby("trigger").tune.idxmin()]
    return best.sort_values("test")[["trigger", "threshold", "tune", "test"]]


if __name__ == "__main__":
    sys.exit(main())


def choose_per_operator(t, k):
    """Each operator gets its own threshold (lowest 2020-21 cost for that operator), scored on 2022-25.
    Returns total test cost per trigger, to compare with one shared threshold (choose_and_score)."""
    t = t.assign(cost=t.error_sum + k * t.normal_error * t.retrains)
    w = t.pivot_table(index=["trigger", "ba", "threshold"], columns="period", values="cost", dropna=False).reset_index()
    w = w.dropna(subset=["tune"])
    best = w.loc[w.groupby(["trigger", "ba"]).tune.idxmin()]
    return best.groupby("trigger").agg(tune=("tune", "sum"), test=("test", "sum")).sort_values("test")
