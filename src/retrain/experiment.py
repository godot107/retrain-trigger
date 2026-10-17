"""Run every trigger x scenario x seed (and the retrain-action and label-delay variants) in parallel.

    python -m retrain.experiment            # calibrate, then run; writes results/synthetic_runs.csv

Each task = one (scenario, seed): the initial tuned model is shared by every trigger in the task.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one core per process: no oversubscription

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retrain import backtest, calibrate, simulate  # noqa: E402

RES = Path(__file__).resolve().parents[2] / "results"
TEST_SEEDS = range(30)
THRESHOLDS = None


def _row(s, r, **extra):
    on = s.onset - backtest.TRAIN_BATCHES
    retr = r.retrains
    wasted = [b for b in retr if b < s.onset or not s.harmful]
    first = next((b for b in retr if b >= s.onset), None)
    return {"scenario": s.scenario, "trigger": r.trigger, "action": r.action,
            "error_total": float(np.sum(r.mae)), "error_after_onset": float(np.mean(r.mae[on:])),
            "retrains": len(retr), "wasted_retrains": len(wasted),
            "first_retrain_delay": None if first is None or not s.harmful else first - s.onset, **extra}


def task(args):
    scenario, seed, th, triggers, extras = args
    s = simulate.stream(scenario, seed)
    init = backtest.initial(s, seed)
    rows = []
    for t in triggers:
        r = backtest.run(s, t, th, action="refit", seed=seed, init=init)
        rows.append(_row(s, r, seed=seed, label_delay=2, study="triggers"))
    if not extras:
        return rows
    for t in ("error", "oracle", "schedule", "error_cp"):
        for a in ("retune", "warm"):
            r = backtest.run(s, t, th, action=a, seed=seed, init=init)
            rows.append(_row(s, r, seed=seed, label_delay=2, study="actions"))
    for d in (1, 5, 10):
        r = backtest.run(s, "error", th, action="refit", label_delay=d, seed=seed, init=init)
        rows.append(_row(s, r, seed=seed, label_delay=d, study="delay"))
    return rows


def main(argv=None):
    """No args: calibrate + everything. `--only t1,t2`: run just those triggers with the saved
    thresholds and replace their rows in the saved results."""
    argv = sys.argv[1:] if argv is None else argv
    RES.mkdir(exist_ok=True)
    only = argv[1].split(",") if argv[:1] == ["--only"] else None
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        if only:
            th = json.loads((RES / "thresholds.json").read_text())["thresholds"]
        else:
            th = calibrate.calibrate(path=RES / "thresholds.json", pool=pool)
        print(json.dumps(th, indent=1), flush=True)
        jobs = [(sc, seed, th, only or backtest.TRIGGERS, only is None)
                for sc in simulate.SCENARIOS for seed in TEST_SEEDS]
        rows = [r for rs in pool.map(task, jobs) for r in rs]
    df = pd.DataFrame(rows)
    if only:
        old = pd.read_csv(RES / "synthetic_runs.csv")
        old = old[~((old.study == "triggers") & old.trigger.isin(only))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RES / "synthetic_runs.csv", index=False)
    print(df[df.study == "triggers"].groupby(["scenario", "trigger"]).error_total.mean().unstack(0).round(1))


if __name__ == "__main__":
    sys.exit(main())
