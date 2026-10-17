"""Replay a real operator's weeks through the monitor, locally or against the deployed stack.

    python -m retrain.aws.replay local ERCO            # parity check vs the offline backtest, no AWS
    python -m retrain.aws.replay aws ERCO --weeks 12   # after deploy/deploy.sh: upload + invoke the Lambda

The model is the never-retrained 2019 model, so every week's scores can be compared with
backtest.score() on the same predictions: they must match (Page-Hinkley, error ratio exactly).
"""
import json
import sys
from pathlib import Path

import numpy as np

from retrain import backtest, eia
from retrain.aws import handler

ROOT = Path(__file__).resolve().parents[3]


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)

    def get(self, key):
        p = self.root / key
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key, obj):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj))

    def list_weeks(self, prefix):
        d = self.root / prefix
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def prepare(ba, trigger="page_hinkley", threshold=16.0):
    s = eia.stream(ba)
    st = backtest.initial(s)
    cfg = {"baseline_mae": st.val_mae, "trigger": trigger, "threshold": threshold,
           "trained": s.weeks[s.train_batches - 1].date().isoformat()}
    return s, st, cfg


def offline_scores(s, st, trigger, label_delay=1):
    """What backtest.score gives for the never-retrained model, week by week."""
    mae_hist, out = {}, {}
    for b in range(s.train_batches, len(s.X)):
        pred = st.model.predict(s.X[b])
        mae_hist[b] = float(np.mean(np.abs(pred - s.y[b])))
        out[s.weeks[b].date().isoformat()] = backtest.score(trigger, st, s, b, pred, mae_hist, label_delay)
    return out


def local(ba, trigger="page_hinkley", workdir=None):
    s, st, cfg = prepare(ba, trigger)
    store = LocalStore(workdir or ROOT / "build" / "replay" / ba)
    m = f"models/{ba}/"
    store.put(m + "config.json", cfg)
    for k in ("state.json",):
        p = store.root / m / k
        if p.exists():
            p.unlink()
    lambda_scores = {}
    for b in range(s.train_batches, len(s.X)):
        w = s.weeks[b].date().isoformat()
        store.put(m + f"weeks/{w}.json", {"pred": st.model.predict(s.X[b]).tolist()})
        if b - 1 >= s.train_batches:                   # last week's actuals arrive this week
            pw = s.weeks[b - 1].date().isoformat()
            d = store.get(m + f"weeks/{pw}.json")
            d["actual"] = s.y[b - 1].tolist()
            store.put(m + f"weeks/{pw}.json", d)
        lambda_scores[w] = handler.run(store, ba)["score"]
    off = offline_scores(s, st, trigger)
    diff = max(abs(lambda_scores[w] - off[w]) for w in off)
    first = lambda d: next((w for w, v in d.items() if v > cfg["threshold"]), None)  # noqa: E731
    return {"operator": ba, "trigger": trigger, "weeks": len(off), "max_abs_score_diff": diff,
            "first_alarm_lambda": first(lambda_scores), "first_alarm_offline": first(off)}


def aws(ba, n_weeks=12, trigger="page_hinkley"):
    """Upload the first n production weeks to the deployed bucket and invoke the Lambda once per week."""

    import boto3
    stack = "retrain-trigger-app"
    outputs = {o["OutputKey"]: o["OutputValue"] for o in
               boto3.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]}
    store, lam = handler.S3Store(outputs["BucketName"]), boto3.client("lambda")
    s, st, cfg = prepare(ba, trigger)
    m = f"models/{ba}/"
    store.put(m + "config.json", cfg)
    reports = []
    for b in range(s.train_batches, s.train_batches + n_weeks):
        w = s.weeks[b].date().isoformat()
        store.put(m + f"weeks/{w}.json", {"pred": st.model.predict(s.X[b]).tolist()})
        if b > s.train_batches:
            pw = s.weeks[b - 1].date().isoformat()
            d = store.get(m + f"weeks/{pw}.json")
            d["actual"] = s.y[b - 1].tolist()
            store.put(m + f"weeks/{pw}.json", d)
        r = lam.invoke(FunctionName=outputs["FunctionName"], Payload=json.dumps({"model": ba}).encode())
        reports.append(json.loads(r["Payload"].read()))
    return reports


if __name__ == "__main__":
    mode, ba = sys.argv[1], sys.argv[2]
    if mode == "local":
        print(json.dumps(local(ba), indent=2))
    else:
        n = int(sys.argv[sys.argv.index("--weeks") + 1]) if "--weeks" in sys.argv else 12
        for r in aws(ba, n):
            print(json.dumps(r))
