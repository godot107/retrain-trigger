"""Weekly retraining monitor (AWS Lambda, run by EventBridge Scheduler).

S3 layout, per model (MODEL env var), written by the forecasting job:
    models/<model>/config.json           {"baseline_mae": .., "trigger": .., "threshold": .., "trained": "<week>"}
    models/<model>/reference.json        training-input sample (rows) for the classifier test
    models/<model>/weeks/<week>.json      {"inputs": [[..]], "pred": [..]}, then "actual": [..] once known
    models/<model>/state.json            written here: labelled errors since training
    models/<model>/reports/<week>.json   written here: one report per run

Each run looks at the newest week with predictions and the newest week with actuals, updates the state,
and publishes to SNS when the trigger fires. Retraining itself stays a human decision: the page says why
and suggests what to train on. Pure Python + boto3 (in the Lambda runtime): no layers.
"""
import json
import os

try:
    from retrain.aws import monitor_core as core       # local runs and tests
except ImportError:                                     # Lambda: monitor_core.py sits next to this file
    import monitor_core as core


class S3Store:
    def __init__(self, bucket):
        import boto3
        self.s3, self.bucket = boto3.client("s3"), bucket

    def get(self, key):
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, key, obj):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(obj).encode(),
                           ContentType="application/json")

    def list_weeks(self, prefix):
        out, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"].rsplit("/", 1)[1][:-5] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(out)
            token = r["NextContinuationToken"]


def run(store, model, publish=None):
    p = f"models/{model}/"
    cfg = store.get(p + "config.json")
    if cfg is None:
        return {"error": f"no config for {model}"}
    weeks = [w for w in store.list_weeks(p + "weeks/") if w > cfg.get("trained", "")]
    if not weeks:
        return {"note": "no weeks yet"}
    state = store.get(p + "state.json") or {"weeks": [], "mae": [], "trained": cfg.get("trained")}
    if state.get("trained") != cfg.get("trained"):      # a retrain happened: start over
        state = {"weeks": [], "mae": [], "trained": cfg.get("trained")}
    newest = store.get(p + f"weeks/{weeks[-1]}.json")
    labelled = None
    for w in reversed(weeks):                           # newest week whose actuals have arrived
        d = store.get(p + f"weeks/{w}.json")
        if d and d.get("actual"):
            if w not in state["weeks"]:
                labelled = (w, d["pred"], d["actual"])
            break
    reference = store.get(p + "reference.json") if cfg["trigger"] == "c2st" else None
    trained = state.get("trained")
    state, report = core.evaluate(state, weeks[-1], newest["pred"], labelled=labelled,
                                  inputs=newest.get("inputs"), reference=reference, cfg=cfg)
    state["trained"] = trained
    store.put(p + "state.json", state)
    store.put(p + f"reports/{weeks[-1]}.json", report)
    if report["fired"] and publish:
        publish(f"[retrain-trigger] {model}: {cfg['trigger']} fired on {weeks[-1]}",
                json.dumps(report, indent=2))
    return report


def lambda_handler(event, context):
    import boto3
    topic = os.environ.get("TOPIC_ARN", "")
    sns = boto3.client("sns") if topic else None

    def publish(subject, body):
        sns.publish(TopicArn=topic, Subject=subject[:100], Message=body)

    model = (event or {}).get("model") or os.environ["MODEL"]
    return run(S3Store(os.environ["BUCKET"]), model, publish if sns else None)
