# AWS costs (monitor leg)

Not deployed yet. Everything in `infra/app.yaml` is pay-per-use; nothing bills by the hour.

| Resource | Usage | Expected cost |
|---|---|---|
| Lambda (arm64, 512 MB) | 1 run/week per model, ~10-60 s | ~1-2 GB-s per run: effectively $0 (free tier) |
| EventBridge Scheduler | 4-5 invocations/month | free tier (14 M/month) |
| S3 data bucket | a few MB of predictions/actuals; expires after 400 days | < $0.01/month |
| S3 code bucket | one small zip | < $0.01/month |
| SNS email | a page every few months | free tier |
| CloudWatch Logs | a few KB/week, 14-day retention | ~$0 |
| Budget (cost.yaml) | $5/month alert on `project=retrain-trigger` | free (notifications only) |

Expected total: **cents per month** while deployed. List prices from memory of the AWS pricing pages,
not re-verified this session; check before relying on them. CloudBurn (`cloudburn scan infra`) reports
two medium findings (S3 storage-class lifecycle rule CLDBRN-AWS-S3-2), not applicable at this size.

## Compared with SageMaker Model Monitor

| | This monitor | SageMaker Model Monitor |
|---|---|---|
| What it watches | model error from late labels (Page-Hinkley, error ratio) or a multivariate classifier test | data quality (per-feature statistics vs a baseline), model quality (needs ground truth merged in), bias, explainability |
| Data in | whatever the forecasting job writes to S3 | endpoint data capture (real-time endpoints; serverless endpoints have no data capture) or batch transform output |
| Compute | a Lambda run per week | a processing job per schedule run, billed per instance-hour of the chosen instance type |
| Always-on cost | none | none for the monitor itself, but it presumes a hosted endpoint, which bills per hour |
| Fit for this project | the forecaster is a batch job, not an endpoint | built for hosted endpoints |

Model Monitor's per-feature data-quality checks are the kind of input-drift alarm this project found to be
noisy on weather-driven data; its model-quality monitor is the analogue of the performance trigger.
