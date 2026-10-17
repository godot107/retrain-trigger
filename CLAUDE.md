# retrain-trigger

When should a data scientist retrain a model? Retraining triggers (schedule, input PSI/KS, prediction
PSI, performance with delayed labels, oracle) compared by total cost = error + retrain cost, on synthetic
streams with known drift and on EIA-930 day-ahead load forecasting. Also: refit vs full re-tune vs warm
start. Backlog: Project Checklist #4 item 12 ("Drift Detection"); partner-gateway (#96) covered the
detection side (per-partner PSI) and found drift != harm; this project is the retraining decision.
Started 2026-10-17. Plan: docs/PLAN.md. Open work: TODO.md.

## Run

```bash
source .venv/bin/activate        # own venv, CPU only
pytest -q
```

## Layout / decisions

- `simulate.py`: scenarios with known onset (batch 40 of 80) and known harm; the answer key.
- `detectors.py`: PSI (>0.2), KS (Bonferroni p<0.01), prediction PSI, error ratio (>1.25 x training-time MAE).
- `backtest.py`: walk-forward; labels arrive `label_delay` batches late; retrain = HistGradientBoosting
  on the last `window` labelled batches; cooldown between retrains; cost = sum(batch MAE) + retrain_cost x retrains.
- Oracle trains on post-onset data only, once (a reference, not a bound: gradual drift needs more).
- Real data: EIA-930 parquet from projects/grid-mining (`data/processed/balance.parquet`); don't
  re-download. Real events: COVID 2020, Uri Feb 2021, Elliott Dec 2022, ERCOT load growth.
- Report over many seeds with intervals; day-one table is 1 seed.
