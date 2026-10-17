# retrain-trigger

When should a data scientist retrain a model? Retraining triggers (schedule, input PSI/KS, prediction
PSI, performance with delayed labels, oracle) compared by total cost = error + retrain cost, on synthetic
streams with known drift and on EIA-930 day-ahead load forecasting. Also: refit vs full re-tune vs warm
start. Trello #104 https://trello.com/c/N2cwKk6P. Backlog: Project Checklist #4 item 12 ("Drift Detection"); partner-gateway (#96) covered the
detection side (per-partner PSI) and found drift != harm; this project is the retraining decision.
Started 2026-10-17. Plan: docs/PLAN.md. Open work: TODO.md.

## Run

```bash
source .venv/bin/activate        # own venv, CPU only
pytest -q                         # ~10 s when the machine is idle
python -m retrain.experiment      # calibrate + synthetic study, 6 scenarios x 30 seeds (~8 min, 11 procs)
python -m retrain.experiment --only error_cp,page_hinkley_cp   # rerun some triggers with saved thresholds
python -m retrain.realdata        # EIA-930, 12 operators x 34 trigger/threshold runs (~1 h)
python -m retrain.realdata --extra | --seasonal | --cpw    # appended passes (wider grids, seasonal PSI, weighted cp)
python -m retrain.realdata --weather   # weather model -> results/real_runs_weather.json (~30 min)
python notebooks/build.py && for n in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $n; done
```

## Layout / decisions

- `simulate.py`: 6 scenarios with known onset (batch 40 of 80) and known harm; the answer key.
- `detectors.py`: scores for PSI, KS, classifier two-sample (c2st), prediction PSI, error ratio, Page-Hinkley.
- `calibrate.py`: thresholds set to the same false-alarm rate (1 in 300 batches) on 50 drift-free
  calibration streams (seeds 1000-1049, never tested on). PSI>0.2 kept as an uncalibrated rule row.
- `models.py`: HGB tuned by 12-candidate random search; actions refit / retune / warm.
- `backtest.py`: walk-forward; labels `label_delay` batches late; retrain on last `window` labelled
  batches; `*_cp` triggers retrain only on data since the Page-Hinkley change-point estimate.
  Retrain price applied afterwards (triggers don't know it).
- Oracle trains on post-onset data only, once (a reference, not a bound: gradual drift needs more).
- `eia.py` / `realdata.py`: day-ahead load forecasting (calendar + lags, no weather) per operator,
  weekly batches; thresholds chosen on 2020-21, scored on 2022-25; retrain price = k weeks of normal error.
- Real data: EIA-930 parquet from projects/grid-mining (`data/processed/balance.parquet`); don't
  re-download. Real events: COVID 2020, Uri Feb 2021, Elliott Dec 2022, ERCOT load growth.
- Report over many seeds with intervals.

## Gotchas

- **scikit-learn HGB warm_start re-bins features on the new data but evaluates old trees on the new
  bins while fitting** (read in gradient_boosting.py): after a shift, warm-started models are 4-5x
  worse. Error grows with shift (0.07 / 0.96 / 2.58 vs 0.01 fresh at shift 0 / 1 / 4).
- `models.tune` returns (params, model, mae); `models.retrain` returns (model, params, mae).
- A shared initial State is deep-copied per run (warm start mutates the model).
- Parallel runs: OMP_NUM_THREADS=1 per process (set in experiment.py / realdata.py).
- Real data: calendar features are excluded from input-drift checks (`drift_cols`): they "drift" by design.
- Weekly PSI on day-level features is structurally huge (7 distinct values per week can't fill 10 bins).
- realdata threshold keys: strip `_cpw` before `_cp` (a missed suffix once left `_cpw` thresholds at 1e9).
- `weather.py`: Open-Meteo archive, observed temp as perfect forecast; cached in data/weather/.
