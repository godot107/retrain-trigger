# Plan

**Question:** when should a data scientist be told to retrain a model, and should a retrain just refit
on fresh data or reset the hyperparameters too?

Monitoring tools mostly answer "has the input data drifted?". That is not the question the business
pays for. partner-gateway (Trello #96) already showed the gap: a new market set off its PSI drift alarm
while the model coped fine. This project measures **retraining triggers by what they cost and what
they save**, against drift whose timing and harm are known.

## What is compared

**Triggers**

| Trigger | Needs labels? | Watches |
|---|---|---|
| never | | (baseline) |
| fixed schedule (every N batches) | | the calendar |
| PSI on inputs (> 0.2 rule of thumb) | no | each feature's distribution |
| KS test on inputs (Bonferroni) | no | each feature's distribution |
| PSI on predictions | no | the model's outputs |
| performance (rolling error vs training-time error) | yes, after a delay | what actually matters |
| oracle (knows when the change began) | | reference point |

Later additions: MMD / classifier two-sample test (multivariate input drift), ADWIN / Page-Hinkley on
the error stream, and a combined rule ("input drift raises a flag, performance confirms").

**Retrain actions** (the "reset weights and hyperparameters" part)
1. refit, same hyperparameters, on a recent window;
2. full re-tune (hyperparameter search) on a recent window;
3. warm start / continue training from the old model (where the model type allows it).
Measured question: which kinds of drift need a re-tune, and which only need fresh data?

**Scoring:** total cost = sum of prediction error + (cost of one retrain × number of retrains), shown
across a range of retrain costs, because the right trigger depends on how expensive a retrain is.
Plus detection delay, false alarms (alarms with no harmful drift), and missed drifts.

## Data

1. **Synthetic streams (answer key):** no drift; harmless input drift; harmful input drift (inputs
   move where the model never learned); sudden concept drift (same inputs, new meaning); gradual
   concept drift. `src/retrain/simulate.py`.
2. **Real: day-ahead electricity demand forecasting** on EIA-930 hourly data, 2019-2025, 67 grid
   operators (already downloaded by the grid-mining project). Train on 2019, walk forward week by
   week. Real dated changes: COVID's 2020 shift in the daily load shape, Winter Storm Uri (Feb 2021),
   Winter Storm Elliott (Dec 2022), ERCOT's fast load growth. Actual demand arrives the next day, so
   performance triggers are realistic; label delay can be stretched to test slower feedback.
   Energy-industry framing, matching the portfolio's direction.
3. Optional: Folktables (US Census income by year and state) for a tabular classification case.

## Stages

1. Synthetic: all triggers x scenarios, cost curves over retrain cost, label-delay sensitivity.
2. Retrain actions: refit vs re-tune vs warm start, per scenario.
3. EIA-930 load forecasting backtest, with the real events as checkpoints.
4. Docs: README + blog + executed notebooks; a one-page "which trigger when" decision chart.
5. Optional AWS leg: the winning trigger as a scheduled Lambda over a model's logged predictions and
   delayed labels, compared with SageMaker Model Monitor's built-in drift checks (cost-checked first,
   torn down the same session).

## Day-one result (synthetic, 1 seed, retrain cost = 5)

| Trigger | Concept drift (sudden) | Harmful input drift | Harmless input drift | No drift |
|---|---|---|---|---|
| never | 105.9 | 71.1 | 26.1 | 25.8 |
| schedule (every 10) | 74.9 | 66.4 | 51.8 | 51.7 |
| PSI on inputs | 105.9 (never fires) | 51.7 | 36.8 (2 wasted retrains) | 25.8 |
| performance | 62.6 | 53.9 | **26.1** (no retrain) | **25.8** |
| oracle | 45.2 | 43.6 | 26.1 | 25.8 |

Total cost, lower is better. Input-drift checks are blind to concept drift and pay for retrains that
don't help; the performance trigger is never the worst and never wastes a retrain here. One seed:
the real run repeats over many seeds with intervals.
